"""
data.py
-------
Data loading and input pipelines for the HAM10000-based multimodal models.

Responsibilities
================
* Read ``HAM10000_metadata.csv`` and locate each image on disk (HAM ships the
  images across two folders).
* Derive the label column for whichever stage we're training:
    - ``stage1`` : binary benign vs malignant over ALL 7 diagnoses
    - ``benign`` : 3-way {nv, bkl, vasc}
    - ``malignant`` : 3-way {mel, bcc, akiec}
* Produce stratified train/val/test splits (seed 2024, matching the original).
* Build ``tf.data`` pipelines that yield ``((image, metadata), label)`` for the
  multimodal models or ``(image, label)`` for the image-only baseline, with
  on-the-fly augmentation on the training split.
* Compute balanced ``class_weight`` dicts so we can train on the *full* data
  instead of undersampling (the key methodological fix over the original).

Diagnosis groupings (HAM10000 `dx` codes)
==========================================
    malignant : mel  (melanoma), bcc (basal cell carcinoma),
                akiec (actinic keratoses / intraepithelial carcinoma)
    benign    : nv   (melanocytic nevi), bkl (benign keratosis),
                vasc (vascular), df (dermatofibroma)

Stage 1 maps all 7 -> {benign, malignant} (uses every image). The sub-stages
follow the original project: benign types = {nv, bkl, vasc}, malignant types =
{mel, bcc, akiec}. `df` is grouped benign at stage 1 but not sub-classified
(too few samples, ~115) — a documented limitation.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:  # works both as a package (src.data) and with src/ on sys.path
    from .metadata import MetadataProcessor
except ImportError:  # pragma: no cover - notebook import path
    from metadata import MetadataProcessor

SEED = 2024
IMAGE_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE

MALIGNANT = ["mel", "bcc", "akiec"]
BENIGN = ["nv", "bkl", "vasc"]
BENIGN_EXTRA = ["df"]  # benign but not sub-classified

# Class list per stage (index order is the label order used everywhere).
STAGE_CLASSES: dict[str, list[str]] = {
    "stage1": ["benign", "malignant"],
    "benign": BENIGN,
    "malignant": MALIGNANT,
}


# ---------------------------------------------------------------- image indexing
def build_image_index(image_dirs: list[str]) -> dict[str, str]:
    """Map ``image_id`` -> absolute path by scanning HAM image folders.

    HAM10000 stores images as ``<image_id>.jpg`` split across
    ``HAM10000_images_part_1`` and ``HAM10000_images_part_2``.
    """
    index: dict[str, str] = {}
    for d in image_dirs:
        for path in glob.glob(os.path.join(d, "*.jpg")):
            image_id = os.path.splitext(os.path.basename(path))[0]
            index[image_id] = path
    return index


# --------------------------------------------------------------------- dataframe
def load_ham_dataframe(
    metadata_csv: str,
    image_dirs: list[str],
) -> pd.DataFrame:
    """Load HAM metadata, attach image paths, and derive the binary label.

    Returns a dataframe with at least:
        image_id, image_path, dx, age, sex, localization, binary_label
    Rows whose image file is missing on disk are dropped (with a warning).
    """
    df = pd.read_csv(metadata_csv)
    index = build_image_index(image_dirs)
    df["image_path"] = df["image_id"].map(index)

    missing = int(df["image_path"].isna().sum())
    if missing:
        print(f"[data] WARNING: {missing} images not found on disk; dropping them.")
        df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    df["binary_label"] = np.where(df["dx"].isin(MALIGNANT), "malignant", "benign")
    return df


def filter_for_stage(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Slice + label a dataframe for a given stage.

    Adds a ``label`` (string) and ``label_idx`` (int) column. For ``stage1``
    every row is kept; for the sub-stages only the relevant diagnoses remain.
    """
    if stage not in STAGE_CLASSES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {list(STAGE_CLASSES)}")

    if stage == "stage1":
        out = df.copy()
        out["label"] = out["binary_label"]
    else:
        classes = STAGE_CLASSES[stage]
        out = df[df["dx"].isin(classes)].copy()
        out["label"] = out["dx"]

    class_to_idx = {c: i for i, c in enumerate(STAGE_CLASSES[stage])}
    out["label_idx"] = out["label"].map(class_to_idx).astype(int)
    return out.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split on ``label`` (seed 2024).

    Mirrors the original project: first peel off the test set, then carve a
    validation set out of the remaining training rows.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=SEED, shuffle=True, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=SEED,
        shuffle=True,
        stratify=train_df["label"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_class_weights(df: pd.DataFrame, stage: str) -> dict[int, float]:
    """Balanced class weights ``{label_idx: weight}`` for ``model.fit``."""
    classes = np.arange(len(STAGE_CLASSES[stage]))
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=df["label_idx"].to_numpy()
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


def assign_global_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = SEED,
    stratify_col: str = "dx",
) -> pd.DataFrame:
    """Add a single global ``split`` column ('train' / 'val' / 'test').

    A *shared* partition (stratified on the 7-way diagnosis) is assigned once,
    and every stage then derives its own train/val/test slices by filtering on
    this column. This guarantees that an image used to train any sub-stage
    never appears in the end-to-end test set — the split that ``pipeline.py`` is
    finally evaluated on in notebook 06. Deterministic given ``seed``.
    """
    idx = np.arange(len(df))
    strat = df[stratify_col].to_numpy()
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, shuffle=True, stratify=strat
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=strat[train_idx],
    )
    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    out = df.copy()
    out["split"] = split
    return out


# ------------------------------------------------------------------- augmentation
def build_augmenter(seed: int = SEED) -> tf.keras.Sequential:
    """On-the-fly augmentation kept in the [0, 255] range EfficientNet expects.

    The original notebooks had augmentation commented out and instead relied on
    a pre-augmented dataset; doing it live here is cleaner and reproducible.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            tf.keras.layers.RandomRotation(0.15, seed=seed),
            tf.keras.layers.RandomZoom(0.1, seed=seed),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 255.0), seed=seed),
            tf.keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="augmenter",
    )


def _decode_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    """Read a JPEG/PNG and return float32 pixels in [0, 255] (no rescale).

    ``decode_image`` (rather than ``decode_jpeg``) so the same pipeline can read
    the original HAM JPEGs *and* the PNG synthetic samples produced by the
    diffusion notebook.
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size])
    return tf.cast(img, tf.float32)  # stays in [0, 255] for EfficientNet


# ----------------------------------------------------------------------- pipeline
def make_dataset(
    df: pd.DataFrame,
    stage: str,
    meta_proc: MetadataProcessor | None,
    multimodal: bool,
    batch_size: int = 32,
    image_size: int = IMAGE_SIZE,
    training: bool = False,
    augment: bool = False,
    shuffle_buffer: int = 2048,
) -> tf.data.Dataset:
    """Build a ``tf.data`` pipeline for one split.

    Yields ``((image, metadata), label_onehot)`` when ``multimodal`` else
    ``(image, label_onehot)``. ``meta_proc`` must be a *fitted*
    ``MetadataProcessor`` when ``multimodal`` is True.
    """
    n_classes = len(STAGE_CLASSES[stage])
    paths = df["image_path"].to_numpy()
    labels = tf.keras.utils.to_categorical(df["label_idx"].to_numpy(), n_classes)

    if multimodal:
        if meta_proc is None:
            raise ValueError("meta_proc (fitted MetadataProcessor) required when multimodal=True")
        meta = meta_proc.transform(df)
        ds = tf.data.Dataset.from_tensor_slices((paths, meta, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(min(shuffle_buffer, len(df)), seed=SEED, reshuffle_each_iteration=True)

    augmenter = build_augmenter() if (training and augment) else None

    def _map(*args):
        if multimodal:
            path, meta_vec, label = args
        else:
            path, label = args
        img = _decode_image(path, image_size)
        if augmenter is not None:
            # add/remove batch dim so the augmentation layers run per-image
            img = augmenter(tf.expand_dims(img, 0), training=True)[0]
        if multimodal:
            return (img, meta_vec), label
        return img, label

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

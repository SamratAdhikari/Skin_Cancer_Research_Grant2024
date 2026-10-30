"""Unit tests for src/data.py — loading, splits, class weights, tf.data pipeline."""
import numpy as np
import tensorflow as tf
from PIL import Image

import data as D


def test_build_image_index(mock_ham):
    index = D.build_image_index(mock_ham["dirs"])
    assert len(index) == mock_ham["n"]
    assert all(p.endswith(".jpg") for p in index.values())


def test_load_dataframe_has_paths_and_binary_label(loaded_df):
    assert loaded_df["image_path"].notna().all()
    assert set(loaded_df["binary_label"].unique()) <= {"benign", "malignant"}
    # malignant dx codes must map to the malignant binary label.
    mal = loaded_df[loaded_df["dx"].isin(D.MALIGNANT)]
    assert (mal["binary_label"] == "malignant").all()


def test_filter_for_stage_stage1_keeps_all(loaded_df):
    s1 = D.filter_for_stage(loaded_df, "stage1")
    assert len(s1) == len(loaded_df)
    assert set(s1["label_idx"].unique()) <= {0, 1}


def test_filter_for_stage_subsets_and_indexes(loaded_df):
    mal = D.filter_for_stage(loaded_df, "malignant")
    assert set(mal["dx"].unique()) <= set(D.MALIGNANT)
    # label_idx follows STAGE_CLASSES order exactly.
    class_to_idx = {c: i for i, c in enumerate(D.STAGE_CLASSES["malignant"])}
    for _, row in mal.iterrows():
        assert row["label_idx"] == class_to_idx[row["dx"]]


def test_global_split_is_partition_and_deterministic(loaded_df):
    counts = loaded_df["split"].value_counts()
    for name in ("train", "val", "test"):
        assert counts.get(name, 0) > 0
    # Same seed -> identical assignment.
    again = D.assign_global_split(loaded_df.drop(columns=["split"]))
    assert (again["split"].to_numpy() == loaded_df["split"].to_numpy()).all()


def test_class_weights_keys_and_positive(loaded_df):
    s1 = D.filter_for_stage(loaded_df, "stage1")
    w = D.compute_class_weights(s1, "stage1")
    assert set(w.keys()) == set(range(len(D.STAGE_CLASSES["stage1"])))
    assert all(v > 0 for v in w.values())


def test_make_dataset_image_only_shapes(loaded_df):
    s1 = D.filter_for_stage(loaded_df, "stage1")
    ds = D.make_dataset(s1, "stage1", None, multimodal=False, batch_size=8)
    x, y = next(iter(ds))
    assert x.shape[1:] == (D.IMAGE_SIZE, D.IMAGE_SIZE, 3)
    assert y.shape[-1] == 2
    # EfficientNet range: pixels must stay in [0, 255], not rescaled to [0, 1].
    assert float(tf.reduce_max(x)) > 1.5


def test_make_dataset_multimodal_shapes(loaded_df, meta_proc):
    s1 = D.filter_for_stage(loaded_df, "stage1")
    ds = D.make_dataset(s1, "stage1", meta_proc, multimodal=True, batch_size=8)
    inputs, y = next(iter(ds))
    img, meta = inputs
    assert img.shape[1:] == (D.IMAGE_SIZE, D.IMAGE_SIZE, 3)
    assert meta.shape[-1] == meta_proc.feature_dim
    assert y.shape[-1] == 2


def test_decode_handles_png(tmp_path):
    # The pipeline must read synthetic PNGs (diffusion output), not only JPEGs.
    p = tmp_path / "synthetic_0001.png"
    Image.fromarray(np.zeros((30, 30, 3), "uint8")).save(p)
    img = D._decode_image(tf.constant(str(p)), D.IMAGE_SIZE)
    assert img.shape == (D.IMAGE_SIZE, D.IMAGE_SIZE, 3)
    assert img.dtype == tf.float32

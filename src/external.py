"""
external.py
-----------
External (cross-dataset) validation of the trained HAM10000 models.

The single biggest weakness of a HAM10000-only study is that it never shows the
model generalises beyond HAM10000. This module evaluates the *already-trained*
models (no retraining) on a **different** dermatoscopy dataset, so the numbers it
produces are an honest estimate of out-of-distribution performance — exactly what
a reviewer asks for first.

Default external set: **ISIC 2019** (Kaggle / ISIC Archive). Two things make a
naive ISIC-2019 evaluation *wrong*, and this module handles both explicitly:

1. **Label-space mismatch.** ISIC 2019 has eight diagnoses; two of them have no
   counterpart in our taxonomy. We map the six shared classes onto our `dx`
   codes, keep squamous-cell carcinoma (`SCC`) as *malignant* for the Stage-1
   (benign/malignant) evaluation but exclude it from the malignant-subtype head
   (there is no `scc` subtype), and drop the `UNK` bucket. `DF` stays benign but
   is not sub-classified — the same limitation the HAM pipeline already has.

2. **Train/test overlap (leakage).** ISIC 2019 *contains* the HAM10000 images.
   Because both datasets name images with the same ISIC IDs, we remove any ISIC
   image whose ID appears in the model's training/split manifest before scoring.
   `remove_training_overlap` does this exactly, not heuristically. For a dataset
   disjoint from HAM (BCN20000, PH2, Derm7pt) the removal is simply a no-op.

Anatomical-site vocabularies also differ across datasets (ISIC's
`anterior torso` vs. HAM's `chest/trunk/back`); `ISIC2019_SITE_MAP` provides an
approximate, documented crosswalk so the metadata branch stays meaningful. Sites
we cannot map fall through to the `MetadataProcessor`'s `unknown` bucket rather
than crashing — a graceful degradation, and itself a finding about metadata
portability.

Everything reuses the tested `src/` modules (`data`, `metadata`, `evaluate`,
`pipeline`); TensorFlow is imported lazily inside the scoring functions so the
pure-pandas dataframe/label/dedup helpers stay import-light and unit-testable on
CPU.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:  # works both as a package (src.external) and with src/ on sys.path
    from . import data as D
    from . import evaluate as ev
except ImportError:  # pragma: no cover - notebook/app import path
    import data as D
    import evaluate as ev


# --------------------------------------------------------------- label crosswalk
# ISIC-2019 diagnosis code -> (our `dx` code | None, binary label | None).
#   * shared six classes map onto our HAM codes
#   * SCC (squamous-cell carcinoma) is malignant but has no subtype head -> `scc`
#     (a non-HAM code, so `filter_for_stage('malignant')` excludes it while
#     Stage 1 still counts it as malignant)
#   * DF stays benign, not sub-classified (matches the HAM pipeline)
#   * UNK -> (None, None), dropped
ISIC2019_LABELS: dict[str, tuple[str | None, str | None]] = {
    "MEL": ("mel", "malignant"),
    "NV": ("nv", "benign"),
    "BCC": ("bcc", "malignant"),
    "AK": ("akiec", "malignant"),  # actinic keratosis ≈ HAM `akiec`
    "BKL": ("bkl", "benign"),
    "DF": ("df", "benign"),
    "VASC": ("vasc", "benign"),
    "SCC": ("scc", "malignant"),  # malignant, no matching subtype head
    "UNK": (None, None),
}

ISIC2019_DIAGNOSIS_COLUMNS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

# ISIC anatomical-site vocabulary -> nearest HAM `localization` value.
# Approximate by construction (documented in the report); unmapped sites fall to
# the metadata processor's `unknown` bucket.
ISIC2019_SITE_MAP: dict[str, str] = {
    "anterior torso": "trunk",
    "posterior torso": "back",
    "lateral torso": "trunk",
    "upper extremity": "upper extremity",
    "lower extremity": "lower extremity",
    "head/neck": "face",
    "palms/soles": "acral",
    "oral/genital": "genital",
}

# Diagnoses the hierarchy can actually name at the subtype stage (df/scc/unk
# have no subtype head and are excluded from end-to-end subtype accuracy).
SUBTYPES = ["nv", "bkl", "vasc", "mel", "bcc", "akiec"]


def map_label(code: str) -> tuple[str | None, str | None]:
    """Map one ISIC-2019 diagnosis code to ``(dx, binary_label)``."""
    return ISIC2019_LABELS.get(str(code).strip().upper(), (None, None))


def _map_site(value):
    """Crosswalk one ISIC site string to a HAM `localization`; pass through else."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    return ISIC2019_SITE_MAP.get(str(value).strip().lower(), value)


# ------------------------------------------------------------------- loaders
def load_isic2019_dataframe(
    ground_truth_csv: str,
    metadata_csv: str | None,
    image_dir: str,
    image_ext: str = ".jpg",
) -> pd.DataFrame:
    """Build a HAM-shaped dataframe from the ISIC-2019 training release.

    Expects the official files:
      * ``ISIC_2019_Training_GroundTruth.csv`` — one-hot diagnosis columns
        (``image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK``)
      * ``ISIC_2019_Training_Metadata.csv`` — ``image, age_approx,
        anatom_site_general, lesion_id, sex`` (optional but recommended)

    Returns a dataframe with the same columns the HAM pipeline uses
    (``image_id, image_path, dx, binary_label, age, sex, localization`` plus
    ``orig_label``), so it drops straight into ``data.filter_for_stage`` /
    ``data.make_dataset`` and ``metadata.MetadataProcessor.transform``. Rows with
    an unmappable diagnosis (``UNK``) or a missing image file are dropped.
    """
    gt = pd.read_csv(ground_truth_csv)
    dx_cols = [c for c in ISIC2019_DIAGNOSIS_COLUMNS if c in gt.columns]
    if not dx_cols:
        raise ValueError(
            f"No ISIC-2019 diagnosis columns found in {ground_truth_csv}; "
            f"expected some of {ISIC2019_DIAGNOSIS_COLUMNS}"
        )
    gt = gt.rename(columns={gt.columns[0]: "image"})
    gt["orig_label"] = gt[dx_cols].astype(float).idxmax(axis=1)
    df = gt[["image", "orig_label"]].copy()

    if metadata_csv and os.path.exists(metadata_csv):
        meta = pd.read_csv(metadata_csv)
        meta = meta.rename(columns={meta.columns[0]: "image"})
        anatom = [c for c in meta.columns if c.startswith("anatom_site")]
        if anatom:
            meta = meta.rename(columns={anatom[0]: "localization"})
        meta = meta.rename(columns={"age_approx": "age"})
        keep = ["image"] + [c for c in ("age", "sex", "localization") if c in meta.columns]
        df = df.merge(meta[keep], on="image", how="left")

    for col in ("age", "sex", "localization"):
        if col not in df.columns:
            df[col] = np.nan

    df["localization"] = df["localization"].map(_map_site)
    mapped = df["orig_label"].map(map_label)
    df["dx"] = [t[0] for t in mapped]
    df["binary_label"] = [t[1] for t in mapped]
    df = df.dropna(subset=["dx", "binary_label"]).reset_index(drop=True)

    df = df.rename(columns={"image": "image_id"})
    df["image_path"] = df["image_id"].map(lambda i: os.path.join(image_dir, f"{i}{image_ext}"))
    exists = df["image_path"].map(os.path.exists)
    if (~exists).any():
        print(f"[external] {int((~exists).sum())} ISIC images not found on disk; dropping them.")
        df = df[exists].reset_index(drop=True)
    return df


def load_training_image_ids(*csv_paths: str) -> set[str]:
    """Collect the image IDs the models were trained on (for the leakage guard).

    Reads any of ``models/ham_split.csv`` / ``HAM10000_metadata.csv``; uses the
    ``image_id`` column if present, else derives IDs from ``image_path`` basenames.
    """
    ids: set[str] = set()
    for path in csv_paths:
        if not path or not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        if "image_id" in d.columns:
            ids.update(d["image_id"].astype(str))
        elif "image_path" in d.columns:
            ids.update(
                d["image_path"].map(lambda x: os.path.splitext(os.path.basename(str(x)))[0])
            )
    return ids


def remove_training_overlap(
    ext_df: pd.DataFrame, train_image_ids: set[str]
) -> tuple[pd.DataFrame, int]:
    """Drop external rows whose image ID was seen in training. Returns (df, n_removed).

    ISIC 2019 physically contains the HAM10000 images under the same ISIC IDs, so
    this exact set-difference is what keeps the external evaluation leakage-free.
    A no-op for datasets disjoint from HAM (BCN20000, PH2, ...).
    """
    if not train_image_ids or "image_id" not in ext_df.columns:
        return ext_df.reset_index(drop=True), 0
    ids = {str(i) for i in train_image_ids}
    keep = ~ext_df["image_id"].astype(str).isin(ids)
    n_removed = int((~keep).sum())
    return ext_df[keep].reset_index(drop=True), n_removed


def class_counts(ext_df: pd.DataFrame) -> dict[str, int]:
    """``dx`` -> count, for the dataset-provenance record."""
    return {str(k): int(v) for k, v in ext_df["dx"].value_counts().items()}


# --------------------------------------------------------------- per-stage eval
def evaluate_stage_external(
    ext_df: pd.DataFrame,
    stage: str,
    models_dir: str,
    proc,
    fig_dir: str | None = None,
    batch_size: int = 64,
    make_figures: bool = True,
) -> dict | None:
    """Score the trained baseline + multimodal models for one stage on external data.

    The whole external set is treated as a test set (no split). Returns a dict
    shaped like notebook 06's ``eval_stage`` output — ``{baseline, multimodal
    (+ece), mcnemar, n, classes}`` — with any model that is missing on disk simply
    omitted. Returns ``None`` if no external rows belong to this stage.
    """
    from tensorflow.keras.models import load_model

    classes = D.STAGE_CLASSES[stage]
    sdf = D.filter_for_stage(ext_df, stage)
    if len(sdf) == 0:
        return None
    y_true = sdf["label_idx"].to_numpy()
    out: dict = {"n": int(len(sdf)), "classes": classes}

    base_path = os.path.join(models_dir, f"{stage}_baseline.keras")
    mm_path = os.path.join(models_dir, f"{stage}_multimodal.keras")
    pb = pm = None

    if os.path.exists(mm_path):
        mm = load_model(mm_path)
        ds_m = D.make_dataset(sdf, stage, proc, multimodal=True,
                              batch_size=batch_size, training=False)
        pm = mm.predict(ds_m, verbose=0)
        m_mm = ev.summarize_predictions(y_true, pm, classes)
        m_mm["ece"] = ev.expected_calibration_error(y_true, pm)
        out["multimodal"] = m_mm
        if make_figures and fig_dir:
            ev.plot_confusion_matrix(
                y_true, pm.argmax(1), classes,
                os.path.join(fig_dir, f"ext_{stage}_confusion.png"),
                title=f"External · {stage} — confusion (multimodal)")
            ev.plot_roc_curves(y_true, pm, classes,
                               os.path.join(fig_dir, f"ext_{stage}_roc.png"))
            ev.plot_reliability(y_true, pm,
                                os.path.join(fig_dir, f"ext_{stage}_reliability.png"))

    if os.path.exists(base_path):
        base = load_model(base_path)
        ds_b = D.make_dataset(sdf, stage, None, multimodal=False,
                              batch_size=batch_size, training=False)
        pb = base.predict(ds_b, verbose=0)
        out["baseline"] = ev.summarize_predictions(y_true, pb, classes)

    if pb is not None and pm is not None:
        out["mcnemar"] = ev.mcnemar_test(y_true, pb.argmax(1), pm.argmax(1))
    return out


# --------------------------------------------------------------- end-to-end eval
def run_end_to_end_external(
    ext_df: pd.DataFrame,
    models_dir: str,
    max_n: int | None = None,
    seed: int = 2024,
    gradcam: bool = False,
) -> dict:
    """Run the full hierarchical pipeline over external images (Stage 1 -> Stage 2).

    Mirrors notebook 06's end-to-end block but on the external set. ``max_n`` caps
    the number of images actually scored (a random, seeded subsample) to bound
    runtime on large sets like ISIC 2019; the cap and the count used are returned
    so nothing is silently truncated. Subtype accuracy is measured only over rows
    whose true diagnosis is one of the six nameable subtypes (``df``/``scc``/``unk``
    have no subtype head).
    """
    from PIL import Image

    try:
        from .pipeline import SkinLesionPipeline, MALIGNANT_CLASSES
    except ImportError:  # pragma: no cover - notebook/app import path
        from pipeline import SkinLesionPipeline, MALIGNANT_CLASSES

    df = ext_df
    n_available = len(df)
    capped = max_n is not None and n_available > max_n
    if capped:
        df = df.sample(n=max_n, random_state=seed).reset_index(drop=True)

    pipe = SkinLesionPipeline.from_dir(models_dir)

    def _clean(v):
        return None if (v is None or pd.isna(v) or str(v) == "unknown") else v

    n = n_stage1_correct = 0
    mal_tp = mal_fn = 0
    n_sub = n_sub_correct = 0
    for _, row in df.iterrows():
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            continue
        res = pipe.predict(
            img, age=_clean(row.get("age")), sex=_clean(row.get("sex")),
            localization=_clean(row.get("localization")), gradcam=gradcam,
        )
        n += 1
        true_mal = row["binary_label"] == "malignant"
        n_stage1_correct += int(res.is_malignant == true_mal)
        if true_mal:
            mal_tp += int(res.is_malignant)
            mal_fn += int(not res.is_malignant)
        if row["dx"] in SUBTYPES:
            n_sub += 1
            n_sub_correct += int(res.subtype_label == row["dx"])

    return {
        "stage1_accuracy": n_stage1_correct / max(n, 1),
        "malignant_sensitivity": mal_tp / max(mal_tp + mal_fn, 1),
        "end_to_end_subtype_accuracy": n_sub_correct / max(n_sub, 1),
        "n_test": int(n),
        "n_available": int(n_available),
        "capped": bool(capped),
    }


# --------------------------------------------------------------- orchestrator
def run_external_validation(
    ext_df: pd.DataFrame,
    models_dir: str,
    proc,
    fig_dir: str | None,
    dataset_name: str,
    n_overlap_removed: int = 0,
    n_total: int | None = None,
    max_e2e: int | None = None,
    batch_size: int = 64,
) -> dict:
    """Full external-validation sweep -> the ``external`` block for results.json.

    Produces per-stage baseline-vs-multimodal metrics (+ figures), the end-to-end
    pipeline numbers, and a dataset-provenance record. Sub-stage entries are
    omitted when the external set has no rows for that stage.
    """
    if n_total is None:
        n_total = len(ext_df) + n_overlap_removed

    results: dict = {}
    s1 = evaluate_stage_external(ext_df, "stage1", models_dir, proc, fig_dir,
                                 batch_size=batch_size)
    if s1:
        results["stage1"] = s1
    for stage in ("benign", "malignant"):
        r = evaluate_stage_external(ext_df, stage, models_dir, proc, fig_dir,
                                    batch_size=batch_size)
        if r:
            results[stage] = r

    results["end_to_end"] = run_end_to_end_external(ext_df, models_dir, max_n=max_e2e)
    results["dataset"] = {
        "name": dataset_name,
        "n_total": int(n_total),
        "n_overlap_removed": int(n_overlap_removed),
        "n_evaluated": int(len(ext_df)),
        "class_counts": class_counts(ext_df),
    }
    return results

"""
Shared pytest fixtures for the src/ unit tests.

Everything the tests need is built here **once per session** (the EfficientNetB3
backbones are the slow part), then reused. A tiny synthetic HAM10000-shaped
dataset — real image files on disk plus a metadata CSV — lets us exercise the
data pipeline, metadata processor, models, Grad-CAM and the end-to-end pipeline
without any download or GPU.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# Make ``src/`` importable as top-level modules (data, model, metadata, ...).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Keep TensorFlow quiet during the test run.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@pytest.fixture(scope="session")
def mock_ham(tmp_path_factory):
    """A miniature HAM10000: images across two folders + a metadata CSV.

    Returns a dict with the CSV path and the two image directories, mirroring
    how the real dataset ships.
    """
    d = tmp_path_factory.mktemp("ham")
    img1, img2 = d / "part1", d / "part2"
    img1.mkdir()
    img2.mkdir()
    rng = np.random.RandomState(0)
    # Enough per class to survive a stratified 70/15/15 split on 7-way dx.
    per = {"nv": 16, "mel": 10, "bkl": 8, "bcc": 8, "akiec": 8, "vasc": 8, "df": 8}
    rows = []
    k = 0
    for dx, count in per.items():
        for _ in range(count):
            iid = f"ISIC_{k:06d}"
            k += 1
            folder = img1 if k % 2 else img2
            Image.fromarray(rng.randint(0, 255, (40, 50, 3), "uint8")).save(
                folder / f"{iid}.jpg"
            )
            rows.append(
                {
                    "image_id": iid,
                    "dx": dx,
                    "age": float(rng.randint(20, 85)) if rng.rand() > 0.1 else np.nan,
                    "sex": rng.choice(["male", "female"]),
                    "localization": rng.choice(["back", "face", "trunk", "abdomen"]),
                }
            )
    csv = d / "HAM10000_metadata.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return {"csv": str(csv), "dirs": [str(img1), str(img2)], "n": k}


@pytest.fixture(scope="session")
def loaded_df(mock_ham):
    """DataFrame with image paths, binary label, and a global split column."""
    import data as D

    df = D.load_ham_dataframe(mock_ham["csv"], mock_ham["dirs"])
    return D.assign_global_split(df)


@pytest.fixture(scope="session")
def meta_proc(loaded_df):
    """MetadataProcessor fitted on the training split only (no leakage)."""
    import metadata as M

    return M.MetadataProcessor().fit(loaded_df[loaded_df.split == "train"])


@pytest.fixture(scope="session")
def baseline2():
    """Compiled image-only baseline, 2 classes (built once)."""
    import model as Model

    return Model.compile_model(Model.build_image_only_baseline(2))


@pytest.fixture(scope="session")
def mm2(meta_proc):
    """Compiled multimodal model, 2 classes."""
    import model as Model

    return Model.compile_model(Model.build_multimodal_model(2, meta_proc.feature_dim))


@pytest.fixture(scope="session")
def mm3(meta_proc):
    """Compiled multimodal model, 3 classes (reused for both sub-stages)."""
    import model as Model

    return Model.compile_model(Model.build_multimodal_model(3, meta_proc.feature_dim))


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory, meta_proc, mm2, mm3):
    """A models/ dir the pipeline can load: 3 .keras files + preprocessing.json."""
    d = tmp_path_factory.mktemp("models")
    mm2.save(str(d / "stage1_multimodal.keras"))
    mm3.save(str(d / "benign_multimodal.keras"))
    mm3.save(str(d / "malignant_multimodal.keras"))
    meta_proc.save(str(d / "preprocessing.json"))
    return str(d)

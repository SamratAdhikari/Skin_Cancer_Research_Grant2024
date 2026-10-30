"""Unit tests for src/metadata.py — the patient-metadata preprocessor."""
import numpy as np
import pandas as pd
import pytest

import metadata as M


def test_fit_transform_shape(loaded_df, meta_proc):
    X = meta_proc.transform(loaded_df)
    assert X.shape == (len(loaded_df), meta_proc.feature_dim)
    assert X.dtype == np.float32
    # feature_dim = 1 (age) + |sex vocab| + |loc vocab|
    assert meta_proc.feature_dim == 1 + len(meta_proc.sex_categories) + len(
        meta_proc.loc_categories
    )
    assert len(meta_proc.feature_names) == meta_proc.feature_dim


def test_unknown_bucket_always_present(meta_proc):
    # Every categorical vocab must carry an explicit 'unknown' column.
    assert "unknown" in meta_proc.sex_categories
    assert "unknown" in meta_proc.loc_categories


def test_transform_one_missing_falls_back(meta_proc):
    # No metadata at all -> age imputed to median (z=0-ish) + all-unknown one-hots.
    v = meta_proc.transform_one(age=None, sex=None, localization=None)
    assert v.shape == (meta_proc.feature_dim,)
    # one-hot blocks each sum to 1 (exactly one bucket lit, here 'unknown').
    sex_block = v[1 : 1 + len(meta_proc.sex_categories)]
    loc_block = v[1 + len(meta_proc.sex_categories) :]
    assert sex_block.sum() == pytest.approx(1.0)
    assert loc_block.sum() == pytest.approx(1.0)


def test_unseen_localization_maps_to_unknown(meta_proc):
    v = meta_proc.transform_one(age=50, sex="male", localization="left_earlobe_xyz")
    loc_block = v[1 + len(meta_proc.sex_categories) :]
    unk_idx = meta_proc.loc_categories.index("unknown")
    assert loc_block[unk_idx] == pytest.approx(1.0)
    assert loc_block.sum() == pytest.approx(1.0)


def test_save_load_roundtrip(tmp_path, loaded_df, meta_proc):
    path = tmp_path / "preprocessing.json"
    meta_proc.save(str(path))
    reloaded = M.MetadataProcessor.load(str(path))
    # Identical vocab + identical transform output after a round-trip.
    assert reloaded.feature_dim == meta_proc.feature_dim
    np.testing.assert_allclose(
        reloaded.transform(loaded_df), meta_proc.transform(loaded_df), rtol=1e-6
    )


def test_missing_age_column_tolerated():
    # A metadata-poor dataset (no age column) must still fit/transform.
    df = pd.DataFrame({"sex": ["male", "female"], "localization": ["back", "face"]})
    proc = M.MetadataProcessor().fit(df)
    X = proc.transform(df)
    assert X.shape == (2, proc.feature_dim)
    assert np.isfinite(X).all()


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        M.MetadataProcessor().transform(pd.DataFrame({"age": [1]}))

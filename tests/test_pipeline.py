"""Unit tests for src/pipeline.py — end-to-end hierarchical routing."""
import numpy as np
import pytest
from PIL import Image

import pipeline as P


def _img():
    return Image.fromarray(
        np.random.randint(0, 255, (100, 120, 3), "uint8")
    )


def test_from_dir_loads_all(models_dir):
    pipe = P.SkinLesionPipeline.from_dir(models_dir)
    assert pipe.stage1_model is not None
    assert pipe.benign_model is not None
    assert pipe.malignant_model is not None
    assert pipe.meta_proc is not None


def test_predict_routes_to_matching_subtype(models_dir):
    pipe = P.SkinLesionPipeline.from_dir(models_dir)
    res = pipe.predict(_img(), age=55, sex="male", localization="back", gradcam=False)
    assert res.stage1_label in P.STAGE1_CLASSES
    assert sum(res.stage1_probs.values()) == pytest.approx(1.0, abs=1e-4)
    # The subtype must come from the class set the stage-1 label routes to.
    expected = P.MALIGNANT_CLASSES if res.is_malignant else P.BENIGN_CLASSES
    assert res.subtype_label in expected
    assert res.used_metadata is True


def test_predict_with_gradcam_overlay(models_dir):
    pipe = P.SkinLesionPipeline.from_dir(models_dir)
    res = pipe.predict(_img(), age=55, sex="male", localization="back", gradcam=True)
    # Overlay is optional (must never break inference) but if present is an image.
    if res.gradcam_overlay is not None:
        assert res.gradcam_overlay.shape == (P.IMAGE_SIZE, P.IMAGE_SIZE, 3)
        assert res.gradcam_overlay.dtype == np.uint8


def test_display_names(models_dir):
    pipe = P.SkinLesionPipeline.from_dir(models_dir)
    res = pipe.predict(_img(), gradcam=False)
    assert res.stage1_display in ("Benign", "Malignant")
    if res.subtype_label is not None:
        assert res.subtype_display == P.FULL_NAMES[res.subtype_label]


def test_graceful_without_substage(mm2, meta_proc):
    # Only a stage-1 model available: pipeline still returns the stage-1 call.
    pipe = P.SkinLesionPipeline(mm2, None, None, meta_proc)
    res = pipe.predict(_img(), gradcam=False)
    assert res.stage1_label in P.STAGE1_CLASSES
    assert res.subtype_label is None


def test_missing_metadata_still_predicts(models_dir):
    pipe = P.SkinLesionPipeline.from_dir(models_dir)
    # No patient metadata supplied -> processor fills unknown/median, no crash.
    res = pipe.predict(_img(), gradcam=False)
    assert res.stage1_label in P.STAGE1_CLASSES

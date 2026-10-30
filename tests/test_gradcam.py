"""Unit tests for src/gradcam.py — heatmap and overlay shapes/ranges."""
import numpy as np

import gradcam as G
import model as Model


def _dummy_image():
    return np.random.uniform(0, 255, (Model.IMAGE_SIZE, Model.IMAGE_SIZE, 3)).astype(
        "float32"
    )


def test_heatmap_image_only(baseline2):
    img = _dummy_image()
    hm, idx, conf = G.make_gradcam_heatmap(baseline2, img)
    assert hm.ndim == 2  # spatial map
    assert hm.min() >= 0.0 and hm.max() <= 1.0 + 1e-6
    assert idx in (0, 1)
    assert 0.0 <= conf <= 1.0


def test_heatmap_multimodal_needs_meta(mm2, meta_proc):
    img = _dummy_image()
    meta = meta_proc.transform_one(age=60, sex="male", localization="back").astype(
        "float32"
    )
    hm, idx, conf = G.make_gradcam_heatmap(mm2, img, meta=meta)
    assert hm.ndim == 2
    assert 0.0 <= conf <= 1.0


def test_overlay_shape_and_dtype(baseline2):
    img = _dummy_image()
    hm, _, _ = G.make_gradcam_heatmap(baseline2, img)
    overlay = G.overlay_heatmap(img, hm)
    assert overlay.shape == (Model.IMAGE_SIZE, Model.IMAGE_SIZE, 3)
    assert overlay.dtype == np.uint8


def test_specific_pred_index_respected(mm2, meta_proc):
    img = _dummy_image()
    meta = meta_proc.transform_one(age=40, sex="female", localization="face").astype(
        "float32"
    )
    hm, idx, _ = G.make_gradcam_heatmap(mm2, img, meta=meta, pred_index=0)
    assert idx == 0

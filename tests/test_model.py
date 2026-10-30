"""Unit tests for src/model.py — architecture shapes and the multimodal flag."""
import numpy as np

import model as Model


def test_baseline_is_image_only(baseline2):
    assert not Model.is_multimodal(baseline2)
    # single image input, softmax over 2 classes
    assert baseline2.output_shape[-1] == 2
    assert baseline2.input_shape == (None, Model.IMAGE_SIZE, Model.IMAGE_SIZE, 3)


def test_multimodal_has_two_inputs(mm2, meta_proc):
    assert Model.is_multimodal(mm2)
    img_in, meta_in = mm2.input
    assert tuple(img_in.shape) == (None, Model.IMAGE_SIZE, Model.IMAGE_SIZE, 3)
    assert tuple(meta_in.shape) == (None, meta_proc.feature_dim)
    assert mm2.output_shape[-1] == 2


def test_backbone_named_and_present(mm2):
    # Grad-CAM relies on locating the backbone by this name.
    names = [l.name for l in mm2.layers]
    assert Model.BACKBONE_NAME in names


def test_baseline_forward_is_probability(baseline2):
    x = np.random.uniform(0, 255, (2, Model.IMAGE_SIZE, Model.IMAGE_SIZE, 3)).astype(
        "float32"
    )
    p = baseline2.predict(x, verbose=0)
    assert p.shape == (2, 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)


def test_multimodal_forward_is_probability(mm3, meta_proc):
    x = np.random.uniform(0, 255, (2, Model.IMAGE_SIZE, Model.IMAGE_SIZE, 3)).astype(
        "float32"
    )
    m = np.zeros((2, meta_proc.feature_dim), "float32")
    p = mm3.predict([x, m], verbose=0)
    assert p.shape == (2, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)


def test_callbacks_stack(tmp_path):
    cbs = Model.make_callbacks(str(tmp_path / "ckpt.keras"))
    kinds = {type(c).__name__ for c in cbs}
    assert {"EarlyStopping", "LearningRateScheduler", "ModelCheckpoint"} <= kinds

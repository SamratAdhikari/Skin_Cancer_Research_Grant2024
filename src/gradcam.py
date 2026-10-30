"""
gradcam.py
----------
Grad-CAM explainability for the EfficientNetB3-based classifiers.

Grad-CAM highlights the image regions that most influenced a prediction, which
for skin-lesion models is exactly the evidence a reviewer wants: does the
network attend to the lesion, or to artifacts (hair, rulers, ink, vignetting)?

Why this implementation is a little unusual
===========================================
Both models nest the EfficientNet backbone as a single sub-model and pool it
with ``pooling='max'``, so there is no spatial feature map exposed at the top
level to differentiate against. Rather than rely on the fragile "build a Model
from outer inputs to an inner layer output" trick (which is brittle under
Keras 3), we:

1. build a small feature extractor mapping the image input to the backbone's
   last *spatial* activation (``top_activation``, shape ``H'×W'×1536``);
2. inside a ``GradientTape``, re-apply global-max-pooling and the model's own
   head layers (looked up by the names defined in ``model.py``) to that
   activation, so the class score is a genuine differentiable function of the
   spatial map;
3. take the gradient of the class score w.r.t. the spatial map and form the
   usual Grad-CAM weighted combination.

This keeps the gradient chain intact and works for both the image-only and the
multimodal head (the metadata vector is fed through unchanged).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

try:
    from .model import BACKBONE_NAME, is_multimodal
except ImportError:  # pragma: no cover - notebook import path
    from model import BACKBONE_NAME, is_multimodal

# Last spatial activation in Keras' EfficientNetB3 (before global pooling).
DEFAULT_CONV_LAYER = "top_activation"


def _find_last_spatial_layer(backbone: tf.keras.Model) -> str:
    """Fallback: name of the last layer with a 4-D (spatial) output."""
    for layer in reversed(backbone.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if len(shape) == 4:
            return layer.name
    raise ValueError("No spatial (4-D) layer found in backbone for Grad-CAM.")


def _apply_head(
    model: tf.keras.Model, pooled: tf.Tensor, meta: tf.Tensor | None
) -> tf.Tensor:
    """Re-apply the model's head to a pooled image embedding.

    Uses the layer names defined in ``model.py`` so the class score is a
    function of ``pooled`` (and therefore of the spatial activation upstream).
    """
    x = model.get_layer("img_bn")(pooled)
    x = model.get_layer("img_dense")(x)
    x = model.get_layer("img_dropout")(x, training=False)

    if is_multimodal(model):
        if meta is None:
            raise ValueError("metadata batch required for a multimodal model")
        m = model.get_layer("meta_dense1")(meta)
        m = model.get_layer("meta_dropout1")(m, training=False)
        m = model.get_layer("meta_dense2")(m)
        x = model.get_layer("fusion")([x, m])
        x = model.get_layer("fusion_dense")(x)
        x = model.get_layer("fusion_dropout")(x, training=False)

    return model.get_layer("softmax")(x)


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image: np.ndarray,
    meta: np.ndarray | None = None,
    pred_index: int | None = None,
    conv_layer_name: str | None = None,
) -> tuple[np.ndarray, int, float]:
    """Compute a Grad-CAM heatmap for a single image.

    Args:
        model: a model built by ``model.py`` (image-only or multimodal).
        image: single image as ``(H, W, 3)`` float in [0, 255].
        meta:  metadata vector ``(meta_dim,)`` — required iff multimodal.
        pred_index: class to explain; defaults to the argmax prediction.
        conv_layer_name: override the spatial layer (defaults to top_activation).

    Returns:
        ``(heatmap, pred_index, confidence)`` where ``heatmap`` is a normalized
        ``H'×W'`` array in [0, 1].
    """
    backbone = model.get_layer(BACKBONE_NAME)
    conv_name = conv_layer_name or DEFAULT_CONV_LAYER
    if conv_name not in [l.name for l in backbone.layers]:
        conv_name = _find_last_spatial_layer(backbone)

    feature_model = tf.keras.Model(
        backbone.input, backbone.get_layer(conv_name).output, name="gradcam_features"
    )

    img_batch = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    meta_batch = (
        tf.convert_to_tensor(meta[None, ...], dtype=tf.float32) if meta is not None else None
    )

    with tf.GradientTape() as tape:
        conv_out = feature_model(img_batch, training=False)  # (1, H', W', C)
        tape.watch(conv_out)
        pooled = tf.reduce_max(conv_out, axis=[1, 2])  # matches pooling='max'
        preds = _apply_head(model, pooled, meta_batch)  # (1, n_classes)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_score = preds[:, pred_index]

    grads = tape.gradient(class_score, conv_out)  # (1, H', W', C)
    # Channel importance = spatially-averaged gradient (classic Grad-CAM).
    weights = tf.reduce_mean(grads, axis=[1, 2])  # (1, C)
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]  # (H', W')
    cam = tf.nn.relu(cam)

    max_val = tf.reduce_max(cam)
    if max_val > 0:
        cam = cam / max_val
    confidence = float(preds[0, pred_index])
    return cam.numpy(), int(pred_index), confidence


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap onto an image.

    Args:
        image: ``(H, W, 3)`` in [0, 255] (any float/int range is handled).
        heatmap: ``H'×W'`` array in [0, 1] (auto-resized to the image).
        alpha: heatmap opacity.

    Returns:
        ``(H, W, 3)`` uint8 RGB image. Uses matplotlib's colormap so we avoid a
        hard OpenCV dependency in the app.
    """
    import matplotlib

    h, w = image.shape[:2]
    hm = tf.image.resize(heatmap[..., None], [h, w]).numpy()[..., 0]
    # matplotlib.colormaps replaces the removed cm.get_cmap (matplotlib >= 3.9).
    colored = matplotlib.colormaps["jet"](hm)[..., :3]  # (H, W, 3) in [0, 1]

    base = np.clip(image, 0, 255).astype("float32") / 255.0
    blended = (1 - alpha) * base + alpha * colored
    return (np.clip(blended, 0, 1) * 255).astype("uint8")

"""
model.py
--------
Keras model builders for the skin-lesion classifiers.

Two architectures share the *same* EfficientNetB3 image backbone so that any
accuracy gap between them is attributable to the metadata, not to a different
vision model:

* ``build_image_only_baseline`` — the original project's design (image -> CNN
  -> dense head). This is the control condition for the multimodal comparison.
* ``build_multimodal_model``     — adds a small MLP branch over patient
  metadata (age, sex, anatomical site) and fuses it with the image embedding
  via late concatenation before the classification head.

Important preprocessing note
============================
`tf.keras.applications.EfficientNet*` models have their normalization baked in
and therefore **expect pixel inputs in the [0, 255] range** — `preprocess_input`
for EfficientNet is a no-op. So images must be fed as float32 in [0, 255] and
must NOT be rescaled to [0, 1]. (The original notebooks rescaled to [0, 1],
which starved the backbone's internal normalization; feeding [0, 255] is the
correct usage and is what `src/data.py` now does.) Keeping this consistent
between training, evaluation and the app is essential.
"""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.regularizers import l2

IMAGE_SIZE = 224
# Name we pin on the backbone so Grad-CAM can locate it inside either model.
BACKBONE_NAME = "efficientnetb3"


# --------------------------------------------------------------------- backbone
def _image_backbone(image_size: int = IMAGE_SIZE) -> tuple[Input, tf.Tensor]:
    """Shared image branch: EfficientNetB3 -> 256-d embedding.

    Returns the image ``Input`` and the 256-d feature tensor so both the
    baseline and the multimodal model reuse identical vision layers.
    """
    img_in = Input(shape=(image_size, image_size, 3), name="image")
    base = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
        pooling="max",
        name=BACKBONE_NAME,
    )
    # Fine-tune the whole backbone (as in the original project). The images are
    # expected in [0, 255]; EfficientNet normalizes internally.
    base.trainable = True
    x = base(img_in)
    x = BatchNormalization(name="img_bn")(x)
    x = Dense(256, activation="relu", name="img_dense")(x)
    x = Dropout(0.3, name="img_dropout")(x)
    return img_in, x


# ---------------------------------------------------------------- baseline model
def build_image_only_baseline(
    n_classes: int,
    image_size: int = IMAGE_SIZE,
    l2_reg: float = 1e-3,
) -> Model:
    """Image-only control model (matches the original architecture)."""
    img_in, x = _image_backbone(image_size)
    out = Dense(
        n_classes,
        activation="softmax",
        kernel_regularizer=l2(l2_reg),
        name="softmax",
    )(x)
    return Model(inputs=img_in, outputs=out, name="image_only_baseline")


# --------------------------------------------------------------- multimodal model
def build_multimodal_model(
    n_classes: int,
    meta_dim: int,
    image_size: int = IMAGE_SIZE,
    l2_reg: float = 1e-3,
    meta_dropout: float = 0.3,
) -> Model:
    """Image + metadata late-fusion model.

    Args:
        n_classes: number of output classes for this stage.
        meta_dim:  length of the metadata vector (``MetadataProcessor.feature_dim``).
        image_size: square input resolution.
        l2_reg:    L2 penalty on the softmax layer (as in the baseline).
        meta_dropout: dropout inside the metadata MLP.

    The model takes two named inputs, ``"image"`` (H,W,3 in [0,255]) and
    ``"metadata"`` (meta_dim,), so callers can feed a dict/tuple in that order.
    """
    img_in, img_feat = _image_backbone(image_size)

    # Metadata branch: a small MLP is plenty for ~15-30 tabular features and
    # keeps the fused head from being dominated by the 256-d image embedding.
    meta_in = Input(shape=(meta_dim,), name="metadata")
    m = Dense(64, activation="relu", name="meta_dense1")(meta_in)
    m = Dropout(meta_dropout, name="meta_dropout1")(m)
    m = Dense(32, activation="relu", name="meta_dense2")(m)

    fused = Concatenate(name="fusion")([img_feat, m])
    fused = Dense(128, activation="relu", name="fusion_dense")(fused)
    fused = Dropout(0.3, name="fusion_dropout")(fused)
    out = Dense(
        n_classes,
        activation="softmax",
        kernel_regularizer=l2(l2_reg),
        name="softmax",
    )(fused)

    return Model(inputs=[img_in, meta_in], outputs=out, name="multimodal_fusion")


# ------------------------------------------------------------------ compile/train
def compile_model(model: Model, learning_rate: float = 1e-4) -> Model:
    """Compile with the original project's optimizer/loss choice."""
    model.compile(
        optimizer=Adamax(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_callbacks(checkpoint_path: str, patience: int = 5) -> list:
    """Recreate the original callback stack (early stop + LR decay + checkpoint).

    The scheduler holds the LR flat for the first 5 epochs then decays it
    exponentially, matching the behaviour in the original notebooks.
    """
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    def scheduler(epoch: int, lr: float) -> float:
        decay = -0.1
        if epoch < 5:
            return float(lr)
        return float(lr * math.exp((epoch + 1) * decay))

    lr_scheduler = LearningRateScheduler(scheduler)
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="auto",
        verbose=0,
    )
    return [checkpoint, lr_scheduler, early_stopping]


def is_multimodal(model: Model) -> bool:
    """True if the model expects the (image, metadata) pair of inputs."""
    return isinstance(model.input, (list, tuple)) and len(model.input) == 2

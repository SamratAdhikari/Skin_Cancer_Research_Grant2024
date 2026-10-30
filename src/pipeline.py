"""
pipeline.py
-----------
End-to-end hierarchical, multimodal inference for the Streamlit app and the
evaluation notebook.

The clinical decision flow mirrors the original three-model design, but every
stage is now multimodal (image + patient metadata) and the pipeline also
returns a Grad-CAM explanation:

    Stage 1  : benign vs malignant
      |__ if benign     -> Stage 2a : {nv, bkl, vasc}
      |__ if malignant  -> Stage 2b : {mel, bcc, akiec}

The pipeline loads three ``.keras`` models plus the shared
``preprocessing.json`` (the fitted ``MetadataProcessor``). It degrades
gracefully: if the metadata processor or a sub-stage model is missing it still
returns the stage-1 decision, and if the loaded models happen to be image-only
it simply skips the metadata input.

Human-readable class names are attached so the UI can show
"Melanocytic nevus" rather than "nv".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

try:
    from .metadata import MetadataProcessor
    from .model import is_multimodal
    from .gradcam import make_gradcam_heatmap, overlay_heatmap
except ImportError:  # pragma: no cover - notebook/app import path
    from metadata import MetadataProcessor
    from model import is_multimodal
    from gradcam import make_gradcam_heatmap, overlay_heatmap

IMAGE_SIZE = 224

# Stage class orders MUST match src/data.py STAGE_CLASSES.
STAGE1_CLASSES = ["benign", "malignant"]
BENIGN_CLASSES = ["nv", "bkl", "vasc"]
MALIGNANT_CLASSES = ["mel", "bcc", "akiec"]

# Short code -> clinical name for display.
FULL_NAMES = {
    "benign": "Benign",
    "malignant": "Malignant",
    "nv": "Melanocytic nevus",
    "bkl": "Benign keratosis",
    "vasc": "Vascular lesion",
    "mel": "Melanoma",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratosis / Bowen's",
}


@dataclass
class PredictionResult:
    """Structured output of a single end-to-end prediction."""

    stage1_label: str
    stage1_confidence: float
    stage1_probs: dict[str, float]
    subtype_label: str | None = None
    subtype_confidence: float | None = None
    subtype_probs: dict[str, float] = field(default_factory=dict)
    gradcam_overlay: np.ndarray | None = None  # (H, W, 3) uint8, or None
    used_metadata: bool = False

    @property
    def is_malignant(self) -> bool:
        return self.stage1_label == "malignant"

    @property
    def stage1_display(self) -> str:
        return FULL_NAMES.get(self.stage1_label, self.stage1_label)

    @property
    def subtype_display(self) -> str | None:
        if self.subtype_label is None:
            return None
        return FULL_NAMES.get(self.subtype_label, self.subtype_label)


class SkinLesionPipeline:
    """Load the three trained models and run hierarchical inference.

    Args:
        models_dir: directory containing the ``.keras`` files and
            ``preprocessing.json``.
        stage1_name / benign_name / malignant_name: model filenames.

    Use ``SkinLesionPipeline.from_dir(...)`` to construct; models are loaded
    lazily-eagerly at construction so failures surface early in the app.
    """

    def __init__(
        self,
        stage1_model: Any,
        benign_model: Any | None,
        malignant_model: Any | None,
        meta_proc: MetadataProcessor | None,
    ) -> None:
        self.stage1_model = stage1_model
        self.benign_model = benign_model
        self.malignant_model = malignant_model
        self.meta_proc = meta_proc

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dir(
        cls,
        models_dir: str,
        stage1_name: str = "stage1_multimodal.keras",
        benign_name: str = "benign_multimodal.keras",
        malignant_name: str = "malignant_multimodal.keras",
        preprocessing_name: str = "preprocessing.json",
    ) -> "SkinLesionPipeline":
        # Imported here so importing the module (e.g. for the dataclass) does
        # not require TensorFlow to be installed.
        from tensorflow.keras.models import load_model

        def _maybe(path: str):
            return load_model(path) if os.path.exists(path) else None

        stage1 = load_model(os.path.join(models_dir, stage1_name))
        benign = _maybe(os.path.join(models_dir, benign_name))
        malignant = _maybe(os.path.join(models_dir, malignant_name))

        proc_path = os.path.join(models_dir, preprocessing_name)
        meta_proc = MetadataProcessor.load(proc_path) if os.path.exists(proc_path) else None

        return cls(stage1, benign, malignant, meta_proc)

    # ---------------------------------------------------------------- inference
    def predict(
        self,
        image: Image.Image | np.ndarray,
        age: float | None = None,
        sex: str | None = None,
        localization: str | None = None,
        gradcam: bool = True,
    ) -> PredictionResult:
        """Run the full hierarchical prediction for one lesion.

        Args:
            image: PIL image or ``(H, W, 3)`` array (any size; resized here).
            age/sex/localization: patient metadata for the fusion models.
            gradcam: if True, attach a Grad-CAM overlay explaining stage 1.
        """
        img_arr = self._prepare_image(image)  # (224,224,3) float32 [0,255]
        meta_vec = self._prepare_meta(age, sex, localization)

        # ---- Stage 1: benign vs malignant
        s1_probs = self._infer(self.stage1_model, img_arr, meta_vec)
        s1_idx = int(np.argmax(s1_probs))
        result = PredictionResult(
            stage1_label=STAGE1_CLASSES[s1_idx],
            stage1_confidence=float(s1_probs[s1_idx]),
            stage1_probs={c: float(p) for c, p in zip(STAGE1_CLASSES, s1_probs)},
            used_metadata=meta_vec is not None,
        )

        # ---- Stage 2: route to the sub-classifier
        if result.is_malignant:
            sub_model, sub_classes = self.malignant_model, MALIGNANT_CLASSES
        else:
            sub_model, sub_classes = self.benign_model, BENIGN_CLASSES

        if sub_model is not None:
            sub_probs = self._infer(sub_model, img_arr, meta_vec)
            sub_idx = int(np.argmax(sub_probs))
            result.subtype_label = sub_classes[sub_idx]
            result.subtype_confidence = float(sub_probs[sub_idx])
            result.subtype_probs = {c: float(p) for c, p in zip(sub_classes, sub_probs)}

        # ---- Explainability: Grad-CAM over the stage-1 decision
        if gradcam:
            try:
                meta_for_cam = meta_vec if is_multimodal(self.stage1_model) else None
                heatmap, _, _ = make_gradcam_heatmap(
                    self.stage1_model, img_arr, meta=meta_for_cam, pred_index=s1_idx
                )
                result.gradcam_overlay = overlay_heatmap(img_arr, heatmap)
            except Exception as exc:  # explainability must never break inference
                print(f"[pipeline] Grad-CAM skipped: {exc}")

        return result

    # ----------------------------------------------------------------- helpers
    def _prepare_image(self, image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"))
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        return np.asarray(image, dtype="float32")  # [0,255], EfficientNet-ready

    def _prepare_meta(
        self, age: float | None, sex: str | None, localization: str | None
    ) -> np.ndarray | None:
        if self.meta_proc is None:
            return None
        return self.meta_proc.transform_one(age=age, sex=sex, localization=localization).astype(
            "float32"
        )

    def _infer(self, model: Any, img_arr: np.ndarray, meta_vec: np.ndarray | None) -> np.ndarray:
        """Run a single model, feeding metadata only if it expects two inputs."""
        img_batch = img_arr[None, ...]
        if is_multimodal(model):
            if meta_vec is None:
                # Multimodal model but no processor: feed zeros (documented edge).
                meta_vec = np.zeros(model.input[1].shape[-1], dtype="float32")
            return model.predict([img_batch, meta_vec[None, ...]], verbose=0)[0]
        return model.predict(img_batch, verbose=0)[0]

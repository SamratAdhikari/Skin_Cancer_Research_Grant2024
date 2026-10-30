"""
local_infer.py
--------------
Bridge between the trained hierarchical pipeline (``src/pipeline.py``) and the
Streamlit UI.

The original app POSTed the image to an external Hugging Face endpoint. This
module runs the models **locally** instead: it loads the three ``.keras``
models + ``preprocessing.json`` once (cached across reruns) and turns a
``PredictionResult`` into a plain, UI-friendly dict (percentages, display
names). If local models are not present it reports that cleanly so the app can
fall back to the remote API.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

# Make the importable modules in ../src visible when the app runs from the
# project root (src/ is intentionally not a pip package).
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

MODELS_DIR = os.environ.get("MODELS_DIR", "models")

# HAM10000 anatomical sites (+ 'unknown' catch-all) for the metadata form.
LOCALIZATIONS = [
    "unknown",
    "back",
    "lower extremity",
    "trunk",
    "upper extremity",
    "abdomen",
    "face",
    "chest",
    "foot",
    "neck",
    "scalp",
    "hand",
    "ear",
    "genital",
    "acral",
]
SEXES = ["unknown", "male", "female"]


@dataclass
class PipelineStatus:
    """Result of attempting to load the local pipeline."""

    pipeline: Any | None
    error: str | None

    @property
    def available(self) -> bool:
        return self.pipeline is not None


def load_pipeline(models_dir: str = MODELS_DIR) -> PipelineStatus:
    """Load the hierarchical pipeline from ``models_dir`` (best effort).

    Returns a :class:`PipelineStatus`; ``error`` is a human-readable string when
    loading fails (missing models, TensorFlow not installed, etc.). This never
    raises, so the caller can decide whether to fall back to the remote API.
    """
    stage1_path = os.path.join(models_dir, "stage1_multimodal.keras")
    if not os.path.exists(stage1_path):
        return PipelineStatus(
            None,
            f"No trained models found in '{models_dir}/'. "
            "Train them with the Colab notebooks (see README), or configure the "
            "remote API in a .env file.",
        )
    try:
        from pipeline import SkinLesionPipeline  # noqa: WPS433 (local import by design)

        pipe = SkinLesionPipeline.from_dir(models_dir)
        return PipelineStatus(pipe, None)
    except Exception as exc:  # pragma: no cover - depends on runtime env
        return PipelineStatus(None, f"Failed to load local models: {exc}")


def result_to_dict(result: Any) -> dict:
    """Convert a ``PredictionResult`` into a UI-friendly dict (percentages)."""

    def pct(x: float | None) -> float | None:
        return None if x is None else round(float(x) * 100.0, 1)

    out: dict = {
        "source": "local",
        "stage1_label": result.stage1_label,
        "stage1_display": result.stage1_display,
        "stage1_confidence": pct(result.stage1_confidence),
        "is_malignant": result.is_malignant,
        "stage1_probs": {
            _pretty(k): pct(v) for k, v in result.stage1_probs.items()
        },
        "used_metadata": result.used_metadata,
        "subtype_label": result.subtype_label,
        "subtype_display": result.subtype_display,
        "subtype_confidence": pct(result.subtype_confidence),
        "subtype_probs": {
            _pretty(k): pct(v) for k, v in (result.subtype_probs or {}).items()
        },
    }
    return out


def _pretty(code: str) -> str:
    """Map a class code to its clinical display name (falls back to the code)."""
    try:
        from pipeline import FULL_NAMES

        return FULL_NAMES.get(code, code)
    except Exception:
        return code


def run_analysis(
    pipe: Any,
    image,
    age: float | None,
    sex: str | None,
    localization: str | None,
    gradcam: bool = True,
):
    """Run one end-to-end prediction and return ``(result_dict, overlay)``.

    ``overlay`` is a ``(H, W, 3)`` uint8 Grad-CAM image or ``None``.
    """
    # Treat the 'unknown' UI sentinels as missing so the metadata processor
    # routes them to its unknown bucket rather than a literal category.
    sex = None if (sex in (None, "unknown")) else sex
    localization = None if (localization in (None, "unknown")) else localization

    result = pipe.predict(
        image, age=age, sex=sex, localization=localization, gradcam=gradcam
    )
    return result_to_dict(result), result.gradcam_overlay

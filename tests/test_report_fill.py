"""Unit tests for report/fill_report.py — the results.json -> report filler.

These need no TensorFlow, so they run fast and in isolation.
"""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def fr():
    spec = importlib.util.spec_from_file_location(
        "fill_report", os.path.join(ROOT, "report", "fill_report.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_dotted_path(fr):
    data = {"stage1": {"multimodal": {"sensitivity": 0.82}}}
    assert fr._resolve("stage1.multimodal.sensitivity", data) == 0.82
    assert fr._resolve("stage1.missing.key", data) is None


def test_format_variants(fr):
    assert fr._format(0.8241, "pct") == "82.4%"
    assert fr._format(0.9123, "f3") == "0.912"
    assert fr._format(12.0, "int") == "12"
    assert fr._format(None, "pct") is None  # missing signals a TODO upstream


def test_fill_substitutes_and_flags_missing(fr):
    template = "sens={{s.m.sensitivity|pct}} auc={{s.m.auc|f3}} gone={{s.m.nope|f2}}"
    data = {"s": {"m": {"sensitivity": 0.8, "auc": 0.94}}}
    out, n_filled, missing = fr.fill(template, data)
    assert "80.0%" in out
    assert "0.940" in out
    assert "<TODO: s.m.nope>" in out
    assert n_filled == 2
    assert missing == ["s.m.nope"]


def test_report_template_placeholders_all_resolvable(fr):
    """Every placeholder in report.md must map to a key the eval JSON provides.

    Uses a synthetic results.json with the exact schema notebook 06 emits; if a
    placeholder path is misspelled this fails with the offending path.
    """
    def binary():
        return {
            "accuracy": 0.9, "macro_f1": 0.88,
            "per_class": {
                "benign": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 10},
                "malignant": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 5},
            },
            "roc_auc": 0.94, "sensitivity": 0.82, "specificity": 0.93,
            "tp": 4, "tn": 9, "fp": 1, "fn": 1,
        }

    def multi(names):
        return {
            "accuracy": 0.85, "macro_f1": 0.8,
            "per_class": {n: {"precision": 0.8, "recall": 0.78, "f1": 0.79,
                              "support": 8} for n in names},
            "roc_auc_ovr_macro": 0.93,
        }

    mcnemar = {"b_only_B_correct": 9, "c_only_A_correct": 2, "chi2": 3.27,
               "p_value_chi2": 0.07, "p_value_exact": 0.03}
    data = {
        "stage1": {"baseline": binary(), "multimodal": {**binary(), "ece": 0.04},
                   "mcnemar": mcnemar},
        "benign": {"baseline": multi(["nv", "bkl", "vasc"]),
                   "multimodal": {**multi(["nv", "bkl", "vasc"]), "ece": 0.05},
                   "mcnemar": mcnemar},
        "malignant": {"baseline": multi(["mel", "bcc", "akiec"]),
                      "multimodal": {**multi(["mel", "bcc", "akiec"]), "ece": 0.06},
                      "mcnemar": mcnemar},
        "diffusion_ablation": {"no_synth": {"akiec_recall": 0.55, "macro_f1": 0.78},
                               "with_synth": {"akiec_recall": 0.67, "macro_f1": 0.81}},
        "end_to_end": {"stage1_accuracy": 0.9, "malignant_sensitivity": 0.8,
                       "end_to_end_subtype_accuracy": 0.72, "n_test": 1503},
    }
    template = open(os.path.join(ROOT, "report", "report.md"), encoding="utf-8").read()
    out, n_filled, missing = fr.fill(template, data)
    assert missing == [], f"unresolved placeholders in report.md: {missing}"
    assert n_filled > 0
    assert "{{" not in out and "<TODO:" not in out

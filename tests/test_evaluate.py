"""Unit tests for src/evaluate.py — metrics, calibration, McNemar, plots."""
import os

import numpy as np
import pytest

import evaluate as ev


def test_binary_metrics_keys():
    y = np.array([0, 0, 1, 1, 1, 0])
    prob = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8],
                     [0.3, 0.7], [0.1, 0.9], [0.8, 0.2]])
    m = ev.summarize_predictions(y, prob, ["benign", "malignant"])
    for key in ("accuracy", "macro_f1", "roc_auc", "sensitivity", "specificity"):
        assert key in m
    assert 0.0 <= m["sensitivity"] <= 1.0
    assert m["tp"] + m["fn"] == int((y == 1).sum())


def test_multiclass_metrics_keys():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 3, size=30)
    logits = rng.rand(30, 3)
    prob = logits / logits.sum(1, keepdims=True)
    m = ev.summarize_predictions(y, prob, ["mel", "bcc", "akiec"])
    assert "roc_auc_ovr_macro" in m
    assert set(m["per_class"].keys()) == {"mel", "bcc", "akiec"}


def test_ece_in_unit_interval():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, size=50)
    logits = rng.rand(50, 2)
    prob = logits / logits.sum(1, keepdims=True)
    ece = ev.expected_calibration_error(y, prob)
    assert 0.0 <= ece <= 1.0


def test_mcnemar_identical_predictions():
    y = np.array([0, 1, 0, 1, 1])
    same = np.array([0, 1, 0, 0, 1])
    r = ev.mcnemar_test(y, same, same)
    # No discordant pairs -> p-value 1.0 and an explanatory note.
    assert r["p_value"] == 1.0
    assert "note" in r


def test_mcnemar_discordant_reports_pvalue():
    y = np.array([1, 1, 1, 1, 1, 1])
    a = np.array([0, 0, 0, 0, 1, 1])  # mostly wrong
    b = np.array([1, 1, 1, 1, 1, 0])  # mostly right
    r = ev.mcnemar_test(y, a, b)
    assert r["b_only_B_correct"] >= 0 and r["c_only_A_correct"] >= 0
    assert 0.0 <= r["p_value_exact"] <= 1.0


def test_plots_written(tmp_path):
    rng = np.random.RandomState(2)
    y = rng.randint(0, 2, size=40)
    logits = rng.rand(40, 2)
    prob = logits / logits.sum(1, keepdims=True)
    cm = tmp_path / "cm.png"
    roc = tmp_path / "roc.png"
    rel = tmp_path / "rel.png"
    ev.plot_confusion_matrix(y, prob.argmax(1), ["benign", "malignant"], str(cm))
    ev.plot_roc_curves(y, prob, ["benign", "malignant"], str(roc))
    ev.plot_reliability(y, prob, str(rel))
    for p in (cm, roc, rel):
        assert p.exists() and p.stat().st_size > 0


def test_save_metrics_json_roundtrip(tmp_path):
    import json

    path = tmp_path / "nested" / "results.json"
    ev.save_metrics_json({"a": {"b": 1.0}}, str(path))
    assert json.load(open(path))["a"]["b"] == 1.0

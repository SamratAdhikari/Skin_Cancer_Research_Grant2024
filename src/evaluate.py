"""
evaluate.py
-----------
Medical-grade evaluation utilities for the skin-lesion classifiers.

The original project reported only accuracy. For a clinical decision-support
model that is not enough: missing a melanoma (a false negative) is far more
costly than a false alarm, so we report the metrics reviewers expect:

* per-class precision / recall / F1 and macro averages
* **sensitivity & specificity** (esp. malignant-vs-benign at stage 1)
* **ROC-AUC** — binary and one-vs-rest for the multiclass stages
* **calibration**: expected calibration error (ECE) + reliability curve
* confusion matrices (counts and row-normalized)
* **McNemar's test** to check whether the multimodal model differs
  *significantly* from the image-only baseline on the same test set

Everything here is framework-light (numpy + scikit-learn + matplotlib) and
returns plain dicts / saves PNGs, so it is easy to drop the numbers into the
report tables.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: safe to import in notebooks and scripts alike
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


# ------------------------------------------------------------------ core metrics
def summarize_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute a full metrics dict from labels and predicted probabilities.

    Args:
        y_true: ``(N,)`` int class indices.
        y_prob: ``(N, n_classes)`` predicted probabilities.
        class_names: ordered class names.

    Returns:
        Nested dict of accuracy, macro-F1, per-class P/R/F1, ROC-AUC, and (for
        binary problems) sensitivity/specificity.
    """
    y_pred = y_prob.argmax(axis=1)
    n_classes = len(class_names)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    out: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
    }

    # ROC-AUC: binary uses the positive-class probability; multiclass uses OvR.
    try:
        if n_classes == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            out["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    except ValueError:
        # Happens if a class is absent from y_true in a small split.
        out["roc_auc"] = None

    if n_classes == 2:
        out.update(_binary_sens_spec(y_true, y_pred))

    return out


def _binary_sens_spec(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Sensitivity/specificity for a binary problem (positive class = index 1).

    In stage 1 index 1 is 'malignant', so sensitivity is the malignant recall —
    the single most important number for a screening tool.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "sensitivity": float(sensitivity),  # malignant recall
        "specificity": float(specificity),  # benign recall
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# --------------------------------------------------------------------- calibration
def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error using the top-class confidence.

    ECE measures whether a model's confidence matches its accuracy — important
    if a clinician is to trust the reported percentage.
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


# ------------------------------------------------------------- significance test
def mcnemar_test(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
) -> dict:
    """McNemar's test comparing two models on the *same* test set.

    Tests whether model A and model B (e.g. image-only vs multimodal) disagree
    with the ground truth in a statistically different way. Uses the exact
    binomial test on discordant pairs (robust for small counts) and reports the
    continuity-corrected chi-square statistic too.

    Returns dict with the discordant counts, chi-square, and p-value.
    """
    from scipy.stats import binomtest, chi2

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    # b: A wrong, B right ; c: A right, B wrong
    b = int(np.sum(~correct_a & correct_b))
    c = int(np.sum(correct_a & ~correct_b))

    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0,
                "note": "models make identical errors"}

    chi2_stat = (abs(b - c) - 1) ** 2 / n  # continuity-corrected
    p_chi2 = float(1 - chi2.cdf(chi2_stat, df=1))
    p_exact = float(binomtest(min(b, c), n=n, p=0.5).pvalue)
    return {
        "b_only_B_correct": b,
        "c_only_A_correct": c,
        "chi2": float(chi2_stat),
        "p_value_chi2": p_chi2,
        "p_value_exact": p_exact,
    }


# ------------------------------------------------------------------------- plots
def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], path: str,
    normalize: bool = True, title: str = "Confusion Matrix",
) -> None:
    """Save a (optionally row-normalized) confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    display = cm.astype(float)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(1.6 * len(class_names) + 2, 1.6 * len(class_names) + 1.5))
    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=display.max() or 1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt = f"{display[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if display[i, j] > display.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, path)


def plot_roc_curves(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str], path: str,
) -> None:
    """Save ROC curve(s): a single curve for binary, OvR curves for multiclass."""
    fig, ax = plt.subplots(figsize=(6, 5))
    n_classes = len(class_names)
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        ax.plot(fpr, tpr, label=f"{class_names[1]} (AUC={auc:.3f})")
    else:
        for i, name in enumerate(class_names):
            y_bin = (y_true == i).astype(int)
            if y_bin.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            auc = roc_auc_score(y_bin, y_prob[:, i])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, path)


def plot_reliability(
    y_true: np.ndarray, y_prob: np.ndarray, path: str, n_bins: int = 10,
) -> None:
    """Save a reliability diagram (confidence vs accuracy) with the ECE value."""
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            xs.append(confidences[mask].mean())
            ys.append(accuracies[mask].mean())
    ece = expected_calibration_error(y_true, y_prob, n_bins)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfectly calibrated")
    ax.plot(xs, ys, "o-", label=f"model (ECE={ece:.3f})")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram"); ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, path)


def save_metrics_json(metrics: dict, path: str) -> None:
    """Dump a metrics dict to JSON (creating the parent dir if needed)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

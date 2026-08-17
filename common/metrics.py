"""Locked benchmark metric set (Phase B of PROJECT_PLAN.md).

Aligned with the OmniFall paper (arXiv:2505.19889) + fall-detection literature:
Balanced Accuracy, Accuracy, Sensitivity, Specificity, Precision, F1, Macro-F1, AUC-ROC,
plus the confusion matrix. Binary convention: NoFALL=0 (negative), FALL=1 (positive).

Use the SAME function for smoke-test, full-data, from-scratch, fine-tune, and the
proposed model so every number is directly comparable.
"""
from __future__ import annotations

import numpy as np

CLASS_NAMES = ["NoFALL", "FALL"]   # index 0 = negative, index 1 = positive (FALL)
POS_INDEX = 1


def compute_metrics(y_true, y_score):
    """Compute the full locked metric set.

    Args:
        y_true:  array [N] of int labels in {0,1}.
        y_score: array [N] or [N,2] of model scores/probabilities.
                 If [N,2], column 1 (FALL) is taken as the positive score.
    Returns:
        dict of scalar metrics + 'confusion_matrix' (2x2 list) + 'support'.
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )

    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim == 2:
        pos_score = y_score[:, POS_INDEX]
    else:
        pos_score = y_score.ravel()
    y_pred = (pos_score >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0   # recall of FALL
    specificity = tn / (tn + fp) if (tn + fp) else 0.0   # recall of NoFALL

    out = {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity":       float(sensitivity),                # = recall (FALL)
        "specificity":       float(specificity),
        "precision":         float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1":                float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix":  cm.tolist(),
        "support":           {"NoFALL": int((y_true == 0).sum()), "FALL": int((y_true == 1).sum())},
    }
    # AUC needs both classes present and varying scores
    try:
        if len(np.unique(y_true)) == 2:
            out["auc_roc"] = float(roc_auc_score(y_true, pos_score))
        else:
            out["auc_roc"] = float("nan")
    except Exception:
        out["auc_roc"] = float("nan")
    return out


def format_metrics(m: dict) -> str:
    """Pretty one-block string for logging."""
    keys = ["accuracy", "balanced_accuracy", "sensitivity", "specificity",
            "precision", "f1", "macro_f1", "auc_roc"]
    lines = [f"  {k:<18}: {m[k]:.4f}" for k in keys if k in m]
    cm = m.get("confusion_matrix")
    if cm:
        lines.append(f"  confusion (rows=true NoFALL/FALL, cols=pred): {cm}")
    return "\n".join(lines)

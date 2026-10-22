"""Metric definitions. AUROC is undefined (NaN) when only one class is present."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties get half credit)."""
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    return {
        "tp": int((y & pred).sum()),
        "fn": int((y & ~pred).sum()),
        "fp": int((~y & pred).sum()),
        "tn": int((~y & ~pred).sum()),
    }


def sensitivity(y: np.ndarray, pred: np.ndarray) -> float:
    c = confusion(y, pred)
    d = c["tp"] + c["fn"]
    return c["tp"] / d if d else float("nan")


def specificity(y: np.ndarray, pred: np.ndarray) -> float:
    c = confusion(y, pred)
    d = c["tn"] + c["fp"]
    return c["tn"] / d if d else float("nan")


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    s, p = sensitivity(y, pred), specificity(y, pred)
    if np.isnan(s) or np.isnan(p):
        return float("nan")
    return (s + p) / 2


def all_metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray | None) -> dict[str, float | int]:
    out: dict[str, float | int] = {"auroc": auroc(y, score), "n_test": int(len(y)), "n_test_pos": int(np.sum(y))}
    if pred is None:  # score-only methods without a defined decision rule
        out.update({"balanced_accuracy": float("nan"), "sensitivity": float("nan"), "specificity": float("nan")})
        out.update({"tp": -1, "fn": -1, "fp": -1, "tn": -1})
        return out
    out.update(
        {
            "balanced_accuracy": balanced_accuracy(y, pred),
            "sensitivity": sensitivity(y, pred),
            "specificity": specificity(y, pred),
            **confusion(y, pred),
        }
    )
    return out

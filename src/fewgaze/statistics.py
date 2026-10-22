"""Uncertainty estimation that respects the subject-level structure of the data.

Point estimate: mean of the per-(fold, seed) metric, i.e. every model is evaluated only on the
held-out subjects of its own fold, then averaged.

Confidence interval: fold-stratified subject cluster bootstrap. In each replicate, the test
subjects of every fold are resampled with replacement (all images of a drawn subject come
along), the metric is recomputed for every seed on that resample and averaged exactly like the
point estimate. The SAME subject resamples are used for every method, so paired differences
are computed on identical bootstrap draws.

What the interval represents: uncertainty due to WHICH subjects were tested, holding the
trained models fixed. It does not capture variation from retraining on different training
subjects, and with 56 subjects it is an approximate percentile interval.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fewgaze.metrics import auroc, balanced_accuracy

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def metric_fn(name: str, threshold: float = 0.5) -> MetricFn:
    if name == "auroc":
        return auroc
    if name == "balanced_accuracy":
        return lambda y, s: balanced_accuracy(y, s >= threshold)
    raise ValueError(name)


@dataclass
class FoldBlock:
    """Predictions for one test fold: rows are aligned across seeds and across methods."""

    subjects: np.ndarray  # (n,) subject id per test sample
    labels: np.ndarray  # (n,)
    scores: dict[str, np.ndarray]  # method -> (n_seeds, n) scores


def _subject_index(subjects: np.ndarray) -> list[np.ndarray]:
    uniq = np.unique(subjects)
    return [np.flatnonzero(subjects == s) for s in uniq]


def point_estimate(blocks: list[FoldBlock], method: str, fn: MetricFn) -> float:
    vals = [fn(b.labels, row) for b in blocks for row in b.scores[method]]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap(
    blocks: list[FoldBlock],
    methods: list[str],
    fn: MetricFn,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Return method -> (n_boot,) bootstrap distribution of the fold/seed-averaged metric."""
    rng = np.random.default_rng(seed)
    per_subject = [_subject_index(b.subjects) for b in blocks]
    out = {m: np.empty(n_boot) for m in methods}
    for r in range(n_boot):
        idxs = []
        for groups in per_subject:
            draw = rng.integers(0, len(groups), size=len(groups))
            idxs.append(np.concatenate([groups[d] for d in draw]))
        for m in methods:
            vals = []
            for b, idx in zip(blocks, idxs):
                y = b.labels[idx]
                for row in b.scores[m]:
                    vals.append(fn(y, row[idx]))
            arr = np.asarray(vals)
            out[m][r] = np.nanmean(arr) if np.isfinite(arr).any() else np.nan
    return out


def percentile_ci(dist: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    d = dist[np.isfinite(dist)]
    if d.size == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))

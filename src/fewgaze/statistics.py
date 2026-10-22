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


class Metric:
    """A metric that can also be evaluated in closed form on subject-count bootstrap resamples."""

    def __init__(self, name: str, threshold: float = 0.5):
        if name not in ("auroc", "balanced_accuracy"):
            raise ValueError(name)
        self.name, self.threshold = name, threshold

    def __call__(self, y: np.ndarray, s: np.ndarray) -> float:
        if self.name == "auroc":
            return auroc(y, s)
        return balanced_accuracy(y, s >= self.threshold)


def metric_fn(name: str, threshold: float = 0.5) -> Metric:
    return Metric(name, threshold)


@dataclass
class FoldBlock:
    """Predictions for one test fold: rows are aligned across seeds and across methods."""

    subjects: np.ndarray  # (n,) subject id per test sample
    labels: np.ndarray  # (n,)
    scores: dict[str, np.ndarray]  # method -> (n_seeds, n) scores


def point_estimate(blocks: list[FoldBlock], method: str, fn: MetricFn) -> float:
    vals = [fn(b.labels, row) for b in blocks for row in b.scores[method]]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _onehot(subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, inv = np.unique(subjects, return_inverse=True)
    oh = np.zeros((len(subjects), len(uniq)))
    oh[np.arange(len(subjects)), inv] = 1.0
    return uniq, oh


def metric_on_counts(block: FoldBlock, row: np.ndarray, counts: np.ndarray, metric: Metric) -> np.ndarray:
    """Exact metric value on every resample described by ``counts`` (n_boot, n_subjects).

    Resampling subject i c_i times multiplies each of its images c_i times, so for AUROC
    the number of correctly ordered (positive, negative) pairs is c^T A c with
    A[i, j] = sum over positives p of subject i and negatives n of subject j of H(s_p - s_n)
    (H = 1, 1/2 for ties, 0), and the number of pairs is (c . P)(c . N).
    """
    _, oh = _onehot(block.subjects)
    y = block.labels.astype(bool)
    P, N = oh[y].sum(0), oh[~y].sum(0)
    cp, cn = counts @ P, counts @ N
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric.name == "auroc":
            sp, sn = row[y], row[~y]
            M = (sp[:, None] > sn[None, :]) + 0.5 * (sp[:, None] == sn[None, :])
            A = oh[y].T @ M @ oh[~y]
            return np.einsum("bi,ij,bj->b", counts, A, counts) / (cp * cn)
        pred = row >= metric.threshold
        tp, tn = oh[y & pred].sum(0), oh[~y & ~pred].sum(0)
        return ((counts @ tp) / cp + (counts @ tn) / cn) / 2


def bootstrap(
    blocks: list[FoldBlock],
    methods: list[str],
    fn: Metric,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Return method -> (n_boot,) bootstrap distribution of the fold/seed-averaged metric.

    Subjects are resampled with replacement within each fold; the same resamples are used for
    every method and seed, so differences between methods are paired.
    """
    rng = np.random.default_rng(seed)
    counts = []
    for b in blocks:
        n_subj = len(np.unique(b.subjects))
        draws = rng.integers(0, n_subj, size=(n_boot, n_subj))
        c = np.zeros((n_boot, n_subj))
        np.add.at(c, (np.arange(n_boot)[:, None], draws), 1.0)
        counts.append(c)
    out = {}
    for m in methods:
        vals = np.stack([metric_on_counts(b, row, c, fn) for b, c in zip(blocks, counts) for row in b.scores[m]])
        with np.errstate(invalid="ignore"):
            out[m] = np.where(np.isfinite(vals).any(0), np.nanmean(np.where(np.isfinite(vals), vals, np.nan), axis=0), np.nan)
    return out


def percentile_ci(dist: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    d = dist[np.isfinite(dist)]
    if d.size == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))

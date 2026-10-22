import math

import numpy as np
from sklearn.metrics import roc_auc_score

from fewgaze.metrics import all_metrics, auroc, balanced_accuracy, confusion, sensitivity, specificity
from fewgaze.statistics import FoldBlock, bootstrap, metric_fn, percentile_ci, point_estimate


def test_auroc_known_values():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5  # ties get half credit
    assert math.isclose(auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)


def test_auroc_matches_sklearn_on_random_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    s[:30] = np.round(s[:30])  # include ties
    assert math.isclose(auroc(y, s), roc_auc_score(y, s), rel_tol=1e-12)


def test_auroc_single_class_is_nan():
    assert math.isnan(auroc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert math.isnan(auroc([0, 0], [0.1, 0.2]))


def test_threshold_metrics():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    assert confusion(y, p) == {"tp": 2, "fn": 1, "fp": 1, "tn": 4}
    assert math.isclose(sensitivity(y, p), 2 / 3)
    assert math.isclose(specificity(y, p), 4 / 5)
    assert math.isclose(balanced_accuracy(y, p), (2 / 3 + 4 / 5) / 2)
    assert math.isnan(balanced_accuracy(np.ones(3), np.ones(3)))


def test_all_metrics_without_decision_rule():
    m = all_metrics(np.array([0, 1]), np.array([0.2, 0.7]), None)
    assert m["auroc"] == 1.0 and math.isnan(m["balanced_accuracy"]) and m["tp"] == -1


def _blocks(seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for f in range(3):
        subj = np.repeat(np.arange(8) + 10 * f, 6)
        y = np.tile([1, 0, 0, 0, 0, 0], 8)
        good = y + rng.normal(0, 0.5, (2, len(y)))
        bad = rng.normal(0, 1, (2, len(y)))
        blocks.append(FoldBlock(subj, y, {"good": good, "bad": bad}))
    return blocks


def test_bootstrap_ci_contains_point_and_is_deterministic():
    b = _blocks()
    fn = metric_fn("auroc")
    d1 = bootstrap(b, ["good", "bad"], fn, 200, seed=1)
    d2 = bootstrap(b, ["good", "bad"], fn, 200, seed=1)
    assert np.array_equal(d1["good"], d2["good"])
    lo, hi = percentile_ci(d1["good"])
    assert lo <= point_estimate(b, "good", fn) <= hi
    diff = d1["good"] - d1["bad"]
    assert percentile_ci(diff)[0] > 0  # clearly better method -> CI of difference excludes 0


def test_bootstrap_resamples_whole_subjects():
    # if each subject's scores are constant within subject, every resample is a mix of whole subjects
    subj = np.repeat(np.arange(5), 4)
    y = np.tile([1, 0, 0, 0], 5)
    s = np.where(y == 1, 1.0, 0.0)[None, :]
    b = [FoldBlock(subj, y, {"m": s})]
    d = bootstrap(b, ["m"], metric_fn("auroc"), 50, seed=0)
    assert np.allclose(d["m"], 1.0)


def test_closed_form_bootstrap_equals_explicit_resampling():
    from fewgaze.statistics import metric_on_counts

    rng = np.random.default_rng(3)
    subj = np.repeat(np.array(["a", "b", "c", "d", "e"]), 7)
    y = np.tile([1, 1, 0, 0, 0, 0, 0], 5)
    row = np.round(rng.normal(size=len(y)) + y, 1)  # rounding creates ties
    block = FoldBlock(subj, y, {"m": row[None, :]})
    uniq = np.unique(subj)
    for _ in range(20):
        draw = rng.integers(0, len(uniq), len(uniq))
        counts = np.bincount(draw, minlength=len(uniq)).astype(float)[None, :]
        idx = np.concatenate([np.flatnonzero(subj == uniq[d]) for d in draw])
        for name in ("auroc", "balanced_accuracy"):
            fn = metric_fn(name, 0.5)
            fast = metric_on_counts(block, row, counts, fn)[0]
            slow = fn(y[idx], row[idx])
            assert math.isclose(fast, slow, rel_tol=1e-12, abs_tol=1e-12) or (math.isnan(fast) and math.isnan(slow))

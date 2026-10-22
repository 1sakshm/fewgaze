"""Subject-independent folds, label-budget sampling, and simulated label noise.

Design decisions (documented in docs/methodology.md):
* Folds are assigned per SUBJECT: subjects are sorted, permuted with a fixed seed and dealt
  round-robin into ``n_folds`` folds. Every Columbia subject contributes both classes, so
  plain grouped assignment already yields both classes in every test fold; this is verified
  by ``check_folds`` rather than assumed.
* Budget sampling is NESTED: for a given (fold, seed, class) one random permutation of the
  eligible training samples is drawn and the first k are taken. The k=5 set is therefore a
  subset of the k=10 set, which reduces sampling noise between budgets.
* Noise is applied only to a copy of the selected TRAINING labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class SplitError(RuntimeError):
    pass


def _rng(*keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in keys]))


def assign_folds(subjects: pd.Series | list[str], n_folds: int, seed: int) -> dict[str, int]:
    uniq = sorted(set(subjects))
    if len(uniq) < n_folds:
        raise SplitError(f"Only {len(uniq)} subjects for {n_folds} folds")
    order = _rng(seed, 0).permutation(len(uniq))
    return {uniq[j]: i % n_folds for i, j in enumerate(order)}


@dataclass
class Fold:
    index: int
    train_subjects: list[str]
    test_subjects: list[str]
    train_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)


def make_folds(df: pd.DataFrame, n_folds: int, seed: int, subject_col: str = "subject") -> list[Fold]:
    """``df`` must hold one row per eligible labelled sample with columns sample_id, subject, label."""
    mapping = assign_folds(df[subject_col], n_folds, seed)
    fold_of = df[subject_col].map(mapping)
    folds = []
    for f in range(n_folds):
        test = df[fold_of == f]
        train = df[fold_of != f]
        folds.append(
            Fold(
                index=f,
                train_subjects=sorted(train[subject_col].unique().tolist()),
                test_subjects=sorted(test[subject_col].unique().tolist()),
                train_ids=train["sample_id"].tolist(),
                test_ids=test["sample_id"].tolist(),
            )
        )
    return folds


def check_folds(df: pd.DataFrame, folds: list[Fold], subject_col: str = "subject") -> list[str]:
    """Return a list of human-readable problems (empty list = all invariants hold)."""
    problems = []
    all_ids = set(df["sample_id"])
    seen_test: set[str] = set()
    label_of = dict(zip(df["sample_id"], df["label"]))
    subj_of = dict(zip(df["sample_id"], df[subject_col]))
    for fold in folds:
        tr, te = set(fold.train_ids), set(fold.test_ids)
        if set(fold.train_subjects) & set(fold.test_subjects):
            problems.append(f"fold {fold.index}: subject overlap")
        if tr & te:
            problems.append(f"fold {fold.index}: sample overlap")
        if tr | te != all_ids:
            problems.append(f"fold {fold.index}: samples missing from partition")
        if {subj_of[i] for i in te} != set(fold.test_subjects):
            problems.append(f"fold {fold.index}: test subject list inconsistent with samples")
        if len({label_of[i] for i in te}) < 2:
            problems.append(f"fold {fold.index}: test fold lacks one class")
        if seen_test & te:
            problems.append(f"fold {fold.index}: sample appears in more than one test fold")
        seen_test |= te
    if seen_test != all_ids:
        problems.append("union of test folds does not cover all samples")
    return problems


def sample_budget(train: pd.DataFrame, budget: int | str, seed: int, fold_index: int) -> list[str] | None:
    """Select ``budget`` samples per class from ``train`` (training subjects only).

    Returns sorted sample IDs, or ``None`` if a class has fewer than ``budget`` samples
    (the run is then recorded as unavailable; we never oversample).
    """
    chosen: list[str] = []
    for cls in (0, 1):
        ids = np.array(sorted(train.loc[train["label"] == cls, "sample_id"]))
        if budget == "all":
            chosen.extend(ids.tolist())
            continue
        k = int(budget)
        if len(ids) < k:
            return None
        perm = _rng(seed, fold_index, cls, 7919).permutation(len(ids))
        chosen.extend(ids[perm[:k]].tolist())
    return sorted(chosen)


def corrupt_labels(y: np.ndarray, rate: float, seed: int, fold_index: int, budget_key: int) -> tuple[np.ndarray, np.ndarray]:
    """Class-balanced symmetric flips: exactly round(rate * n_c) labels of each class c are flipped.

    Returns (noisy_labels, flip_mask). ``y`` itself is never modified.
    """
    if not 0 <= rate < 0.5:
        raise ValueError("noise rate must be in [0, 0.5)")
    y = np.asarray(y)
    noisy = y.copy()
    mask = np.zeros(len(y), dtype=bool)
    rng = _rng(seed, fold_index, budget_key, int(round(rate * 1000)), 104729)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        n_flip = int(round(rate * len(idx)))
        if n_flip:
            pick = rng.choice(idx, size=n_flip, replace=False)
            mask[pick] = True
    noisy[mask] = 1 - noisy[mask]
    return noisy, mask

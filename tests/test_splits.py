import numpy as np
import pytest

from fewgaze.splits import SplitError, assign_folds, check_folds, corrupt_labels, make_folds, sample_budget


def test_folds_are_subject_disjoint_and_cover_everything(labelled):
    folds = make_folds(labelled, 5, seed=1)
    assert check_folds(labelled, folds) == []
    subj = dict(zip(labelled["sample_id"], labelled["subject"]))
    for f in folds:
        assert not ({subj[i] for i in f.train_ids} & {subj[i] for i in f.test_ids})
        assert not (set(f.train_ids) & set(f.test_ids))
        assert len(f.train_ids) + len(f.test_ids) == len(labelled)
    all_test = [i for f in folds for i in f.test_ids]
    assert len(all_test) == len(set(all_test)) == len(labelled)


def test_every_test_fold_has_both_classes(labelled):
    lab = dict(zip(labelled["sample_id"], labelled["label"]))
    for f in make_folds(labelled, 5, seed=3):
        assert {lab[i] for i in f.test_ids} == {0, 1}


def test_folds_deterministic_and_seed_dependent(labelled):
    a = make_folds(labelled, 5, seed=7)
    b = make_folds(labelled, 5, seed=7)
    c = make_folds(labelled, 5, seed=8)
    assert [f.test_subjects for f in a] == [f.test_subjects for f in b]
    assert [f.test_subjects for f in a] != [f.test_subjects for f in c]
    for seed in range(10):  # any seed keeps the subject-independence invariant
        assert check_folds(labelled, make_folds(labelled, 5, seed=seed)) == []


def test_check_folds_detects_leakage(labelled):
    folds = make_folds(labelled, 5, seed=1)
    folds[0].train_ids.append(folds[0].test_ids[0])
    folds[0].train_subjects.append(folds[0].test_subjects[0])
    problems = check_folds(labelled, folds)
    assert any("overlap" in p for p in problems)


def test_too_few_subjects():
    with pytest.raises(SplitError):
        assign_folds(["a", "b"], 5, 0)


def test_budget_sampling_uses_only_training_subjects_and_is_nested(labelled):
    fold = make_folds(labelled, 5, seed=1)[0]
    train = labelled[labelled["sample_id"].isin(fold.train_ids)]
    s5 = sample_budget(train, 5, seed=0, fold_index=0)
    s10 = sample_budget(train, 10, seed=0, fold_index=0)
    assert set(s5) <= set(fold.train_ids)
    assert set(s5) <= set(s10)
    lab = dict(zip(labelled["sample_id"], labelled["label"]))
    assert sum(lab[i] for i in s10) == 10 and len(s10) == 20
    assert s10 == sample_budget(train, 10, seed=0, fold_index=0)
    assert s10 != sample_budget(train, 10, seed=1, fold_index=0)


def test_unavailable_budget_returns_none_and_all_uses_everything(labelled):
    fold = make_folds(labelled, 5, seed=1)[0]
    train = labelled[labelled["sample_id"].isin(fold.train_ids)]
    n_pos = int((train["label"] == 1).sum())
    assert sample_budget(train, n_pos + 1, seed=0, fold_index=0) is None
    assert sorted(sample_budget(train, "all", seed=0, fold_index=0)) == sorted(train["sample_id"])


def test_label_noise_flips_exact_counts_and_preserves_input():
    y = np.array([0] * 20 + [1] * 20)
    y_copy = y.copy()
    noisy, mask = corrupt_labels(y, 0.2, seed=0, fold_index=0, budget_key=20)
    assert np.array_equal(y, y_copy)  # original labels preserved
    assert mask.sum() == 8 and mask[:20].sum() == 4 and mask[20:].sum() == 4
    assert np.array_equal(noisy[mask], 1 - y[mask]) and np.array_equal(noisy[~mask], y[~mask])
    noisy2, mask2 = corrupt_labels(y, 0.2, seed=0, fold_index=0, budget_key=20)
    assert np.array_equal(mask, mask2)
    with pytest.raises(ValueError):
        corrupt_labels(y, 0.5, 0, 0, 0)

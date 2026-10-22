"""End-to-end protocol tests on synthetic features: training-only preprocessing, shared
selections, noise isolation, result schema, unequal eligibility, and determinism."""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import SYNTH_METHODS, synthetic_tables

from fewgaze.evaluate import RESULT_FIELDS, run_protocol, save_protocol
from fewgaze.summary import paired_differences, summarize_curves
from fewgaze.train import build_classifier, fit_predict


def test_scaler_is_fit_on_training_data_only(base_cfg):
    rng = np.random.default_rng(0)
    Xtr, ytr = rng.normal(0, 1, (40, 3)), np.array([0, 1] * 20)
    Xte = rng.normal(100, 1, (10, 3))  # wildly shifted test set must not affect the scaler
    _, model, _, _ = fit_predict(Xtr, ytr, Xte, base_cfg["classifier"], 1, 0)
    scaler = model.steps[0][1]
    assert np.allclose(scaler.mean_, Xtr.mean(axis=0))


def test_polynomial_pipeline_separates_centered_class(base_cfg):
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (400, 2))
    y = (np.linalg.norm(X, axis=1) < 0.4).astype(int)  # positives in a central disc
    lin, _, _, _ = fit_predict(X[:300], y[:300], X[300:], base_cfg["classifier"], 1, 0)
    quad, _, _, _ = fit_predict(X[:300], y[:300], X[300:], base_cfg["classifier"], 2, 0)
    from fewgaze.metrics import auroc

    assert auroc(y[300:], quad) > 0.95 > auroc(y[300:], lin)
    assert len(build_classifier(base_cfg["classifier"], 2).steps) == 4


def test_single_class_training_raises(base_cfg):
    with pytest.raises(ValueError, match="single class"):
        fit_predict(np.zeros((4, 2)), np.zeros(4), np.zeros((2, 2)), base_cfg["classifier"])


def _run(cfg, labelled, drop=0, run_id="test"):
    tables = synthetic_tables(labelled, drop=drop)
    return run_protocol(cfg, labelled, tables, SYNTH_METHODS, run_id)


def test_protocol_shares_selections_and_test_sets(cfg, labelled):
    out = _run(cfg, labelled)
    ok = out.results[out.results["status"] == "ok"]
    # same selection key -> same n_train for both methods
    g = ok[ok["condition"] == "clean"].groupby("selection_key")["n_train"].nunique()
    assert (g == 1).all()
    p = out.predictions
    for _key, grp in p[p["condition"] == "clean"].groupby(["fold", "budget", "seed"]):
        ids = [set(q["sample_id"]) for _, q in grp.groupby("method")]
        assert all(s == ids[0] for s in ids)
    # every training selection is from the training subjects of its fold
    subj = dict(zip(labelled["sample_id"], labelled["subject"]))
    for sel in out.selections.values():
        fold = out.folds[sel["fold"]]
        assert {subj[i] for i in sel["sample_ids"]} <= set(fold.train_subjects)


def test_noise_only_touches_training_labels(cfg, labelled):
    out = _run(cfg, labelled)
    noisy = out.results[out.results["condition"] == "noise"]
    assert len(noisy) > 0 and (noisy["n_flipped"] == 4).all()  # 20% of 10 per class -> 2+2
    clean = out.results[out.results["condition"] == "clean"]
    m = noisy.merge(clean, on=["method", "fold", "seed", "budget"], suffixes=("_n", "_c"))
    assert (m["n_test_pos_n"] == m["n_test_pos_c"]).all()  # test labels unchanged
    assert (m["n_test_n"] == m["n_test_c"]).all()
    for sel in out.selections.values():
        for nz in sel.get("noise", {}).values():
            assert set(nz["flipped_ids"]) <= set(sel["sample_ids"])


def test_unequal_eligibility_uses_common_subset(cfg, labelled):
    tables = synthetic_tables(labelled, drop=30)
    tables["geo"].valid[:] = True  # geometry 'sees' everything, embedding drops 30
    tables["geo"].X[np.isnan(tables["geo"].X)] = 0
    out = run_protocol(cfg, labelled, tables, SYNTH_METHODS, "elig")
    assert out.eligibility["geo_poly"].sum() == len(labelled)
    assert out.eligibility["emb_lin"].sum() == len(labelled) - 30
    assert len(out.common_ids) == len(labelled) - 30
    assert not set(out.predictions["sample_id"]) - set(out.common_ids)


def test_result_schema_and_files(cfg, labelled, tmp_path):
    out = _run(cfg, labelled)
    assert list(out.results.columns) == RESULT_FIELDS
    save_protocol(out, tmp_path / "run", {"config": "x"})
    lines = (tmp_path / "run" / "results.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    assert set(rec) == set(RESULT_FIELDS)
    for f in ["results.csv", "predictions.csv.gz", "eligibility.csv", "selections.json.gz", "provenance.json"]:
        assert (tmp_path / "run" / f).exists()


def test_unavailable_budget_is_recorded_not_faked(cfg, labelled):
    cfg["sampling"]["budgets"] = [5, 1000]
    out = _run(cfg, labelled)
    un = out.results[out.results["budget"] == "1000"]
    assert len(un) > 0 and (un["status"] == "unavailable").all() and un["auroc"].isna().all()


def test_full_pipeline_is_deterministic(cfg, labelled):
    a, b = _run(cfg, labelled), _run(cfg, labelled)
    pd.testing.assert_frame_equal(
        a.results.drop(columns=["fit_seconds", "predict_seconds"]), b.results.drop(columns=["fit_seconds", "predict_seconds"])
    )
    pd.testing.assert_frame_equal(a.predictions, b.predictions)


def test_summary_tables(cfg, labelled):
    out = _run(cfg, labelled)
    curves = summarize_curves(out.predictions, labelled, out.results, cfg, list(SYNTH_METHODS), set())
    au = curves[(curves["metric"] == "auroc") & (curves["condition"] == "clean")]
    assert set(au["budget"]) == {"5", "10", "all"}
    assert (au["ci_low"] <= au["estimate"] + 1e-9).all() and (au["estimate"] <= au["ci_high"] + 1e-9).all()
    assert (au["n_test_subjects"] == labelled["subject"].nunique()).all()
    # synthetic geometry is informative -> AUROC well above chance with all data
    est = au[(au["method"] == "geo_poly") & (au["budget"] == "all")]["estimate"].iloc[0]
    assert est > 0.8
    diffs = paired_differences(out.predictions, labelled, cfg, ["10"], [["geo_poly", "emb_lin"]])
    assert len(diffs) == 1 and diffs["ci_low"].iloc[0] <= diffs["delta_auroc"].iloc[0] <= diffs["ci_high"].iloc[0]

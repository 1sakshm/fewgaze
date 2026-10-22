"""The shared evaluation protocol.

For every outer fold f, selection seed s and budget k, ONE training selection is drawn from the
training subjects' commonly-eligible samples and reused by every trained method. Zero-shot
methods are scored once per fold. Every run writes one JSONL record; scores are stored per test
sample so that all summary statistics can be recomputed from disk.
"""

from __future__ import annotations

import gzip
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fewgaze.extraction import MethodSpec
from fewgaze.features import FeatureTable
from fewgaze.gaze_estimator import angular_distance_from_camera
from fewgaze.metrics import all_metrics
from fewgaze.splits import Fold, check_folds, corrupt_labels, make_folds, sample_budget
from fewgaze.train import fit_predict
from fewgaze.utils import get_logger, stable_hash

log = get_logger("fewgaze.evaluate")

RESULT_FIELDS = [
    "run_id",
    "method",
    "kind",
    "condition",
    "noise_rate",
    "budget",
    "seed",
    "fold",
    "n_train",
    "n_train_pos",
    "n_flipped",
    "selection_key",
    "n_test",
    "n_test_pos",
    "n_test_subjects",
    "auroc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "tp",
    "fn",
    "fp",
    "tn",
    "fit_seconds",
    "predict_seconds",
    "status",
    "error",
]


@dataclass
class ProtocolOutput:
    results: pd.DataFrame
    predictions: pd.DataFrame
    selections: dict[str, dict[str, Any]]
    folds: list[Fold]
    eligibility: pd.DataFrame
    common_ids: list[str]
    extra: dict[str, Any] = field(default_factory=dict)


def eligibility_table(data: pd.DataFrame, tables: dict[str, FeatureTable], methods: dict[str, MethodSpec]) -> pd.DataFrame:
    elig = data[["sample_id", "subject", "label"]].copy()
    for name, spec in methods.items():
        t = tables[spec.cache]
        ok = dict(zip(t.sample_ids, t.valid & np.isfinite(t.X).all(axis=1)))
        elig[name] = elig["sample_id"].map(ok).fillna(False).astype(bool)
    elig["common"] = elig[list(methods)].all(axis=1)
    return elig


def zeroshot_scores(spec: MethodSpec, X: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray | None]:
    if spec.name == "l2cs_zeroshot":
        # higher score = closer to camera direction. No decision threshold is defined without labels.
        return -angular_distance_from_camera(X[:, 0], X[:, 1]), None
    if spec.name == "clip_zeroshot":
        p = X[:, 0]
        return p, p >= threshold  # argmax of the two fixed prompts
    raise ValueError(spec.name)


def run_protocol(
    cfg: dict[str, Any],
    data: pd.DataFrame,
    tables: dict[str, FeatureTable],
    methods: dict[str, MethodSpec],
    run_id: str,
    include_noise: bool = True,
) -> ProtocolOutput:
    clf_cfg = cfg["classifier"]
    thr = float(clf_cfg["threshold"])
    budgets = cfg["sampling"]["budgets"]
    seeds = [int(s) for s in cfg["sampling"]["seeds"]]
    noise_rates = [float(r) for r in cfg.get("noise", {}).get("rates", [])] if include_noise else []
    noise_budgets = set(cfg.get("noise", {}).get("budgets", []))

    folds = make_folds(data, int(cfg["splits"]["n_folds"]), int(cfg["splits"]["seed"]))
    problems = check_folds(data, folds)
    if problems:
        raise RuntimeError("Fold invariants violated: " + "; ".join(problems))
    elig = eligibility_table(data, tables, methods)
    common = set(elig.loc[elig["common"], "sample_id"])
    data_c = data[data["sample_id"].isin(common)].reset_index(drop=True)
    label_of = dict(zip(data["sample_id"], data["label"]))
    subj_of = dict(zip(data["sample_id"], data["subject"]))

    records: list[dict[str, Any]] = []
    preds: list[pd.DataFrame] = []
    selections: dict[str, dict[str, Any]] = {}
    trained = [m for m in methods.values() if m.kind == "trained"]
    zeroshot = [m for m in methods.values() if m.kind == "zeroshot"]

    def record(spec: MethodSpec, base: dict[str, Any], test_ids: list[str], score, pred, fit_s, pred_s) -> None:
        y = np.array([label_of[i] for i in test_ids])
        rec = {
            "run_id": run_id,
            "method": spec.name,
            "kind": spec.kind,
            **base,
            "n_test_subjects": len({subj_of[i] for i in test_ids}),
            **all_metrics(y, score, pred),
            "fit_seconds": fit_s,
            "predict_seconds": pred_s,
            "status": "ok",
            "error": "",
        }
        records.append(rec)
        preds.append(
            pd.DataFrame(
                {
                    "method": spec.name,
                    "condition": base["condition"],
                    "noise_rate": base["noise_rate"],
                    "budget": str(base["budget"]),
                    "seed": base["seed"],
                    "fold": base["fold"],
                    "sample_id": test_ids,
                    "score": np.asarray(score, dtype=np.float32),
                }
            )
        )

    def failed(spec: MethodSpec, base: dict[str, Any], status: str, err: str) -> None:
        rec = {k: None for k in RESULT_FIELDS}
        rec.update({"run_id": run_id, "method": spec.name, "kind": spec.kind, **base, "status": status, "error": err})
        records.append(rec)

    for fold in folds:
        train_set, test_set = set(fold.train_ids) & common, set(fold.test_ids) & common
        test_ids = sorted(test_set)
        pool = data_c[data_c["sample_id"].isin(train_set)]
        for spec in zeroshot:
            Xte = tables[spec.cache].subset(test_ids)
            t0 = time.perf_counter()
            score, pred = zeroshot_scores(spec, Xte, thr)
            base = {
                "condition": "clean",
                "noise_rate": 0.0,
                "budget": "0",
                "seed": -1,
                "fold": fold.index,
                "n_train": 0,
                "n_train_pos": 0,
                "n_flipped": 0,
                "selection_key": "",
            }
            record(spec, base, test_ids, score, pred, 0.0, time.perf_counter() - t0)

        for seed in seeds:
            for k in budgets:
                sel = sample_budget(pool, k, seed, fold.index)
                sel_key = f"f{fold.index}_s{seed}_k{k}"
                base0 = {"budget": str(k), "seed": seed, "fold": fold.index, "selection_key": sel_key}
                if sel is None:
                    for spec in trained:
                        failed(
                            spec,
                            {**base0, "condition": "clean", "noise_rate": 0.0},
                            "unavailable",
                            f"fewer than {k} eligible training samples in a class",
                        )
                    continue
                y_clean = np.array([label_of[i] for i in sel])
                selections[sel_key] = {
                    "fold": fold.index,
                    "seed": seed,
                    "budget": str(k),
                    "sample_ids": sel,
                    "hash": stable_hash(sel),
                    "n_pos": int(y_clean.sum()),
                    "n_neg": int((1 - y_clean).sum()),
                }
                conditions = [("clean", 0.0, y_clean, 0)]
                if k in noise_budgets:
                    bkey = 0 if k == "all" else int(k)
                    for rate in noise_rates:
                        y_noisy, mask = corrupt_labels(y_clean, rate, seed, fold.index, bkey)
                        selections[sel_key].setdefault("noise", {})[str(rate)] = {"flipped_ids": [s for s, m in zip(sel, mask) if m]}
                        conditions.append(("noise", rate, y_noisy, int(mask.sum())))
                for spec in trained:
                    tab = tables[spec.cache]
                    Xtr, Xte = tab.subset(sel), tab.subset(test_ids)
                    for cond, rate, ytr, n_flip in conditions:
                        base = {
                            **base0,
                            "condition": cond,
                            "noise_rate": rate,
                            "n_train": len(sel),
                            "n_train_pos": int(ytr.sum()),
                            "n_flipped": n_flip,
                        }
                        try:
                            score, _, fit_s, pred_s = fit_predict(Xtr, ytr, Xte, clf_cfg, spec.poly, seed)
                            record(spec, base, test_ids, score, score >= thr, fit_s, pred_s)
                        except Exception as exc:  # recorded, never hidden
                            failed(spec, base, "error", f"{type(exc).__name__}: {exc}")
                            log.error("%s %s failed: %s", spec.name, sel_key, traceback.format_exc(limit=2))
            log.info("fold %d seed %d done (%d records)", fold.index, seed, len(records))

    results = pd.DataFrame(records)[RESULT_FIELDS]
    predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return ProtocolOutput(results, predictions, selections, folds, elig, sorted(common))


def save_protocol(out: ProtocolOutput, run_dir: Path, provenance: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for rec in out.results.to_dict(orient="records"):
            fh.write(json.dumps(_clean(rec)) + "\n")
    out.results.to_csv(run_dir / "results.csv", index=False)
    out.predictions.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
    out.eligibility.to_csv(run_dir / "eligibility.csv", index=False)
    folds = [
        {
            "fold": f.index,
            "train_subjects": f.train_subjects,
            "test_subjects": f.test_subjects,
            "n_train": len(f.train_ids),
            "n_test": len(f.test_ids),
        }
        for f in out.folds
    ]
    with gzip.open(run_dir / "selections.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(out.selections, fh)
    with (run_dir / "provenance.json").open("w", encoding="utf-8") as fh:
        json.dump(_clean({**provenance, "folds": folds, "n_common_eligible": len(out.common_ids)}), fh, indent=2, default=str)


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    return obj

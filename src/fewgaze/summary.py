"""Aggregate saved per-sample predictions into the reported tables.

Everything here is recomputed from ``predictions.csv.gz`` + the label table, so every number
in the report can be regenerated from disk.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fewgaze.metrics import auroc
from fewgaze.statistics import FoldBlock, bootstrap, metric_fn, percentile_ci, point_estimate


def budget_order(b: str) -> float:
    return {"0": 0.0, "all": float("inf")}.get(str(b), float(b) if str(b).replace(".", "").isdigit() else np.nan)


def build_blocks(
    preds: pd.DataFrame,
    meta: pd.DataFrame,
    methods: list[str],
    condition: str,
    noise_rate: float,
    budget: str,
    subset: pd.Series | None = None,
) -> list[FoldBlock] | None:
    """FoldBlocks for one (condition, noise_rate, budget). Zero-shot methods are looked up at budget '0'."""
    blocks = []
    meta_i = meta.set_index("sample_id")
    for fold in sorted(preds["fold"].unique()):
        scores = {}
        ref_ids = None
        for m in methods:
            q = preds[(preds["method"] == m) & (preds["fold"] == fold)]
            if (q["budget"] == "0").all() and len(q):
                q = q[q["condition"] == "clean"]
            else:
                q = q[(q["condition"] == condition) & (np.isclose(q["noise_rate"], noise_rate)) & (q["budget"] == budget)]
            if q.empty:
                return None
            wide = q.pivot(index="sample_id", columns="seed", values="score").sort_index()
            if subset is not None:
                wide = wide[wide.index.isin(subset[subset].index)]
            if ref_ids is None:
                ref_ids = wide.index
            elif not wide.index.equals(ref_ids):
                raise RuntimeError(f"Methods were evaluated on different test samples in fold {fold}")
            scores[m] = wide.to_numpy().T
        if ref_ids is None or len(ref_ids) == 0:
            continue
        blocks.append(
            FoldBlock(
                subjects=meta_i.loc[ref_ids, "subject"].to_numpy(),
                labels=meta_i.loc[ref_ids, "label"].to_numpy(),
                scores=scores,
            )
        )
    return blocks


def summarize_curves(
    preds: pd.DataFrame,
    meta: pd.DataFrame,
    results: pd.DataFrame,
    cfg: dict[str, Any],
    methods: list[str],
    zeroshot_with_pred: set[str],
) -> pd.DataFrame:
    st = cfg["statistics"]
    thr = float(cfg["classifier"]["threshold"])
    rows = []
    ok = results[results["status"] == "ok"]
    keys = ok[["method", "condition", "noise_rate", "budget"]].drop_duplicates()
    for (cond, rate, budget), grp in keys.groupby(["condition", "noise_rate", "budget"]):
        ms = [m for m in methods if m in set(grp["method"])]
        if not ms:
            continue
        blocks = build_blocks(preds, meta, ms, cond, rate, budget)
        if not blocks:
            continue
        for metric in ("auroc", "balanced_accuracy"):
            fn = metric_fn(metric, thr)
            use = [m for m in ms if metric == "auroc" or budget != "0" or m in zeroshot_with_pred]
            if not use:
                continue
            dist = bootstrap(blocks, use, fn, int(st["n_bootstrap"]), int(st["bootstrap_seed"]))
            for m in use:
                r = ok[(ok["method"] == m) & (ok["condition"] == cond) & np.isclose(ok["noise_rate"], rate) & (ok["budget"] == budget)]
                lo, hi = percentile_ci(dist[m], float(st["ci_level"]))
                rows.append(
                    {
                        "method": m,
                        "condition": cond,
                        "noise_rate": rate,
                        "budget": budget,
                        "metric": metric,
                        "estimate": point_estimate(blocks, m, fn),
                        "ci_low": lo,
                        "ci_high": hi,
                        "sd_across_runs": float(r[metric].std(ddof=1)) if len(r) > 1 else float("nan"),
                        "n_runs": int(len(r)),
                        "n_folds": int(r["fold"].nunique()),
                        "n_seeds": int(r["seed"].nunique()),
                        "mean_n_train": float(r["n_train"].mean()),
                        "n_test_images": int(sum(len(b.labels) for b in blocks)),
                        "n_test_subjects": int(sum(len(np.unique(b.subjects)) for b in blocks)),
                        "mean_sensitivity": float(r["sensitivity"].mean()),
                        "mean_specificity": float(r["specificity"].mean()),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["budget_order"] = out["budget"].map(budget_order)
        out = out.sort_values(["metric", "condition", "noise_rate", "method", "budget_order"]).reset_index(drop=True)
    return out


def paired_differences(
    preds: pd.DataFrame,
    meta: pd.DataFrame,
    cfg: dict[str, Any],
    budgets: list[str],
    pairs: list[list[str]],
) -> pd.DataFrame:
    st = cfg["statistics"]
    rows = []
    for a, b in pairs:
        for budget in budgets:
            blocks = build_blocks(preds, meta, [a, b], "clean", 0.0, budget)
            if not blocks:
                continue
            dist = bootstrap(blocks, [a, b], metric_fn("auroc"), int(st["n_bootstrap"]), int(st["bootstrap_seed"]))
            diff = dist[a] - dist[b]
            lo, hi = percentile_ci(diff, float(st["ci_level"]))
            est = point_estimate(blocks, a, metric_fn("auroc")) - point_estimate(blocks, b, metric_fn("auroc"))
            rows.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "budget": budget,
                    "delta_auroc": est,
                    "ci_low": lo,
                    "ci_high": hi,
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "frac_boot_a_better": float(np.mean(diff > 0)),
                }
            )
    return pd.DataFrame(rows)


def by_head_pose(preds: pd.DataFrame, meta: pd.DataFrame, cfg: dict[str, Any], methods: list[str], budgets: list[str]) -> pd.DataFrame:
    st = cfg["statistics"]
    rows = []
    meta_i = meta.set_index("sample_id")
    for budget in budgets:
        for pose in sorted(meta["head_pose"].unique()):
            subset = meta_i["head_pose"] == pose
            blocks = build_blocks(preds, meta, methods, "clean", 0.0, budget, subset=subset)
            if not blocks:
                continue
            dist = bootstrap(blocks, methods, metric_fn("auroc"), int(st["n_bootstrap"]) // 2, int(st["bootstrap_seed"]))
            for m in methods:
                lo, hi = percentile_ci(dist[m], float(st["ci_level"]))
                rows.append(
                    {
                        "method": m,
                        "budget": budget,
                        "head_pose": int(pose),
                        "auroc": point_estimate(blocks, m, metric_fn("auroc")),
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_images": int(sum(len(b.labels) for b in blocks)),
                        "n_pos": int(sum(b.labels.sum() for b in blocks)),
                    }
                )
    return pd.DataFrame(rows)


def per_subject(preds: pd.DataFrame, meta: pd.DataFrame, methods: list[str], budget: str) -> pd.DataFrame:
    """Per-subject AUROC (each subject is tested in exactly one fold), averaged over seeds."""
    rows = []
    for m in methods:
        q = preds[(preds["method"] == m) & (preds["condition"] == "clean")]
        q = q[q["budget"] == ("0" if (q["budget"] == "0").all() else budget)]
        q = q.merge(meta[["sample_id", "subject", "label"]], on="sample_id")
        for (subj, seed), g in q.groupby(["subject", "seed"]):
            rows.append(
                {
                    "method": m,
                    "subject": subj,
                    "seed": seed,
                    "auroc": auroc(g["label"].to_numpy(), g["score"].to_numpy()),
                    "n_images": len(g),
                    "n_pos": int(g["label"].sum()),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["method", "subject"], as_index=False).agg(
        auroc=("auroc", "mean"), n_images=("n_images", "first"), n_pos=("n_pos", "first")
    )

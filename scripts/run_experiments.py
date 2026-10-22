"""Run the shared subject-independent protocol for every method whose features are cached,
then compute the summary tables (learning curves with subject-bootstrap CIs, paired
differences, head-pose and per-subject breakdowns).

Outputs: results/runs/<run_id>/{results.jsonl,results.csv,predictions.csv.gz,selections.json.gz,
         eligibility.csv,provenance.json,method_status.json}
         results/tables/*.csv  and results/runs/LATEST

Exit codes: 0 ok, 2 dataset problem, 5 no method has usable features.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from fewgaze.config import load_config, output_dir
from fewgaze.dataset import DatasetError, load_manifest, manifest_fingerprint
from fewgaze.evaluate import run_protocol, save_protocol
from fewgaze.extraction import cache_keys, method_registry, open_cache
from fewgaze.features import CacheError
from fewgaze.labels import derive_labels, labelled_view
from fewgaze.summary import by_head_pose, paired_differences, per_subject, summarize_curves
from fewgaze.utils import get_logger, git_commit, runtime_info, stable_hash, utc_now, write_json

log = get_logger("fewgaze.run")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", nargs="*", default=None, help="subset of methods (default: all enabled)")
    ap.add_argument("--no-noise", action="store_true", help="skip the label-noise conditions")
    ap.add_argument("--summary-only", action="store_true", help="recompute tables from the LATEST run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_root = output_dir(cfg)
    try:
        man = load_manifest(cfg)
    except DatasetError as exc:
        log.error("%s", exc)
        return 2
    labels = derive_labels(man.records, cfg["labels"])
    data = labelled_view(man.records, labels)
    fingerprint = manifest_fingerprint(man.records)
    keys = cache_keys(cfg, fingerprint)
    registry = method_registry(cfg)
    if args.methods:
        registry = {k: v for k, v in registry.items() if k in args.methods}

    if args.summary_only:
        run_dir = out_root / "runs" / (out_root / "runs" / "LATEST").read_text().strip()
        results = pd.read_csv(run_dir / "results.csv", dtype={"budget": str})
        preds = pd.read_csv(run_dir / "predictions.csv.gz", dtype={"budget": str, "sample_id": str})
        methods = [m for m in registry if m in set(results["method"])]
        summarize(cfg, registry, methods, results, preds, data, out_root)
        return 0

    tables, status = {}, {}
    ids = man.records["sample_id"].tolist()
    for name, spec in registry.items():
        try:
            if spec.cache not in tables:
                tables[spec.cache] = open_cache(cfg, keys, spec.cache).load(ids)
            status[name] = {"status": "run", "cache": spec.cache, "key": keys[spec.cache]}
        except CacheError as exc:
            status[name] = {"status": "not_run", "reason": str(exc)}
            log.warning("method %s skipped: %s", name, exc)
    methods = {k: v for k, v in registry.items() if status[k]["status"] == "run"}
    if not methods:
        log.error("No method has complete features. Run scripts/extract_features.py first.")
        return 5

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + stable_hash(cfg)[:6]
    t0 = time.perf_counter()
    out = run_protocol(cfg, data, tables, methods, run_id, include_noise=not args.no_noise)
    provenance = {
        "run_id": run_id,
        "created": utc_now(),
        "git_commit": git_commit(),
        "config": cfg,
        "config_hash": stable_hash(cfg),
        "manifest_fingerprint": fingerprint,
        "cache_keys": keys,
        "methods": {k: {**v.__dict__} for k, v in methods.items()},
        "runtime": runtime_info(),
        "feature_meta": {n: {k: v for k, v in t.meta.items() if k != "feature_names"} for n, t in tables.items()},
        "protocol_seconds": time.perf_counter() - t0,
    }
    run_dir = out_root / "runs" / run_id
    save_protocol(out, run_dir, provenance)
    write_json(status, run_dir / "method_status.json")
    (out_root / "runs" / "LATEST").write_text(run_id, encoding="utf-8")
    n_ok = int((out.results["status"] == "ok").sum())
    log.info("run %s: %d ok records, %d not ok, %.0f s", run_id, n_ok, len(out.results) - n_ok, time.perf_counter() - t0)
    summarize(cfg, registry, list(methods), out.results, out.predictions, data, out_root)
    return 0


def summarize(cfg, registry, methods, results, preds, data, out_root) -> None:
    t0 = time.perf_counter()
    tdir = out_root / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    zs_pred = {m for m in methods if registry[m].kind == "zeroshot" and m == "clip_zeroshot"}
    curves = summarize_curves(preds, data, results, cfg, methods, zs_pred)
    curves.to_csv(tdir / "learning_curves.csv", index=False)
    log.info("learning curves: %d rows (%.0f s)", len(curves), time.perf_counter() - t0)
    budgets = [str(b) for b in cfg["sampling"]["budgets"]]
    pairs = [p for p in cfg["statistics"].get("paired_comparisons", []) if all(m in methods for m in p)]
    paired_differences(preds, data, cfg, budgets, pairs).to_csv(tdir / "paired_differences.csv", index=False)
    focus = str(cfg["analysis"]["focus_budget"])
    primary = [m for m in methods if registry[m].primary]
    by_head_pose(preds, data, cfg, primary, [focus, "all"]).to_csv(tdir / "head_pose_auroc.csv", index=False)
    per_subject(preds, data, primary, focus).to_csv(tdir / "per_subject_auroc.csv", index=False)
    ok = results[results["status"] == "ok"]
    ok.groupby(["method", "condition", "noise_rate", "budget"], as_index=False).agg(
        fit_seconds_mean=("fit_seconds", "mean"), predict_seconds_mean=("predict_seconds", "mean"), n_runs=("fold", "size")
    ).to_csv(tdir / "classifier_timing.csv", index=False)
    results.groupby(["method", "status"]).size().rename("n").reset_index().to_csv(tdir / "run_status_counts.csv", index=False)
    log.info("summary tables written to %s (%.0f s)", tdir, time.perf_counter() - t0)


if __name__ == "__main__":
    sys.exit(main())

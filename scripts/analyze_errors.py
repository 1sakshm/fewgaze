"""Error analysis on the LATEST run at the configured focus budget.

Outputs (public, no images):
  results/tables/errors_<method>.csv           per-sample mean score, prediction, metadata category
  results/tables/error_examples.csv            prespecified example selection (image IDs only)
  results/tables/error_rates_by_condition.csv  predicted-positive rate per head pose x gaze cell
  results/tables/error_category_counts.csv
  results/tables/detection_failures.csv        images where MediaPipe found no face
  results/tables/confusion_matrices.json
Local-only (gitignored, contains dataset imagery):
  results/figures/local_only/fig9_error_examples.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from fewgaze.config import load_config, output_dir
from fewgaze.dataset import load_manifest, manifest_fingerprint
from fewgaze.errors import error_table, rates_by_condition, select_examples
from fewgaze.extraction import cache_keys, crop_dir, method_registry
from fewgaze.labels import derive_labels, labelled_view
from fewgaze.metrics import confusion
from fewgaze.utils import get_logger, write_json

log = get_logger("fewgaze.errors")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--budget", default=None, help="override analysis.focus_budget")
    ap.add_argument("--no-images", action="store_true", help="skip the local-only example montage")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    run_dir = out / "runs" / (out / "runs" / "LATEST").read_text().strip()
    preds = pd.read_csv(run_dir / "predictions.csv.gz", dtype={"budget": str, "sample_id": str})
    elig = pd.read_csv(run_dir / "eligibility.csv", dtype={"sample_id": str, "subject": str})
    man = load_manifest(cfg)
    data = labelled_view(man.records, derive_labels(man.records, cfg["labels"]))
    budget = str(args.budget or cfg["analysis"]["focus_budget"])
    thr = float(cfg["classifier"]["threshold"])
    reg = method_registry(cfg)
    methods = [m for m in reg if reg[m].primary and m in set(preds["method"])]
    tdir = out / "tables"

    errs, confs = [], {}
    for m in methods:
        e = error_table(preds, data, m, budget, thr)
        e.to_csv(tdir / f"errors_{m}.csv", index=False)
        errs.append(e)
        # confusion of seed-0 predictions pooled over the 5 test folds (each image counted once)
        q = preds[(preds["method"] == m) & (preds["condition"] == "clean")]
        q = q[q["budget"] == ("0" if (q["budget"] == "0").all() else budget)]
        q = q[q["seed"] == q["seed"].min()].merge(data[["sample_id", "label"]], on="sample_id")
        if reg[m].kind == "zeroshot" and m != "clip_zeroshot":
            continue  # no decision rule without labels
        confs[m] = {
            **confusion(q["label"].to_numpy(), q["score"].to_numpy() >= thr),
            "budget": "0" if reg[m].kind == "zeroshot" else budget,
            "seed": int(q["seed"].min()),
            "n_images": len(q),
        }
    write_json(confs, tdir / "confusion_matrices.json")
    allerr = pd.concat(errs, ignore_index=True)
    trained = [m for m in methods if reg[m].kind == "trained"] + (["clip_zeroshot"] if "clip_zeroshot" in methods else [])
    rates = pd.concat([rates_by_condition(allerr[allerr["method"] == m]).assign(method=m) for m in trained], ignore_index=True)
    rates.to_csv(tdir / "error_rates_by_condition.csv", index=False)
    cat = allerr[(allerr["error"] != "") & allerr["method"].isin(trained)].groupby(["method", "error", "category"]).size()
    cat.rename("n").reset_index().to_csv(tdir / "error_category_counts.csv", index=False)

    n_trained = max(1, len([m for m in trained if reg[m].kind == "trained"]))
    n_per = max(1, -(-int(cfg["analysis"]["n_error_examples"]) // (2 * n_trained)))  # ceil
    ex = pd.concat([select_examples(allerr[allerr["method"] == m], n_per) for m in trained if reg[m].kind == "trained"], ignore_index=True)
    cols = ["method", "sample_id", "subject", "head_pose", "gaze_v", "gaze_h", "label", "score", "score_sd", "error", "category"]
    ex[cols].to_csv(tdir / "error_examples.csv", index=False)

    geo_fail = elig[~elig["common"]][["sample_id", "subject"]].merge(
        data[["sample_id", "head_pose", "gaze_v", "gaze_h", "label"]], on="sample_id"
    )
    geo_fail.to_csv(tdir / "detection_failures.csv", index=False)
    log.info("%d detection failures; %d examples selected", len(geo_fail), len(ex))

    if not args.no_images and len(ex):
        keys = cache_keys(cfg, manifest_fingerprint(man.records))
        montage(ex, crop_dir(cfg, keys), out / "figures" / "local_only" / "fig9_error_examples", budget)
    return 0


def montage(ex: pd.DataFrame, crops, stem, budget: str) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    from fewgaze.plots import apply_style, save

    apply_style()
    n = len(ex)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.0))
    for ax, (_, r) in zip(np.ravel(axes), ex.iterrows()):
        ax.imshow(Image.open(crops / f"{r.sample_id}.jpg"))
        ax.set_title(f"{r.method} {r.error}  p={r.score:.2f}\nP={r.head_pose:+d} V={r.gaze_v:+d} H={r.gaze_h:+d}", fontsize=7)
        ax.axis("off")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle(f"Most confident errors at k={budget} per class (mean score over seeds) — LOCAL ONLY, not for redistribution", fontsize=8)
    save(fig, stem)


if __name__ == "__main__":
    sys.exit(main())

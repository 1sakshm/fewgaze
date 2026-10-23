"""Extra data for the project website (site/data/extras.json). Aggregates only: no images, no per-image rows.

* geometry_by_gaze: median iris offsets / eye opening per gaze angle at frontal head pose (real measurements,
  drives the "how small is 5 degrees" visual)
* study2: layer-wise decoding/readout summaries, if results/study2 exists
* demo: cross-validated performance of the demo model
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from fewgaze.config import REPO_ROOT, load_config, output_dir
from fewgaze.dataset import load_manifest, manifest_fingerprint
from fewgaze.extraction import cache_keys, open_cache
from fewgaze.geometry import FEATURE_NAMES
from fewgaze.labels import derive_labels, labelled_view
from fewgaze.probing import MODELS
from fewgaze.utils import get_logger, read_json, write_json

log = get_logger("fewgaze.site")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiment.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    man = load_manifest(cfg)
    data = labelled_view(man.records, derive_labels(man.records, cfg["labels"]))
    keys = cache_keys(cfg, manifest_fingerprint(man.records))
    geo = open_cache(cfg, keys, "geometry").load(man.records["sample_id"].tolist())
    X = pd.DataFrame(geo.subset(data["sample_id"]), columns=FEATURE_NAMES)
    df = pd.concat([data[["subject", "head_pose", "gaze_v", "gaze_h", "label"]].reset_index(drop=True), X], axis=1)
    front = df[df["head_pose"] == 0]
    # mean of the two eyes; median over subjects at each gaze cell
    front = front.assign(
        iris_x=(front["right_iris_x"] + front["left_iris_x"]) / 2,
        iris_y=(front["right_iris_y"] + front["left_iris_y"]) / 2,
        opening=(front["right_aperture"] + front["left_aperture"]) / 2,
    )
    cells = (
        front.groupby(["gaze_v", "gaze_h"])
        .agg(
            iris_x=("iris_x", "median"),
            iris_y=("iris_y", "median"),
            opening=("opening", "median"),
            iris_x_iqr=("iris_x", lambda s: float(s.quantile(0.75) - s.quantile(0.25))),
            n=("iris_x", "size"),
        )
        .reset_index()
    )
    extras = {
        "geometry_by_gaze": {
            "head_pose": 0,
            "note": "median over subjects; iris offsets in eye-widths",
            "cells": cells.round(5).to_dict(orient="records"),
        },
    }
    s2 = output_dir(cfg) / "study2"
    if (s2 / "decoding.csv").exists():
        dec = pd.read_csv(s2 / "decoding.csv")
        rd = pd.read_csv(s2 / "readout.csv")
        ext = read_json(s2 / "extraction.json") if (s2 / "extraction.json").exists() else {}
        extras["study2"] = {
            "models": {
                m.key: {
                    "label": m.label,
                    **{k: ext.get(m.key, {}).get(k) for k in ("params", "layers", "grid", "eye_patches_per_image_mean")},
                }
                for m in MODELS
                if m.key in set(dec["model"])
            },
            "decoding": dec.round(4).to_dict(orient="records"),
            "readout": rd.drop(columns=["n"], errors="ignore").round(4).to_dict(orient="records"),
        }
        if (s2 / "nested_layer_summary.csv").exists():
            extras["study2"]["nested"] = pd.read_csv(s2 / "nested_layer_summary.csv").round(4).to_dict(orient="records")
    model = read_json(REPO_ROOT / "site" / "demo" / "model.json")
    extras["demo"] = model["cross_validated"]
    path = REPO_ROOT / "site" / "data" / "extras.json"
    write_json(extras, path)
    log.info("wrote %s (%.0f kB)", path, path.stat().st_size / 1024)
    print(cells[cells["gaze_v"] == 0][["gaze_h", "iris_x", "iris_x_iqr", "n"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

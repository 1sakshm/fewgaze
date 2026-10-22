"""Export the geometry classifier for the in-browser demo (site/demo).

Fits the prespecified geometry pipeline (standardise -> degree-2 polynomial -> standardise -> logistic
regression, C=1, balanced weights) on ALL commonly-eligible Columbia images, i.e. every subject. This is
the model a user would deploy; its expected performance on new people is the cross-validated estimate
at k=ALL from the main study, which is copied into the JSON together with the run it came from.

Output: site/demo/model.json (no image data; parameters and feature ranges only).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from fewgaze.config import REPO_ROOT, load_config, output_dir
from fewgaze.dataset import load_manifest, manifest_fingerprint
from fewgaze.extraction import cache_keys, open_cache
from fewgaze.geometry import FEATURE_NAMES
from fewgaze.labels import derive_labels, labelled_view
from fewgaze.train import build_classifier
from fewgaze.utils import get_logger, git_commit, utc_now, write_json

log = get_logger("fewgaze.export")


def export(model, X: np.ndarray, y: np.ndarray, cfg: dict, cv: dict) -> dict:
    s1, poly, s2, lr = (step for _, step in model.steps)
    return {
        "format": "fewgaze-geometry-v1",
        "created": utc_now(),
        "git_commit": git_commit(),
        "feature_names": FEATURE_NAMES,
        "scaler1": {"mean": s1.mean_.tolist(), "scale": s1.scale_.tolist()},
        "poly_powers": poly.powers_.astype(int).tolist(),
        "scaler2": {"mean": s2.mean_.tolist(), "scale": s2.scale_.tolist()},
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "threshold": float(cfg["classifier"]["threshold"]),
        "training": {
            "n_images": int(len(y)),
            "n_positive": int(y.sum()),
            "n_subjects": cv.pop("n_subjects"),
            "feature_p01": np.percentile(X, 1, axis=0).tolist(),
            "feature_p99": np.percentile(X, 99, axis=0).tolist(),
        },
        "cross_validated": cv,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--out", default=str(REPO_ROOT / "site" / "demo" / "model.json"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    man = load_manifest(cfg)
    data = labelled_view(man.records, derive_labels(man.records, cfg["labels"]))
    keys = cache_keys(cfg, manifest_fingerprint(man.records))
    geo = open_cache(cfg, keys, "geometry").load(man.records["sample_id"].tolist())
    ok = dict(zip(geo.sample_ids, geo.valid))
    data = data[data["sample_id"].map(ok)].reset_index(drop=True)
    X = geo.subset(data["sample_id"]).astype(float)
    y = data["label"].to_numpy()
    model = build_classifier(cfg["classifier"], int(cfg["methods"]["geometry"]["polynomial_degree"]), seed=0)
    model.fit(X, y)

    out = output_dir(cfg)
    run_id = (out / "runs" / "LATEST").read_text().strip()
    lc = pd.read_csv(out / "tables" / "learning_curves.csv", dtype={"budget": str})
    sel = lc[(lc["method"] == "geometry") & (lc["condition"] == "clean") & (lc["budget"] == "all")].set_index("metric")
    cv = {
        "run_id": run_id,
        "auroc": float(sel.loc["auroc", "estimate"]),
        "auroc_ci": [float(sel.loc["auroc", "ci_low"]), float(sel.loc["auroc", "ci_high"])],
        "balanced_accuracy": float(sel.loc["balanced_accuracy", "estimate"]),
        "sensitivity": float(sel.loc["auroc", "mean_sensitivity"]),
        "specificity": float(sel.loc["auroc", "mean_specificity"]),
        "n_subjects": int(data["subject"].nunique()),
    }
    blob = export(model, X, y, cfg, cv)
    # self-check: the exported parameters reproduce sklearn's probabilities exactly
    p_ref = model.predict_proba(X)[:, 1]
    p_exp = predict_exported(blob, X)
    err = float(np.abs(p_ref - p_exp).max())
    if err > 1e-9:
        log.error("exported parameters disagree with sklearn (max |dp| = %.3g)", err)
        return 1
    blob["self_check_max_abs_prob_diff"] = err
    write_json(blob, args.out)
    log.info("wrote %s (%d images, max |dp| = %.2g)", args.out, len(y), err)
    return 0


def predict_exported(m: dict, X: np.ndarray) -> np.ndarray:
    """Reference implementation of the JS predictor (site/demo/demo.js: predict)."""
    z = (X - np.array(m["scaler1"]["mean"])) / np.array(m["scaler1"]["scale"])
    powers = np.array(m["poly_powers"])
    P = np.prod(z[:, None, :] ** powers[None, :, :], axis=2)
    P = (P - np.array(m["scaler2"]["mean"])) / np.array(m["scaler2"]["scale"])
    logit = P @ np.array(m["coef"]) + m["intercept"]
    return 1 / (1 + np.exp(-logit))


if __name__ == "__main__":
    sys.exit(main())

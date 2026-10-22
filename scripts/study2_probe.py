"""Study 2, step 2: probes on every model x layer x representation (docs/methodology.md, Study 2).

Outputs: results/study2/decoding.csv  (ridge R^2 for H, V, head pose; pooled over held-out subjects + per-fold SD)
         results/study2/readout.csv   (camera-directed AUROC: linear logistic probe, and gaze readout from ridge)
The nine Study-1 geometry features are probed identically as a reference ("geometry", layer -1).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fewgaze.config import cache_dir, load_config, output_dir
from fewgaze.dataset import load_manifest, manifest_fingerprint
from fewgaze.extraction import cache_keys, open_cache
from fewgaze.labels import derive_labels, labelled_view
from fewgaze.metrics import auroc
from fewgaze.probing import MODELS
from fewgaze.probing import cache_dir as s2_dir
from fewgaze.splits import make_folds
from fewgaze.utils import get_logger

log = get_logger("fewgaze.study2")
TARGETS = ["gaze_h", "gaze_v", "head_pose"]
ALPHAS = np.logspace(-2, 5, 8)


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def probe(X, data, folds, clf_cfg, do_linear=True):
    """Return (decoding rows, readout rows) for one representation matrix X aligned with `data`."""
    Y = data[TARGETS].to_numpy(float)
    lab = data["label"].to_numpy()
    idx_of = {s: i for i, s in enumerate(data["sample_id"])}
    pred = np.zeros_like(Y)
    fold_r2 = {t: [] for t in TARGETS}
    auc_read, auc_lin = [], []
    for f in folds:
        tr = np.array([idx_of[s] for s in f.train_ids if s in idx_of])
        te = np.array([idx_of[s] for s in f.test_ids if s in idx_of])
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, alpha_per_target=True))
        ridge.fit(X[tr], Y[tr])
        p = ridge.predict(X[te])
        pred[te] = p
        for j, t in enumerate(TARGETS):
            fold_r2[t].append(r2(Y[te, j], p[:, j]))
        sh, sv = Y[tr, 0].std(), Y[tr, 1].std()
        auc_read.append(auroc(lab[te], -((p[:, 0] / sh) ** 2 + (p[:, 1] / sv) ** 2)))
        if do_linear:
            lin = make_pipeline(StandardScaler(), LogisticRegression(C=float(clf_cfg["C"]), class_weight="balanced", max_iter=3000))
            lin.fit(X[tr], lab[tr])
            auc_lin.append(auroc(lab[te], lin.predict_proba(X[te])[:, 1]))
    dec = [
        {
            "target": t,
            "r2_pooled": r2(Y[:, j], pred[:, j]),
            "r2_fold_mean": float(np.mean(fold_r2[t])),
            "r2_fold_sd": float(np.std(fold_r2[t], ddof=1)),
        }
        for j, t in enumerate(TARGETS)
    ]
    rd = [{"readout": "gaze_readout", "auroc_mean": float(np.mean(auc_read)), "auroc_sd": float(np.std(auc_read, ddof=1))}]
    if do_linear:
        rd.append({"readout": "linear_probe", "auroc_mean": float(np.mean(auc_lin)), "auroc_sd": float(np.std(auc_lin, ddof=1))})
    return dec, rd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", default=[m.key for m in MODELS])
    args = ap.parse_args()
    cfg = load_config(args.config)
    man = load_manifest(cfg)
    data = labelled_view(man.records, derive_labels(man.records, cfg["labels"]))
    folds = make_folds(data, int(cfg["splits"]["n_folds"]), int(cfg["splits"]["seed"]))  # identical to Study 1
    out = output_dir(cfg) / "study2"
    out.mkdir(parents=True, exist_ok=True)
    dec_rows, rd_rows = [], []
    dec_path, rd_path = out / "decoding.csv", out / "readout.csv"
    done = set()
    if dec_path.exists():
        prev_d, prev_r = pd.read_csv(dec_path), pd.read_csv(rd_path)
        dec_rows, rd_rows = prev_d.to_dict("records"), prev_r.to_dict("records")
        done = set(prev_d["model"].unique())

    if "geometry" not in done:
        keys = cache_keys(cfg, manifest_fingerprint(man.records))
        geo = open_cache(cfg, keys, "geometry").load(man.records["sample_id"].tolist())
        X = geo.subset(data["sample_id"]).astype(float)
        d, r = probe(X, data, folds, cfg["classifier"])
        dec_rows += [{"model": "geometry", "layer": -1, "rep": "geometry", **x} for x in d]
        rd_rows += [{"model": "geometry", "layer": -1, "rep": "geometry", **x} for x in r]

    for spec in [m for m in MODELS if m.key in args.models and m.key not in done]:
        path = s2_dir(cache_dir(cfg)) / f"{spec.key}.npz"
        if not path.exists():
            log.warning("%s: no cached features (run study2_extract.py)", spec.key)
            continue
        z = np.load(path)
        pos = {s: i for i, s in enumerate(z["sample_ids"])}
        ok = z["ok"]
        sub = data[[ok[pos[s]] for s in data["sample_id"]]].reset_index(drop=True)
        rows = np.array([pos[s] for s in sub["sample_id"]])
        for rep in ("global", "eyes"):
            arr = z["glob" if rep == "global" else "eyes"]
            for layer in range(arr.shape[1]):
                t0 = time.perf_counter()
                X = arr[rows, layer, :].astype(np.float32)
                d, r = probe(X, sub, folds, cfg["classifier"])
                dec_rows += [{"model": spec.key, "layer": layer, "rep": rep, **x} for x in d]
                rd_rows += [{"model": spec.key, "layer": layer, "rep": rep, "n": len(sub), **x} for x in r]
                log.info(
                    "%s %s L%02d: R2(H)=%.3f R2(V)=%.3f R2(P)=%.3f AUROC lin=%.3f readout=%.3f (%.0f s)",
                    spec.key,
                    rep,
                    layer,
                    d[0]["r2_pooled"],
                    d[1]["r2_pooled"],
                    d[2]["r2_pooled"],
                    r[1]["auroc_mean"],
                    r[0]["auroc_mean"],
                    time.perf_counter() - t0,
                )
        pd.DataFrame(dec_rows).to_csv(dec_path, index=False)
        pd.DataFrame(rd_rows).to_csv(rd_path, index=False)
    pd.DataFrame(dec_rows).to_csv(dec_path, index=False)
    pd.DataFrame(rd_rows).to_csv(rd_path, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

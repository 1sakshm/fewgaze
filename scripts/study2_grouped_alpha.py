"""Study 2 sensitivity analysis S2-b (docs/methodology.md): ridge decoding with alpha chosen by person-grouped
inner cross-validation instead of leave-one-image-out. Eye-patch representation, every layer, every model, plus the
nine geometry features as reference.

Output: results/study2/decoding_grouped_alpha.csv
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from fewgaze.config import cache_dir, load_config, output_dir
from fewgaze.dataset import load_manifest, manifest_fingerprint
from fewgaze.extraction import cache_keys, open_cache
from fewgaze.labels import derive_labels, labelled_view
from fewgaze.probing import MODELS
from fewgaze.probing import cache_dir as s2_dir
from fewgaze.splits import make_folds
from fewgaze.utils import get_logger

log = get_logger("fewgaze.s2b")
TARGETS = ["gaze_h", "gaze_v", "head_pose"]
ALPHAS = np.logspace(0, 6, 7)


def ridge_path(Xtr, Ytr, Xte):
    """Predictions on Xte for every alpha: array (n_alpha, n_te, n_targets). Inputs already standardised."""
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Xtr - xm, full_matrices=False)
    UtY = U.T @ (Ytr - ym)  # (r, t)
    XV = (Xte - xm) @ Vt.T  # (n_te, r)
    out = np.empty((len(ALPHAS), Xte.shape[0], Ytr.shape[1]))
    for i, a in enumerate(ALPHAS):
        out[i] = XV @ ((s / (s**2 + a))[:, None] * UtY) + ym
    return out


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def decode(X, data, folds):
    Y = data[TARGETS].to_numpy(float)
    groups = data["subject"].to_numpy()
    idx_of = {s: i for i, s in enumerate(data["sample_id"])}
    pred = np.zeros_like(Y)
    chosen = []
    for f in folds:
        tr = np.array([idx_of[s] for s in f.train_ids if s in idx_of])
        te = np.array([idx_of[s] for s in f.test_ids if s in idx_of])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xs = (X - mu) / sd
        err = np.zeros((len(ALPHAS), Y.shape[1]))
        for itr, iva in GroupKFold(3).split(tr, groups=groups[tr]):
            P = ridge_path(Xs[tr[itr]], Y[tr[itr]], Xs[tr[iva]])
            err += ((P - Y[tr[iva]][None]) ** 2).sum(axis=1)
        best = err.argmin(axis=0)  # per target
        P = ridge_path(Xs[tr], Y[tr], Xs[te])
        for j in range(Y.shape[1]):
            pred[te, j] = P[best[j], :, j]
        chosen.append([float(ALPHAS[b]) for b in best])
    return [
        {"target": t, "r2_pooled": r2(Y[:, j], pred[:, j]), "alpha_median": float(np.median([c[j] for c in chosen]))}
        for j, t in enumerate(TARGETS)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", default=[m.key for m in MODELS])
    args = ap.parse_args()
    cfg = load_config(args.config)
    man = load_manifest(cfg)
    data = labelled_view(man.records, derive_labels(man.records, cfg["labels"]))
    folds = make_folds(data, int(cfg["splits"]["n_folds"]), int(cfg["splits"]["seed"]))
    out = output_dir(cfg) / "study2" / "decoding_grouped_alpha.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {r["model"] for r in rows}
    if "geometry" not in done:
        keys = cache_keys(cfg, manifest_fingerprint(man.records))
        geo = open_cache(cfg, keys, "geometry").load(man.records["sample_id"].tolist())
        rows += [
            {"model": "geometry", "layer": -1, "rep": "geometry", **r}
            for r in decode(geo.subset(data["sample_id"]).astype(float), data, folds)
        ]
    for spec in [m for m in MODELS if m.key in args.models and m.key not in done]:
        path = s2_dir(cache_dir(cfg)) / f"{spec.key}.npz"
        if not path.exists():
            log.warning("%s: no features", spec.key)
            continue
        z = np.load(path)
        pos = {s: i for i, s in enumerate(z["sample_ids"])}
        sub = data[[z["ok"][pos[s]] for s in data["sample_id"]]].reset_index(drop=True)
        rws = np.array([pos[s] for s in sub["sample_id"]])
        E = z["eyes"]
        for layer in range(E.shape[1]):
            t0 = time.perf_counter()
            res = decode(E[rws, layer, :].astype(np.float64), sub, folds)
            rows += [{"model": spec.key, "layer": layer, "rep": "eyes", **r} for r in res]
            log.info(
                "%s eyes L%02d: R2(H)=%.3f R2(V)=%.3f R2(P)=%.3f (%.0f s)",
                spec.key,
                layer,
                res[0]["r2_pooled"],
                res[1]["r2_pooled"],
                res[2]["r2_pooled"],
                time.perf_counter() - t0,
            )
        pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

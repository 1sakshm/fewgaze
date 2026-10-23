"""Study 2 post-hoc analysis P3 (docs/methodology.md): the eye-contact readout with the layer chosen without test data.

The primary Study 2 tables report every layer; quoting the best one is selection on test folds. Here, inside each
outer fold, the layer and the ridge alphas are chosen by 3-fold person-grouped inner CV on the training people
(criterion: inner MSE of H/sd_H plus V/sd_V), then the gaze readout is scored once on the held-out people.
Eye-patch representation for every model; the nine geometry features go through the identical procedure.

Output: results/study2/nested_layer.csv (one row per model x outer fold) and nested_layer_summary.csv
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
from fewgaze.metrics import auroc
from fewgaze.probing import MODELS
from fewgaze.probing import cache_dir as s2_dir
from fewgaze.splits import make_folds
from fewgaze.utils import get_logger

log = get_logger("fewgaze.p3")
ALPHAS = np.logspace(0, 6, 7)


def ridge_path(Xtr, Ytr, Xte):
    """Predictions on Xte for every alpha: (n_alpha, n_te, n_targets). Inputs already standardised."""
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Xtr - xm, full_matrices=False)
    UtY = U.T @ (Ytr - ym)
    XV = (Xte - xm) @ Vt.T
    return np.stack([XV @ ((s / (s**2 + a))[:, None] * UtY) + ym for a in ALPHAS])


def run(layers, data, folds):
    """layers: list of (n, d) matrices aligned with data. Returns one dict per outer fold."""
    Y = data[["gaze_h", "gaze_v"]].to_numpy(float)
    lab = data["label"].to_numpy()
    groups = data["subject"].to_numpy()
    idx_of = {s: i for i, s in enumerate(data["sample_id"])}
    rows = []
    for k, f in enumerate(folds):
        tr = np.array([idx_of[s] for s in f.train_ids if s in idx_of])
        te = np.array([idx_of[s] for s in f.test_ids if s in idx_of])
        sd = Y[tr].std(0)
        Ys = Y / sd
        inner = list(GroupKFold(3).split(tr, groups=groups[tr]))
        best = (np.inf, None, None)
        for li, X in enumerate(layers):
            mu, xs = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xs = ((X - mu) / xs).astype(np.float32)
            err = np.zeros((len(ALPHAS), 2))
            for itr, iva in inner:
                P = ridge_path(Xs[tr[itr]], Ys[tr[itr]].astype(np.float32), Xs[tr[iva]])
                err += ((P - Ys[tr[iva]][None]) ** 2).sum(axis=1)
            a = err.argmin(axis=0)
            total = float(err[a[0], 0] + err[a[1], 1])
            if total < best[0]:
                best = (total, li, a)
        _, li, a = best
        X = layers[li]
        mu, xs = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = ((X - mu) / xs).astype(np.float32)
        P = ridge_path(Xs[tr], Ys[tr].astype(np.float32), Xs[te])
        h, v = P[a[0], :, 0], P[a[1], :, 1]
        rows.append(
            {
                "fold": k,
                "layer": li,
                "alpha_h": float(ALPHAS[a[0]]),
                "alpha_v": float(ALPHAS[a[1]]),
                "auroc": auroc(lab[te], -(h**2 + v**2)),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", default=[m.key for m in MODELS])
    args = ap.parse_args()
    cfg = load_config(args.config)
    man = load_manifest(cfg)
    data = labelled_view(man.records, derive_labels(man.records, cfg["labels"]))
    folds = make_folds(data, int(cfg["splits"]["n_folds"]), int(cfg["splits"]["seed"]))
    out = output_dir(cfg) / "study2"
    path = out / "nested_layer.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {r["model"] for r in rows}
    if "geometry" not in done:
        keys = cache_keys(cfg, manifest_fingerprint(man.records))
        geo = open_cache(cfg, keys, "geometry").load(man.records["sample_id"].tolist())
        rows += [{"model": "geometry", **r} for r in run([geo.subset(data["sample_id"]).astype(float)], data, folds)]
    for spec in [m for m in MODELS if m.key in args.models and m.key not in done]:
        t0 = time.perf_counter()
        z = np.load(s2_dir(cache_dir(cfg)) / f"{spec.key}.npz")
        pos = {s: i for i, s in enumerate(z["sample_ids"])}
        sub = data[[z["ok"][pos[s]] for s in data["sample_id"]]].reset_index(drop=True)
        E = z["eyes"][np.array([pos[s] for s in sub["sample_id"]])]
        res = run([E[:, layer, :].astype(np.float32) for layer in range(E.shape[1])], sub, folds)
        rows += [{"model": spec.key, **r} for r in res]
        log.info(
            "%s: layers %s, AUROC %.3f (%.0f s)",
            spec.key,
            [r["layer"] for r in res],
            np.mean([r["auroc"] for r in res]),
            time.perf_counter() - t0,
        )
        pd.DataFrame(rows).to_csv(path, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    summ = (
        df.groupby("model")
        .agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"), layers=("layer", lambda s: " ".join(map(str, s))))
        .reset_index()
    )
    geo = df[df.model == "geometry"].set_index("fold")["auroc"]
    summ["delta_vs_geometry"] = [float((df[df.model == m].set_index("fold")["auroc"] - geo).mean()) for m in summ.model]
    summ["delta_sd"] = [float((df[df.model == m].set_index("fold")["auroc"] - geo).std()) for m in summ.model]
    summ.to_csv(out / "nested_layer_summary.csv", index=False)
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

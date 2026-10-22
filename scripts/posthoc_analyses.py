"""POST-HOC exploratory analyses (P1, P2 in docs/methodology.md). Not part of the prespecified study.

P1  Linear decoding diagnostic: how well does each cached representation linearly decode the continuous
    annotations (H, V, head pose) for unseen subjects? Ridge regression, 5-fold GroupKFold by subject;
    alpha chosen by RidgeCV's internal leave-one-image-out, which ignores person grouping and can make held-out R^2 optimistic or pessimistic (see docs/methodology.md, Study 2 correction).
P2  Figure comparing DINOv2 on the face crop vs an eye-region crop, from the separate post-hoc run
    (`run_experiments.py --config configs/posthoc_eyes.yaml --no-noise`).

Outputs: results/posthoc/tables/decoding_diagnostic.csv, results/posthoc/figures/figP*.png|pdf
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fewgaze.config import REPO_ROOT, load_config, output_dir
from fewgaze.dataset import load_manifest, manifest_fingerprint
from fewgaze.extraction import cache_keys, method_registry, open_cache
from fewgaze.features import CacheError
from fewgaze.plots import INK2, apply_style, bottom_legend, learning_curve, save
from fewgaze.utils import get_logger

log = get_logger("fewgaze.posthoc")
CACHES = {
    "geometry": "Geometry (9 MediaPipe features)",
    "dinov2_cls": "DINOv2-S CLS, face crop",
    "dinov2_patchmean": "DINOv2-S mean-patch, face crop",
    "dinov2_eyes": "DINOv2-S CLS, eye-region crop (P2)",
    "l2cs_angles": "L2CS-Net (pitch, yaw)",
    "l2cs_embed": "L2CS-Net 2048-d features",
    "clip_embed": "CLIP ViT-B/32 embedding",
}


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "posthoc_eyes.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    man = load_manifest(cfg)
    rec = man.records.set_index("sample_id")
    keys = cache_keys(cfg, manifest_fingerprint(man.records))
    ids = man.records["sample_id"].tolist()

    rows = []
    for name, label in CACHES.items():
        try:
            t = open_cache(cfg, keys, name).load(ids)
        except CacheError:
            log.warning("P1: cache %s not available; skipped", name)
            continue
        ok = t.valid & np.isfinite(t.X).all(axis=1)
        X, meta = t.X[ok], rec.loc[t.sample_ids[ok]]
        for target in ("gaze_h", "gaze_v", "head_pose"):
            y = meta[target].to_numpy(float)
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
            p = cross_val_predict(model, X, y, groups=meta["subject"].to_numpy(), cv=GroupKFold(5))
            rows.append({"representation": name, "label": label, "target": target, "r2": r2(y, p), "dim": X.shape[1], "n": len(y)})
        log.info("P1: %s done", name)
    diag = pd.DataFrame(rows)
    diag.to_csv(out / "tables" / "decoding_diagnostic.csv", index=False)
    print(diag.pivot(index="label", columns="target", values="r2").round(3).to_string())

    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    piv = diag.pivot(index="label", columns="target", values="r2").reindex([v for v in CACHES.values() if v in set(diag["label"])])
    y = np.arange(len(piv))
    for i, (t, c, name) in enumerate(
        [("gaze_h", "#2a78d6", "horizontal gaze H"), ("gaze_v", "#eb6834", "vertical gaze V"), ("head_pose", "#9aa0a6", "head pose P")]
    ):
        ax.barh(y + (i - 1) * 0.26, piv[t].clip(lower=-0.2), height=0.24, color=c, label=name)
    ax.set_yticks(y, piv.index, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-0.2, 1)
    ax.set_xlabel("held-out-subject R² of a linear (ridge) decoder (values < −0.2 clipped)")
    ax.set_title("P1 (post-hoc): what each frozen representation encodes linearly")
    bottom_legend(fig, ax, ncol=3, y=0.0)
    save(fig, out / "figures" / "figP1_decoding_diagnostic")

    lc_path = out / "tables" / "learning_curves.csv"
    if lc_path.exists():
        curves = pd.read_csv(lc_path, dtype={"budget": str})
        reg = method_registry(cfg)
        ms = [m for m in ("geometry", "dinov2", "dinov2_eyes") if m in set(curves["method"])]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        learning_curve(ax, curves, reg, ms, "auroc")
        ax.set_ylabel("AUROC")
        ax.set_title("P2 (post-hoc): DINOv2 on the face crop vs an eye-region crop")
        bottom_legend(fig, ax, ncol=2, y=0.0)
        fig.text(
            0.01, -0.13, "Same folds, selections and statistics as the primary study; 95% subject-bootstrap CI.", fontsize=7, color=INK2
        )
        save(fig, out / "figures" / "figP2_dinov2_eyes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Additional figures for the extended paper (report/figures/fig_*.pdf and results/figures/*.png).

All panels are computed from cached features and result tables. No dataset image is drawn: the teaser renders
MediaPipe's canonical face mesh (a synthetic model, Apache-2.0), not a participant.
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from fewgaze.config import REPO_ROOT, load_config, output_dir
from fewgaze.dataset import load_manifest, manifest_fingerprint
from fewgaze.extraction import cache_keys, open_cache
from fewgaze.geometry import FEATURE_NAMES
from fewgaze.labels import derive_labels, labelled_view
from fewgaze.plots import FAMILY_COLOR, GRID, INK, INK2, apply_style, bottom_legend, save
from fewgaze.probing import MODELS
from fewgaze.utils import get_logger

log = get_logger("fewgaze.paperfigs")
COBALT, SIGNAL = "#2a44f5", "#ff4a1c"
MODEL_COLORS = {"dinov2_s": "#eb6834", "dinov2_b": "#b5441b", "clip_b16": "#c98500", "siglip_b16": "#6250d6", "mae_b16": "#1baf7a"}


def out(cfg, name):
    for d in (output_dir(cfg) / "figures", REPO_ROOT / "report" / "figures"):
        d.mkdir(parents=True, exist_ok=True)
    return output_dir(cfg) / "figures" / name


def save_both(fig, cfg, name):
    stem = out(cfg, name)
    save(fig, stem)
    import shutil

    shutil.copy2(stem.with_suffix(".pdf"), REPO_ROOT / "report" / "figures" / f"{name}.pdf")


def teaser(cfg):
    mesh = json.loads((REPO_ROOT / "site" / "assets" / "face_mesh.json").read_text())
    V = np.array(mesh["vertices"])
    V = V - V.mean(axis=0)
    F = np.array(mesh["faces"])
    fig = plt.figure(figsize=(12, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.05)
    for k, yaw in enumerate([0, -30]):
        ax = fig.add_subplot(gs[0, k])
        a = np.radians(yaw)
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        P = V @ R.T
        order = np.argsort(P[F].mean(axis=1)[:, 2])
        for f in F[order]:
            tri = P[f]
            ax.fill(tri[:, 0], tri[:, 1], facecolor="#eef0ec", edgecolor="#9aa0a6", linewidth=0.25, alpha=0.95)
        for key, col in (("rightEye", COBALT), ("leftEye", COBALT)):
            ring = P[mesh[key] + [mesh[key][0]]]
            ax.plot(ring[:, 0], ring[:, 1], color=col, lw=1.2)
            c = P[mesh[key]].mean(axis=0)
            w = np.ptp(P[mesh[key]][:, 0])
            ax.add_patch(plt.Circle((c[0], c[1]), w * 0.19, color=COBALT, zorder=5))
            ax.add_patch(plt.Circle((c[0], c[1]), w * 0.08, color=INK, zorder=6))
        if k == 0:
            # annotate the measured quantities on the subject's right eye (image left)
            idx = mesh["rightEye"]
            a_, b_ = P[33], P[133]
            ax.annotate("", xy=b_[:2], xytext=a_[:2], arrowprops={"arrowstyle": "<->", "color": SIGNAL, "lw": 1.2})
            ax.text((a_[0] + b_[0]) / 2, a_[1] - 1.4, "eye width w", ha="center", fontsize=7, color=SIGNAL)
            up, lo = P[159], P[145]
            ax.annotate("", xy=up[:2] + [0, 0.5], xytext=lo[:2] - [0, 0.5], arrowprops={"arrowstyle": "<->", "color": INK2, "lw": 1})
            ax.text(up[0] + 0.3, up[1] + 1.4, "opening / w", fontsize=7, color=INK2)
            c = P[idx].mean(axis=0)
            ax.annotate(
                "iris offset\n(along, across) / w",
                xy=c[:2],
                xytext=(c[0] - 6.5, c[1] + 5),
                fontsize=7,
                color=COBALT,
                arrowprops={"arrowstyle": "->", "color": COBALT, "lw": 1},
            )
        ax.set_title("head pose 0°" if yaw == 0 else "head pose −30° (camera moved)", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    for i, v in enumerate([10, 0, -10]):
        for j, h in enumerate([-15, -10, -5, 0, 5, 10, 15]):
            pos = h == 0 and v == 0
            ax.add_patch(plt.Circle((1.2 + j * 1.25, 7.6 - i * 1.6), 0.32 if pos else 0.2, color=SIGNAL if pos else "#9aa0a6", zorder=3))
    ax.text(0.2, 9.4, "Columbia gaze targets per camera position (7 × 3)", fontsize=9, color=INK)
    ax.text(
        0.2,
        3.4,
        "positive: H = V = 0° (looking into the lens)\nnegative: the other 20 targets, nearest 5° away",
        fontsize=8,
        color=INK2,
        va="top",
    )
    ax.text(
        0.2,
        1.5,
        "9 features: iris along / across and opening for each eye,\nplus head pitch, yaw, roll → degree-2 logistic regression",
        fontsize=8,
        color=INK2,
        va="top",
    )
    fig.suptitle("FewGaze geometry features on MediaPipe's canonical face mesh (a synthetic model, not a participant)", fontsize=10, y=0.99)
    save_both(fig, cfg, "fig_teaser")


def load_tables(cfg):
    man = load_manifest(cfg)
    data = labelled_view(man.records, derive_labels(man.records, cfg["labels"]))
    keys = cache_keys(cfg, manifest_fingerprint(man.records))
    ids = man.records["sample_id"].tolist()
    t = {n: open_cache(cfg, keys, n).load(ids) for n in ("geometry", "dinov2_cls", "l2cs_angles")}
    return data.reset_index(drop=True), t


def iris_violin(cfg, data, t):
    X = pd.DataFrame(t["geometry"].subset(data["sample_id"]), columns=FEATURE_NAMES)
    df = pd.concat([data[["subject", "head_pose", "gaze_v", "gaze_h"]], X], axis=1)
    df["iris_x"] = (df["right_iris_x"] + df["left_iris_x"]) / 2
    df["iris_y"] = (df["right_iris_y"] + df["left_iris_y"]) / 2
    f0 = df[df["head_pose"] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    for ax, col, by, fixed, vals in (
        (axes[0], "iris_x", "gaze_h", ("gaze_v", 0), [-15, -10, -5, 0, 5, 10, 15]),
        (axes[1], "iris_y", "gaze_v", ("gaze_h", 0), [-10, 0, 10]),
    ):
        sub = f0[f0[fixed[0]] == fixed[1]]
        groups = [sub[sub[by] == v][col].to_numpy() for v in vals]
        parts = ax.violinplot(groups, positions=range(len(vals)), showextrema=False, widths=0.8)
        for b, v in zip(parts["bodies"], vals):
            b.set_facecolor(SIGNAL if v == 0 else COBALT)
            b.set_alpha(0.25)
        rng = np.random.default_rng(0)
        for i, (g, v) in enumerate(zip(groups, vals)):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(g)), g, s=7, color=SIGNAL if v == 0 else COBALT, alpha=0.7, lw=0)
            ax.plot([i - 0.3, i + 0.3], [np.median(g)] * 2, color=INK, lw=1.5)
        ax.set_xticks(range(len(vals)), [f"{v:+d}°" if v else "0°" for v in vals])
        ax.set_xlabel(("horizontal" if by == "gaze_h" else "vertical") + " gaze (head pose 0°, other axis 0°)")
        ax.set_ylabel(f"mean iris offset {'along' if col == 'iris_x' else 'across'} eye (eye-widths)")
    d5 = np.median(f0[(f0.gaze_v == 0) & (f0.gaze_h == 5)]["iris_x"]) - np.median(f0[(f0.gaze_v == 0) & (f0.gaze_h == 0)]["iris_x"])
    iqr0 = np.subtract(*np.percentile(f0[(f0.gaze_v == 0) & (f0.gaze_h == 0)]["iris_x"], [75, 25]))
    axes[0].set_title(f"0° → +5° moves the median iris {d5:.3f} eye-widths; people differ by {iqr0:.3f} (IQR) at 0°", fontsize=9)
    axes[1].set_title("vertical gaze also changes eyelid geometry", fontsize=9)
    fig.text(
        0.01,
        -0.04,
        "Each dot is one of the 56 people (one image per cell). MediaPipe measurements from Study 1; no images shown.",
        fontsize=7,
        color=INK2,
    )
    save_both(fig, cfg, "fig_iris_violin")
    return {"shift_0_5": float(d5), "iqr_0": float(iqr0)}


def tsne_maps(cfg, data, t):
    lab = data["label"].to_numpy()
    subj = pd.factorize(data["subject"])[0]
    absH = np.abs(data["gaze_h"].to_numpy())
    reps = {
        "Geometry (9 features)": (t["geometry"].subset(data["sample_id"]) - 0),
        "DINOv2-S CLS (384-d)": t["dinov2_cls"].subset(data["sample_id"]),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    stats = {}
    for col, (name, X) in enumerate(reps.items()):
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        if X.shape[1] > 50:
            X = PCA(50, random_state=0).fit_transform(X)
        Z = TSNE(2, perplexity=40, init="pca", random_state=0).fit_transform(X)
        cmap = plt.get_cmap("tab20")
        axes[0, col].scatter(Z[:, 0], Z[:, 1], c=[cmap(s % 20) for s in subj], s=2, lw=0, alpha=0.8)
        axes[0, col].set_title(f"{name}: coloured by person", fontsize=9)
        order = np.argsort(lab)
        axes[1, col].scatter(
            Z[order, 0],
            Z[order, 1],
            c=np.where(lab[order] == 1, SIGNAL, np.where(absH[order] >= 10, "#c9ccd2", "#7b8bd8")),
            s=np.where(lab[order] == 1, 4, 1.5),
            lw=0,
            alpha=0.85,
        )
        axes[1, col].set_title(f"{name}: by gaze", fontsize=9)
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        # how much does a point's nearest neighbour share its identity?
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X, return_distance=False)[:, 1]
        stats[name] = {"nn_same_person": float(np.mean(subj[nn] == subj)), "nn_same_label": float(np.mean(lab[nn] == lab))}
        axes[0, col].text(
            0.01,
            0.01,
            f"nearest neighbour is the same person: {stats[name]['nn_same_person']:.0%}",
            transform=axes[0, col].transAxes,
            fontsize=8,
            color=INK,
        )
    fig.suptitle("t-SNE of 5,878 images: DINOv2 collapses each person into one tight cluster", fontsize=11, y=0.995)
    fig.text(
        0.5,
        0.005,
        "Bottom row: red = camera-directed (positive), blue = |H| ≤ 5° negatives, grey = |H| ≥ 10° negatives. Colours in the top row identify the 56 people (20 colours, cycled).",
        ha="center",
        fontsize=8,
        color=INK2,
    )
    save_both(fig, cfg, "fig_tsne")
    return stats


def l2cs_bias(cfg, data, t):
    A = t["l2cs_angles"].subset(data["sample_id"])
    df = data[["head_pose", "gaze_v", "gaze_h"]].copy()
    df["pitch"], df["yaw"] = A[:, 0], A[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    cm = plt.get_cmap("coolwarm")
    poses = sorted(df["head_pose"].unique())
    for i, p in enumerate(poses):
        s = df[(df.head_pose == p) & (df.gaze_v == 0)].groupby("gaze_h")["yaw"].agg(["mean", "sem"])
        axes[0].errorbar(
            s.index,
            s["mean"],
            yerr=1.96 * s["sem"],
            color=cm(i / (len(poses) - 1)),
            marker="o",
            ms=4,
            lw=1.5,
            capsize=2,
            label=f"head {p:+d}°",
        )
        s = df[(df.head_pose == p) & (df.gaze_h == 0)].groupby("gaze_v")["pitch"].agg(["mean", "sem"])
        axes[1].errorbar(
            s.index,
            s["mean"],
            yerr=1.96 * s["sem"],
            color=cm(i / (len(poses) - 1)),
            marker="o",
            ms=4,
            lw=1.5,
            capsize=2,
            label=f"head {p:+d}°",
        )
    for ax, xl, yl in (
        (axes[0], "true horizontal gaze H (deg, V = 0)", "mean L2CS yaw output (deg)"),
        (axes[1], "true vertical gaze V (deg, H = 0)", "mean L2CS pitch output (deg)"),
    ):
        ax.axhline(0, color=INK2, lw=0.8, ls=":")
        ax.axvline(0, color=SIGNAL, lw=1, alpha=0.5)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[0].set_title("Yaw tracks gaze but shifts with head pose", fontsize=9)
    axes[1].set_title("Pitch tracks gaze with a constant negative offset", fontsize=9)
    bottom_legend(fig, axes, ncol=5, y=0.0)
    fig.text(
        0.01,
        -0.1,
        "Frozen L2CS-Net (Gaze360 weights) on all 5,878 Columbia images; bars: 95% CI of the mean. At the gaze-locked target (red line) an unbiased estimator would output (0, 0).",
        fontsize=7,
        color=INK2,
    )
    save_both(fig, cfg, "fig_l2cs_bias")


def forest(cfg):
    d = pd.read_csv(output_dir(cfg) / "tables" / "paired_differences.csv", dtype={"budget": str})
    order = ["5", "10", "25", "50", "100", "all"]
    names = {
        ("dinov2", "geometry"): "DINOv2 − Geometry",
        ("l2cs", "geometry"): "L2CS-Net − Geometry",
        ("l2cs", "dinov2"): "L2CS-Net − DINOv2",
    }
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = 0
    ticks, labels = [], []
    for (a, b), nm in names.items():
        q = d[(d.method_a == a) & (d.method_b == b)].set_index("budget").loc[order]
        for bud, r in q.iterrows():
            ax.errorbar(
                r["delta_auroc"],
                y,
                xerr=[[r["delta_auroc"] - r["ci_low"]], [r["ci_high"] - r["delta_auroc"]]],
                fmt="o",
                color=COBALT if r["delta_auroc"] > 0 else SIGNAL,
                ms=4,
                capsize=2,
                lw=1.4,
            )
            ticks.append(y)
            labels.append(f"{nm}  k={'ALL' if bud == 'all' else bud}")
            y -= 1
        y -= 0.8
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.set_xlabel("ΔAUROC on identical subject-bootstrap resamples (95% CI)")
    ax.set_title("Prespecified paired comparisons at every budget", fontsize=10)
    ax.grid(axis="y", visible=False)
    save_both(fig, cfg, "fig_forest")


def per_subject_scatter(cfg):
    ps = pd.read_csv(output_dir(cfg) / "tables" / "per_subject_auroc.csv", dtype={"subject": str}).pivot(
        index="subject", columns="method", values="auroc"
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(ps["geometry"], ps["l2cs_zeroshot"], s=26, color=FAMILY_COLOR["l2cs"], alpha=0.85, edgecolor="white", lw=0.6)
    ax.plot([0.25, 1], [0.25, 1], color=INK2, lw=0.8, ls=":")
    ax.axhline(0.5, color=GRID, lw=1)
    ax.axvline(0.5, color=GRID, lw=1)
    ax.set_xlabel("geometry, k = 25 (per-person AUROC)")
    ax.set_ylabel("L2CS-Net, zero labels (per-person AUROC)")
    sd_g, sd_l = ps["geometry"].std(), ps["l2cs_zeroshot"].std()
    ax.set_title(f"56 people: L2CS spread SD {sd_l:.2f} vs geometry {sd_g:.2f}", fontsize=9)
    ax.set_xlim(0.25, 1)
    ax.set_ylim(0.25, 1)
    save_both(fig, cfg, "fig_per_subject")


def study2_figs(cfg):
    s2 = output_dir(cfg) / "study2"
    if not (s2 / "decoding.csv").exists():
        log.warning("Study 2 results missing; skipping its figures")
        return None
    dec, rd = pd.read_csv(s2 / "decoding.csv"), pd.read_csv(s2 / "readout.csv")
    labels = {m.key: m.label for m in MODELS}
    models = [m.key for m in MODELS if m.key in set(dec["model"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, tgt, title in zip(axes, ["gaze_h", "gaze_v", "head_pose"], ["horizontal gaze H", "vertical gaze V", "head pose P"]):
        for mk in models:
            for rep, ls in (("global", "--"), ("eyes", "-")):
                q = dec[(dec.model == mk) & (dec.rep == rep) & (dec.target == tgt)].sort_values("layer")
                ax.plot(
                    q["layer"],
                    q["r2_pooled"].clip(lower=-0.2),
                    ls=ls,
                    color=MODEL_COLORS[mk],
                    lw=1.8 if rep == "eyes" else 1.2,
                    marker="o" if rep == "eyes" else None,
                    ms=3,
                )
        g = dec[(dec.model == "geometry") & (dec.target == tgt)]
        if len(g):
            ax.axhline(g["r2_pooled"].iloc[0], color=INK, lw=1.2, ls=":")
            ax.text(12, g["r2_pooled"].iloc[0] + 0.02, "9 geometry features", ha="right", fontsize=7, color=INK)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"decoding {title}", fontsize=10)
        ax.set_xlabel("layer (0 = patch embedding)")
        ax.set_xticks(range(0, 13, 2))
    axes[0].set_ylabel("held-out-subject R² (clipped at −0.2)")
    handles = [Line2D([], [], color=MODEL_COLORS[m], lw=2, label=labels[m]) for m in models]
    handles += [
        Line2D([], [], color=INK2, lw=1.2, ls="--", label="summary token"),
        Line2D([], [], color=INK2, lw=1.8, marker="o", ms=3, label="eye patches"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=7, fontsize=8, frameon=False)
    save_both(fig, cfg, "fig_s2_layers")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, rep in zip(axes, ["global", "eyes"]):
        M = np.array(
            [
                [
                    dec[(dec.model == mk) & (dec.rep == rep) & (dec.target == "gaze_h") & (dec.layer == L)]["r2_pooled"].iloc[0]
                    for L in range(13)
                ]
                for mk in models
            ]
        )
        im = ax.imshow(M.clip(0, 1), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.55 else INK)
        ax.set_xticks(range(13))
        ax.set_yticks(range(len(models)), [labels[m] for m in models])
        ax.set_xlabel("layer")
        ax.set_title(f"horizontal-gaze R², {'summary token' if rep == 'global' else 'eye patches'}", fontsize=10)
        ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    save_both(fig, cfg, "fig_s2_heat")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, ro, title in zip(
        axes,
        ["linear_probe", "gaze_readout"],
        ["linear logistic probe on the layer", "gaze readout: −(Ĥ/σ)² − (V̂/σ)² from the ridge decoder"],
    ):
        for mk in models:
            for rep, ls in (("global", "--"), ("eyes", "-")):
                q = rd[(rd.model == mk) & (rd.rep == rep) & (rd.readout == ro)].sort_values("layer")
                ax.plot(
                    q["layer"],
                    q["auroc_mean"],
                    ls=ls,
                    color=MODEL_COLORS[mk],
                    lw=1.8 if rep == "eyes" else 1.2,
                    marker="o" if rep == "eyes" else None,
                    ms=3,
                )
        g = rd[(rd.model == "geometry") & (rd.readout == ro)]
        if len(g):
            ax.axhline(g["auroc_mean"].iloc[0], color=INK, ls=":", lw=1.2)
        ax.axhline(0.5, color=INK2, lw=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("eye-contact AUROC (held-out subjects)")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=7, fontsize=8, frameon=False)
    save_both(fig, cfg, "fig_s2_readout")
    return True


def study2_table(cfg):
    """LaTeX summary of Study 2. 'max' columns pick the best layer using the test folds: descriptive, optimistic."""
    s2 = output_dir(cfg) / "study2"
    if not (s2 / "decoding.csv").exists():
        return None
    dec, rd = pd.read_csv(s2 / "decoding.csv"), pd.read_csv(s2 / "readout.csv")
    grp = pd.read_csv(s2 / "decoding_grouped_alpha.csv") if (s2 / "decoding_grouped_alpha.csv").exists() else None
    labels = {m.key: m.label for m in MODELS}

    def best(df, **q):
        x = df
        for k, v in q.items():
            x = x[x[k] == v]
        return x.sort_values(x.columns[-1] if "auroc_mean" not in x else "auroc_mean").iloc[-1] if len(x) else None

    rows = []
    summary = {}
    for mk in [m.key for m in MODELS if m.key in set(dec["model"])]:
        d = dec[dec.model == mk]
        last = int(d["layer"].max())
        cell = {}
        for rep in ("global", "eyes"):
            for t in ("gaze_h", "gaze_v", "head_pose"):
                q = d[(d.rep == rep) & (d.target == t)]
                b = q.loc[q["r2_pooled"].idxmax()]
                cell[(rep, t)] = (float(q[q.layer == last]["r2_pooled"].iloc[0]), float(b["r2_pooled"]), int(b["layer"]))
            r = rd[(rd.model == mk) & (rd.rep == rep) & (rd.readout == "gaze_readout")]
            b = r.loc[r["auroc_mean"].idxmax()]
            cell[(rep, "readout")] = (float(r[r.layer == last]["auroc_mean"].iloc[0]), float(b["auroc_mean"]), int(b["layer"]))
        if grp is not None:
            g = grp[(grp.model == mk) & (grp.target == "gaze_h")]
            if len(g):
                b = g.loc[g["r2_pooled"].idxmax()]
                cell[("grouped", "gaze_h")] = (float(g[g.layer == last]["r2_pooled"].iloc[0]), float(b["r2_pooled"]), int(b["layer"]))
        summary[mk] = {f"{a}|{b}": v for (a, b), v in cell.items()}
        f = lambda v: f"{v[0]:.2f}" if v[0] > -0.995 else "$<$-1"  # noqa: E731
        mx = lambda v: f"{v[1]:.2f} ({v[2]})"  # noqa: E731
        rows.append(
            f"{labels[mk]} & {f(cell[('global', 'gaze_h')])} & {mx(cell[('global', 'gaze_h')])} & {f(cell[('eyes', 'gaze_h')])} & {mx(cell[('eyes', 'gaze_h')])} & "
            f"{mx(cell[('eyes', 'gaze_v')])} & {f(cell[('global', 'head_pose')])} & {mx(cell[('eyes', 'readout')])}"
            + (f" & {mx(cell[('grouped', 'gaze_h')])}" if ("grouped", "gaze_h") in cell else " & --")
            + r" \\"
        )
    g = dec[dec.model == "geometry"].set_index("target")["r2_pooled"]
    gr = rd[(rd.model == "geometry") & (rd.readout == "gaze_readout")]["auroc_mean"]
    gg = grp[(grp.model == "geometry") & (grp.target == "gaze_h")]["r2_pooled"] if grp is not None else pd.Series(dtype=float)
    rows.append(r"\midrule")
    rows.append(
        "9 geometry features & "
        + r"\multicolumn{4}{c}{"
        + f"{g['gaze_h']:.2f}"
        + "} & "
        + f"{g['gaze_v']:.2f} & {g['head_pose']:.2f} & {gr.iloc[0]:.3f} & "
        + (f"{gg.iloc[0]:.2f}" if len(gg) else "--")
        + r" \\"
    )
    head = [
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{$R^2(H)$, summary token} & \multicolumn{2}{c}{$R^2(H)$, eye patches} & $R^2(V)$, eyes & $R^2(P)$, token & eye-contact AUROC & $R^2(H)$, eyes \\",
        r"Model & last layer & max (layer) & last layer & max (layer) & max (layer) & last layer & eyes, max (layer) & grouped $\alpha$, max \\",
        r"\midrule",
    ]
    tex = "\n".join(head + rows + [r"\bottomrule", r"\end{tabular}"])
    (REPO_ROOT / "report" / "generated" / "table_study2.tex").write_text(tex, encoding="utf-8")
    (output_dir(cfg) / "study2" / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_style()
    run = lambda n: args.only is None or n in args.only  # noqa: E731
    stats = {}
    if run("teaser"):
        teaser(cfg)
    if any(run(n) for n in ("violin", "tsne", "l2cs")):
        data, t = load_tables(cfg)
        if run("violin"):
            stats["iris"] = iris_violin(cfg, data, t)
        if run("tsne"):
            stats["tsne"] = tsne_maps(cfg, data, t)
        if run("l2cs"):
            l2cs_bias(cfg, data, t)
    if run("forest"):
        forest(cfg)
    if run("subject"):
        per_subject_scatter(cfg)
    if run("study2"):
        study2_figs(cfg)
        study2_table(cfg)
    if stats:
        p = output_dir(cfg) / "tables" / "paper_figure_stats.json"
        prev = json.loads(p.read_text()) if p.exists() else {}
        prev.update(stats)
        p.write_text(json.dumps(prev, indent=2))
        print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

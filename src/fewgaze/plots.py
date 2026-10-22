"""Shared figure style and plotting functions. Every plotted value is read from result tables;
nothing here contains a metric value."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (dataviz reference palette, light mode), fixed slot order.
# Colour follows the method FAMILY; zero-shot variants reuse the family hue with a dashed line.
FAMILY_COLOR = {"geometry": "#2a78d6", "dinov2": "#eb6834", "l2cs": "#1baf7a", "clip": "#eda100"}
ABLATION_COLOR = {
    "geometry_linear": "#e87ba4",
    "dinov2_patchmean": "#008300",
    "l2cs_embed": "#6250d6",
    "clip_probe": "#e34948",
    "dinov2_eyes": "#6250d6",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ALL_X = 400.0  # x-position used for the 'all' budget on the log axis


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "lines.linewidth": 2,
            "lines.markersize": 5,
        }
    )


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def bottom_legend(fig: plt.Figure, axes, ncol: int = 2, y: float = -0.02) -> None:
    """One de-duplicated legend for the whole figure, placed below the axes so it never covers data."""
    handles, labels = {}, []
    for ax in np.atleast_1d(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab] = h
                labels.append(lab)
    if handles:
        fig.legend(
            [handles[lab] for lab in labels], labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=8, frameon=False
        )


def color_for(method: str, family: str) -> str:
    return ABLATION_COLOR.get(method, FAMILY_COLOR.get(family, INK2))


def budget_x(b: str) -> float:
    return ALL_X if b == "all" else float(b)


def learning_curve(ax: plt.Axes, curves: pd.DataFrame, specs: dict, methods: list[str], metric: str, chance: float = 0.5) -> None:
    sub = curves[(curves["metric"] == metric) & (curves["condition"] == "clean")]
    numeric = sorted({float(b) for b in sub["budget"] if b not in ("0", "all")})
    for m in methods:
        q = sub[sub["method"] == m]
        if q.empty:
            continue
        spec = specs[m]
        c = color_for(m, spec.family)
        if spec.kind == "zeroshot":
            r = q.iloc[0]
            ax.axhline(r["estimate"], color=c, ls="--", lw=1.6, label=f"{spec.label}")
            ax.axhspan(r["ci_low"], r["ci_high"], color=c, alpha=0.10, lw=0)
            continue
        q = q.assign(x=q["budget"].map(budget_x)).sort_values("x")
        num, allp = q[q["budget"] != "all"], q[q["budget"] == "all"]
        ax.fill_between(num["x"], num["ci_low"], num["ci_high"], color=c, alpha=0.15, lw=0)
        ax.plot(num["x"], num["estimate"], "-o", color=c, label=spec.label, ms=5, mec="white", mew=1)
        if len(allp):
            r = allp.iloc[0]
            ax.errorbar(
                [ALL_X],
                [r["estimate"]],
                yerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                fmt="s",
                color=c,
                ms=6,
                mec="white",
                mew=1,
                capsize=3,
                lw=1.5,
            )
    ax.axhline(chance, color=INK2, lw=0.8, ls=":")
    ax.set_xscale("log")
    ticks = numeric + [ALL_X]
    ax.set_xticks(ticks, [f"{int(t)}" for t in numeric] + ["ALL"])
    ax.minorticks_off()
    if numeric:
        ax.axvspan(numeric[-1] * 1.35, ALL_X / 1.35, color=GRID, alpha=0.5, lw=0)  # visual gap: no curve drawn between
    ax.set_xlabel("labelled training examples per class (k)")


def confusion_grid(axes, mats: list[tuple[str, dict]]) -> None:
    for ax, (title, c) in zip(axes, mats):
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)
        norm = m / m.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(m):
            ax.text(
                j, i, f"{norm[i, j]:.2f}\n(n={int(v)})", ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.6 else INK
            )
        ax.set_xticks([0, 1], ["pred. away", "pred. camera"])
        ax.set_yticks([0, 1], ["true away", "true camera"])
        ax.grid(False)
        ax.set_title(title, fontsize=9)

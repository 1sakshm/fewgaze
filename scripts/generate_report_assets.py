"""Generate every public figure, the LaTeX result tables and the website data file from the
result tables on disk. No metric value is hard-coded here.

Outputs: results/figures/fig1..fig8 (+ extras) as PNG and PDF, copies in report/figures/,
         report/generated/*.tex, site/data/results.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fewgaze.config import REPO_ROOT, load_config, output_dir
from fewgaze.extraction import method_registry
from fewgaze.plots import FAMILY_COLOR, INK, INK2, apply_style, bottom_legend, color_for, confusion_grid, learning_curve, save
from fewgaze.utils import get_logger, read_json, write_json

log = get_logger("fewgaze.assets")


def fig_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 22)
    ax.axis("off")
    steps = [
        ("Columbia Gaze\n5,880 images\n56 subjects", "data"),
        ("Annotation\naudit\nV=H=0 → positive", "data"),
        ("5 subject-\ndisjoint\nfolds", "protocol"),
        ("Frozen features\ngeometry\nDINOv2 · L2CS\nCLIP", "model"),
        ("Budget sampling\nk per class\n× 5 seeds\n(shared)", "protocol"),
        ("Logistic\nregression\n(train-only\nscaling)", "model"),
        ("Held-out\nsubjects\nAUROC\nbal. acc.", "protocol"),
        ("Learning curves\nsubject\nbootstrap\nerror analysis", "data"),
    ]
    col = {"data": "#f3f2ee", "protocol": "#e8f0fb", "model": "#fdeee6"}
    w, gap = 10.6, 1.9
    for i, (txt, kind) in enumerate(steps):
        x = 0.5 + i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 3), w, 16, fc=col[kind], ec=INK2, lw=0.8))
        ax.text(x + w / 2, 11, txt, ha="center", va="center", fontsize=8, color=INK, linespacing=1.35)
        if i < len(steps) - 1:
            ax.annotate(
                "", xy=(x + w + gap - 0.2, 11), xytext=(x + w + 0.2, 11), arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 0.9}
            )
    ax.text(0.5, 21, "FewGaze protocol: the same folds, selections and test images for every method", fontsize=9, color=INK)
    save(fig, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    tdir, fdir = out / "tables", out / "figures"
    reg = method_registry(cfg)
    apply_style()
    fig_pipeline(fdir / "fig1_pipeline")

    curves_path = tdir / "learning_curves.csv"
    if not curves_path.exists():
        log.error("No results yet (%s missing). Run scripts/run_experiments.py first.", curves_path)
        return 3
    curves = pd.read_csv(curves_path, dtype={"budget": str})
    run_id = (out / "runs" / "LATEST").read_text().strip()
    prov = read_json(out / "runs" / run_id / "provenance.json")
    methods = [m for m in reg if m in set(curves["method"])]
    primary = [m for m in methods if reg[m].primary]
    ablation = [m for m in methods if not reg[m].primary]
    n_subj = int(curves["n_test_subjects"].max())
    n_img = int(curves["n_test_images"].max())
    n_folds, n_seeds = int(cfg["splits"]["n_folds"]), len(cfg["sampling"]["seeds"])
    cap = f"{n_subj} held-out subjects, {n_img} test images; {n_folds} subject folds × {n_seeds} selection seeds; band = 95% subject-bootstrap CI"

    for metric, stem, ylab in [
        ("auroc", "fig3_learning_curves_auroc", "AUROC"),
        ("balanced_accuracy", "fig4_learning_curves_balacc", "balanced accuracy"),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        learning_curve(ax, curves, reg, primary, metric)
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab} vs. labelled examples per class")
        bottom_legend(fig, ax, ncol=2, y=0.0)
        cap_y = -0.2
        fig.text(0.01, cap_y, cap, fontsize=7, color=INK2)
        save(fig, fdir / stem)
    if ablation:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
        for ax, fam_methods, title in [
            (
                axes[0],
                [m for m in methods if reg[m].family in ("geometry", "l2cs") and reg[m].kind == "trained"],
                "Geometry and L2CS-Net representations",
            ),
            (
                axes[1],
                [m for m in methods if reg[m].family in ("dinov2", "clip") and reg[m].kind == "trained"],
                "Generic visual representations",
            ),
        ]:
            learning_curve(ax, curves, reg, fam_methods, "auroc")
            ax.set_title(title)
        axes[0].set_ylabel("AUROC")
        bottom_legend(fig, axes, ncol=3, y=0.0)
        fig.text(0.01, -0.12, cap, fontsize=7, color=INK2)
        save(fig, fdir / "figS1_ablations_auroc")

    # Fig 5: label noise
    noisy = curves[(curves["metric"] == "auroc") & (curves["condition"] == "noise")]
    if len(noisy):
        nb = sorted(noisy["budget"].unique(), key=float)
        fig, axes = plt.subplots(1, len(nb), figsize=(3.0 * len(nb), 3.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, b in zip(axes, nb):
            for m in [m for m in primary if reg[m].kind == "trained"] + [m for m in ablation if m == "dinov2_patchmean"]:
                clean = curves[
                    (curves["metric"] == "auroc") & (curves["condition"] == "clean") & (curves["method"] == m) & (curves["budget"] == b)
                ]
                q = noisy[(noisy["method"] == m) & (noisy["budget"] == b)].sort_values("noise_rate")
                q = pd.concat([clean, q]).sort_values("noise_rate")
                if q.empty:
                    continue
                c = color_for(m, reg[m].family)
                ax.fill_between(q["noise_rate"], q["ci_low"], q["ci_high"], color=c, alpha=0.12, lw=0)
                ax.plot(q["noise_rate"], q["estimate"], "-o", color=c, label=reg[m].label, mec="white", mew=1)
            ax.set_title(f"k = {b} per class")
            ax.set_xlabel("training-label flip rate")
            ax.set_xticks([0, 0.1, 0.2, 0.3])
            ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        axes[0].set_ylabel("AUROC (clean test labels)")
        bottom_legend(fig, axes, ncol=4, y=0.0)
        fig.suptitle("Simulated annotation noise: class-balanced symmetric flips of training labels only", y=1.02, fontsize=10)
        fig.text(0.01, -0.1, cap, fontsize=7, color=INK2)
        save(fig, fdir / "fig5_label_noise")

    # Fig 6: head pose
    hp_path = tdir / "head_pose_auroc.csv"
    if hp_path.exists():
        hp = pd.read_csv(hp_path, dtype={"budget": str})
        bud = [b for b in [str(cfg["analysis"]["focus_budget"]), "all"] if b in set(hp["budget"])]
        fig, axes = plt.subplots(1, len(bud), figsize=(5.4 * len(bud), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        poses = sorted(hp["head_pose"].unique())
        for ax, b in zip(axes, bud):
            q = hp[hp["budget"] == b]
            ms = [m for m in primary if m in set(q["method"])]
            width = 0.8 / max(len(ms), 1)
            for i, m in enumerate(ms):
                r = q[q["method"] == m].set_index("head_pose").loc[poses]
                x = np.arange(len(poses)) + (i - (len(ms) - 1) / 2) * width
                c = color_for(m, reg[m].family)
                ax.errorbar(
                    x,
                    r["auroc"],
                    yerr=[r["auroc"] - r["ci_low"], r["ci_high"] - r["auroc"]],
                    fmt="o",
                    color=c,
                    ms=5,
                    capsize=2,
                    lw=1.2,
                    label=reg[m].label,
                    mfc="white" if reg[m].kind == "zeroshot" else c,
                )
            ax.set_xticks(range(len(poses)), [f"{p:+d}°" for p in poses])
            ax.set_xlabel("head pose relative to camera")
            ax.set_title("zero-shot methods and k = " + ("ALL" if b == "all" else b) + " per class")
            ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        axes[0].set_ylabel("AUROC within head pose")
        bottom_legend(fig, axes, ncol=3, y=0.0)
        fig.text(
            0.01,
            -0.16,
            f"Each pose: {int(hp['n_images'].min())}-{int(hp['n_images'].max())} test images incl. {int(hp['n_pos'].min())} positives (one per subject). AUROC within each test fold restricted to the pose, averaged over folds and seeds; bars = 95% subject-bootstrap CI (500 replicates).",
            fontsize=7,
            color=INK2,
        )
        save(fig, fdir / "fig6_head_pose")

    # Fig 7: confusion matrices
    cm_path = tdir / "confusion_matrices.json"
    if cm_path.exists():
        cms = read_json(cm_path)
        items = []
        for m, v in cms.items():
            if m in reg:
                cond = "(no training labels)" if v["budget"] == "0" else f"(k={v['budget']} per class, seed {v['seed']})"
                items.append((f"{reg[m].label}\n{cond}", v))
        fig, axes = plt.subplots(1, len(items), figsize=(3.3 * len(items), 3.2))
        confusion_grid(np.atleast_1d(axes), items)
        for ax in np.atleast_1d(axes)[1:]:
            ax.set_yticklabels([])
        fig.subplots_adjust(wspace=0.08)
        fig.suptitle("Confusion matrices on held-out subjects (row-normalised; threshold 0.5)", y=1.04, fontsize=10)
        save(fig, fdir / "fig7_confusion")

    # Error rates by gaze condition
    er_path = tdir / "error_rates_by_condition.csv"
    if er_path.exists():
        er = pd.read_csv(er_path)
        ms = [m for m in primary if m in set(er["method"])]
        fig, axes = plt.subplots(1, len(ms), figsize=(3.3 * len(ms), 2.6))
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, ms):
            q = (
                er[er["method"] == m]
                .groupby(["gaze_v", "gaze_h"], as_index=False)
                .apply(lambda g: pd.Series({"rate": np.average(g["predicted_positive_rate"], weights=g["n"])}), include_groups=False)
            )
            grid = q.pivot(index="gaze_v", columns="gaze_h", values="rate").sort_index(ascending=False)
            ax.imshow(grid.to_numpy(), cmap="Blues", vmin=0, vmax=1)
            for (i, j), v in np.ndenumerate(grid.to_numpy()):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else INK)
            ax.set_xticks(range(grid.shape[1]), grid.columns, fontsize=7)
            ax.set_yticks(range(grid.shape[0]), grid.index, fontsize=7)
            ax.set_xlabel("H (deg)")
            ax.grid(False)
            ax.set_title(reg[m].label, fontsize=8)
        axes[0].set_ylabel("V (deg)")
        fig.suptitle(
            f"Rate of 'camera-directed' predictions per gaze cell (k={cfg['analysis']['focus_budget']}; centre = true positives)",
            y=1.05,
            fontsize=9,
        )
        save(fig, fdir / "figS2_prediction_rate_by_gaze")

    # Per-subject spread
    ps_path = tdir / "per_subject_auroc.csv"
    if ps_path.exists():
        ps = pd.read_csv(ps_path, dtype={"subject": str})
        ms = [m for m in primary if m in set(ps["method"])]
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        rng = np.random.default_rng(0)
        for i, m in enumerate(ms):
            v = ps[ps["method"] == m]["auroc"].dropna().to_numpy()
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color=color_for(m, reg[m].family), alpha=0.7, lw=0)
            ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, lw=1.5)
        ax.set_xticks(range(len(ms)), [reg[m].label.replace(" (", "\n(") for m in ms], fontsize=6.5)
        ax.set_ylabel("per-subject AUROC")
        ax.set_title(f"Variation across subjects (k={cfg['analysis']['focus_budget']}, mean over seeds; line = median)")
        save(fig, fdir / "figS3_per_subject")

    # Fig 8: efficiency
    ext = read_json(tdir / "extraction_summary.json") if (tdir / "extraction_summary.json").exists() else {}
    timing = (
        pd.read_csv(tdir / "classifier_timing.csv", dtype={"budget": str}) if (tdir / "classifier_timing.csv").exists() else pd.DataFrame()
    )
    eff_rows = []
    cache_of = {m: reg[m].cache for m in methods}
    for m in methods:
        e = ext.get(cache_of[m], {})
        t = timing[(timing["method"] == m) & (timing["condition"] == "clean")] if len(timing) else pd.DataFrame()
        fit_all = t[t["budget"] == "all"]["fit_seconds_mean"].mean() if len(t) else np.nan
        fit_25 = t[t["budget"] == "25"]["fit_seconds_mean"].mean() if len(t) else np.nan
        eff_rows.append(
            {
                "method": m,
                "label": reg[m].label,
                "family": reg[m].family,
                "extract_s_per_img": (e.get("timing") or {}).get("seconds_per_image", np.nan),
                "extract_device": (e.get("model") or {}).get("device", "cpu" if reg[m].family == "geometry" else ""),
                "params": (e.get("model") or {}).get("params"),
                "fit_s_k25": fit_25,
                "fit_s_all": fit_all,
                "dim": e.get("dim"),
            }
        )
    eff = pd.DataFrame(eff_rows)
    eff.to_csv(tdir / "efficiency.csv", index=False)
    if len(eff):
        geo_t = eff.loc[eff["family"] == "geometry", "extract_s_per_img"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        e1 = eff.drop_duplicates("family").copy()
        # appearance models also depend on the geometry stage (face crop), reported separately here
        axes[0].barh(e1["family"], e1["extract_s_per_img"] * 1000, color=[FAMILY_COLOR[f] for f in e1["family"]], height=0.55)
        for y, (v, d) in enumerate(zip(e1["extract_s_per_img"] * 1000, e1["extract_device"])):
            axes[0].text(v, y, f"  {v:.0f} ms ({d})", va="center", fontsize=7, color=INK)
        axes[0].set_xscale("log")
        axes[0].set_xlim(right=float(e1["extract_s_per_img"].max()) * 1000 * 3.5)
        axes[0].set_xlabel("feature extraction time per image (ms, log scale)")
        axes[0].set_title("Feature extraction per image")
        e2 = eff[eff["fit_s_all"].notna()]
        axes[1].barh(e2["label"], e2["fit_s_all"], color=[color_for(m, f) for m, f in zip(e2["method"], e2["family"])], height=0.55)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("logistic-regression fit time at k = ALL (s, log scale)")
        axes[1].tick_params(axis="y", labelsize=7)
        axes[1].set_title("Downstream training cost")
        hw = prov.get("runtime", {})
        fig.text(
            0.01,
            -0.05,
            f"Hardware: {hw.get('gpu', 'CPU only')} / {hw.get('processor', '')}; geometry stage {float(geo_t.iloc[0]) * 1000 if len(geo_t) else float('nan'):.0f} ms/img incl. JPEG decode. Appearance models (DINOv2, L2CS-Net, CLIP) read the shared face crop, so their end-to-end cost also includes that stage.",
            fontsize=7,
            color=INK2,
        )
        save(fig, fdir / "fig8_efficiency")

    copy_to_report(fdir)
    write_latex(curves, reg, primary, ablation, tdir, cfg)
    write_site_data(cfg, curves, reg, methods, tdir, out, prov)
    log.info("assets written")
    return 0


def copy_to_report(fdir: Path) -> None:
    dest = REPO_ROOT / "report" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    for f in fdir.glob("fig*.pdf"):
        shutil.copy2(f, dest / f.name)
    site = REPO_ROOT / "site" / "assets"
    site.mkdir(parents=True, exist_ok=True)
    for f in fdir.glob("fig*.png"):
        shutil.copy2(f, site / f.name)


def fmt(r: pd.Series) -> str:
    return f"{r['estimate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def write_latex(curves, reg, primary, ablation, tdir, cfg) -> None:
    gdir = REPO_ROOT / "report" / "generated"
    gdir.mkdir(parents=True, exist_ok=True)
    budgets = [str(b) for b in cfg["sampling"]["budgets"]]
    au = curves[(curves["metric"] == "auroc") & (curves["condition"] == "clean")]
    lines = [
        r"\begin{tabular}{l" + "c" * len(budgets) + "}",
        r"\toprule",
        "Method & " + " & ".join(("ALL" if b == "all" else f"$k={b}$") for b in budgets) + r" \\",
        r"\midrule",
    ]
    for m in [m for m in primary + ablation if reg[m].kind == "trained"]:
        cells = []
        for b in budgets:
            r = au[(au["method"] == m) & (au["budget"] == b)]
            cells.append(f"{r.iloc[0]['estimate']:.3f}" if len(r) else "--")
        lines.append(reg[m].label.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
        if m == [x for x in primary if reg[x].kind == "trained"][-1]:
            lines.append(r"\midrule")
    zs = [m for m in primary if reg[m].kind == "zeroshot"]
    if zs:
        lines.append(r"\midrule")
        for m in zs:
            r = au[(au["method"] == m)]
            v = f"{r.iloc[0]['estimate']:.3f}" if len(r) else "--"
            lines.append(reg[m].label + r" & \multicolumn{" + str(len(budgets)) + "}{c}{" + v + r" (no training labels)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gdir / "table_auroc.tex").write_text("\n".join(lines), encoding="utf-8")
    pd_path = tdir / "paired_differences.csv"
    if pd_path.exists():
        d = pd.read_csv(pd_path, dtype={"budget": str})
        rows = [r"\begin{tabular}{llccc}", r"\toprule", r"Comparison & $k$ & $\Delta$AUROC & 95\% CI & CI excludes 0 \\", r"\midrule"]
        for _, r in d[d["budget"].isin(["5", "25", "all"])].iterrows():  # full table: paired_differences.csv
            rows.append(
                f"{reg[r['method_a']].family} $-$ {reg[r['method_b']].family} & {'ALL' if r['budget'] == 'all' else r['budget']} & "
                f"{r['delta_auroc']:+.3f} & [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] & {'yes' if r['excludes_zero'] else 'no'} \\\\"
            )
        rows += [r"\bottomrule", r"\end{tabular}"]
        (gdir / "table_paired.tex").write_text("\n".join(rows), encoding="utf-8")


def write_site_data(cfg, curves, reg, methods, tdir, out, prov) -> None:
    def tab(name):
        p = tdir / name
        return pd.read_csv(p, dtype={"budget": str, "subject": str}).replace({np.nan: None}).to_dict(orient="records") if p.exists() else []

    audit = read_json(out / "audit" / "audit.json") if (out / "audit" / "audit.json").exists() else {}
    data = {
        "status": "verified_experimental_results",
        "run_id": prov.get("run_id"),
        "created": prov.get("created"),
        "git_commit": prov.get("git_commit"),
        "protocol": {
            "n_folds": cfg["splits"]["n_folds"],
            "seeds": cfg["sampling"]["seeds"],
            "budgets": [str(b) for b in cfg["sampling"]["budgets"]],
            "noise_rates": cfg["noise"]["rates"],
            "n_bootstrap": cfg["statistics"]["n_bootstrap"],
            "threshold": cfg["classifier"]["threshold"],
        },
        "dataset": {
            k: audit.get(k)
            for k in (
                "n_images",
                "n_subjects",
                "head_poses",
                "gaze_v",
                "gaze_h",
                "label_status_counts",
                "n_missing_grid_cells",
                "n_unreadable",
            )
        },
        "methods": {m: {"label": reg[m].label, "family": reg[m].family, "kind": reg[m].kind, "primary": reg[m].primary} for m in methods},
        "curves": curves.drop(columns=["budget_order"], errors="ignore").replace({np.nan: None}).to_dict(orient="records"),
        "paired": tab("paired_differences.csv"),
        "head_pose": tab("head_pose_auroc.csv"),
        "efficiency": tab("efficiency.csv"),
        "error_categories": tab("error_category_counts.csv"),
        "confusion": read_json(tdir / "confusion_matrices.json") if (tdir / "confusion_matrices.json").exists() else {},
        "n_common_eligible": prov.get("n_common_eligible"),
        "folds": [{"fold": f["fold"], "test_subjects": f["test_subjects"]} for f in prov.get("folds", [])],
        "error_rates": tab("error_rates_by_condition.csv"),
    }
    p = REPO_ROOT / "site" / "data" / "results.json"
    write_json(data, p)
    log.info("site data: %s (%.0f kB)", p, p.stat().st_size / 1024)
    json.loads(p.read_text())  # sanity: valid JSON


if __name__ == "__main__":
    sys.exit(main())

"""Audit the locally supplied dataset before any modelling.

Writes  results/audit/audit.json        (machine-readable)
        results/audit/audit_summary.md  (human-readable)
        results/audit/labels.csv        (derived labels, separate from annotations)
        results/audit/manifest.csv      (original annotations + file metadata)
        results/figures/fig2_dataset_composition.png

Exit codes: 0 ok, 2 dataset missing/unparseable, 3 audit found blocking problems.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fewgaze.config import load_config, output_dir, resolve_path
from fewgaze.dataset import DatasetError, load_manifest, manifest_fingerprint
from fewgaze.labels import derive_labels
from fewgaze.utils import file_sha256, get_logger, runtime_info, utc_now, write_json

log = get_logger("fewgaze.audit")

LICENSE_NOTE = (
    "Columbia Gaze Data Set: obtained via the CAVE download form (email required). The project page states the data "
    "may be used for non-commercial purposes only and that the UIST 2013 paper must be cited. No redistribution "
    "permission is stated, so FewGaze never commits or publishes dataset images."
)


def inspect_images(root: Path, relpaths: list[str], hash_files: bool) -> pd.DataFrame:
    rows = []
    for i, rel in enumerate(relpaths):
        p = root / rel
        row = {"relpath": rel, "readable": True, "width": None, "height": None, "format": None, "mode": None, "sha256": None}
        try:
            with Image.open(p) as im:
                row.update(width=im.width, height=im.height, format=im.format, mode=im.mode)
                im.verify()  # structural check without full decode
        except (OSError, ValueError) as exc:
            row["readable"] = False
            row["error"] = str(exc)
        if hash_files and row["readable"]:
            row["sha256"] = file_sha256(p)
        rows.append(row)
        if (i + 1) % 1000 == 0:
            log.info("inspected %d/%d images", i + 1, len(relpaths))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 duplicate detection (faster)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    try:
        man = load_manifest(cfg)
    except DatasetError as exc:
        log.error("%s", exc)
        return 2
    rec, invalid = man.records, man.invalid
    root = resolve_path(cfg["dataset"]["root"])
    log.info("parsed %d images, %d invalid files", len(rec), len(invalid))

    img = inspect_images(root, rec["relpath"].tolist(), hash_files=not args.no_hash)
    rec = rec.merge(img, on="relpath", how="left")
    labels = derive_labels(rec, cfg["labels"])
    rec_l = rec.merge(labels, on="sample_id")

    # completeness of the subject x pose x gaze grid
    poses = sorted(rec["head_pose"].unique())
    gv, gh = sorted(rec["gaze_v"].unique()), sorted(rec["gaze_h"].unique())
    present = set(zip(rec["subject"], rec["head_pose"], rec["gaze_v"], rec["gaze_h"]))
    missing = [
        {"subject": s, "head_pose": int(p), "gaze_v": int(v), "gaze_h": int(h)}
        for s in sorted(rec["subject"].unique())
        for p in poses
        for v in gv
        for h in gh
        if (s, p, v, h) not in present
    ]
    dup_groups: dict[str, list[str]] = defaultdict(list)
    if "sha256" in rec and rec["sha256"].notna().any():
        for sha, g in rec.dropna(subset=["sha256"]).groupby("sha256"):
            if len(g) > 1:
                dup_groups[str(sha)] = g["relpath"].tolist()

    per_subject = rec_l.groupby("subject")["label_status"].value_counts().unstack(fill_value=0)
    subjects_one_class = per_subject.index[(per_subject.get("positive", 0) == 0) | (per_subject.get("negative", 0) == 0)].tolist()
    near_camera = rec_l[(rec_l["label_status"] == "negative") & (rec_l["gaze_v"] == 0) & (rec_l["gaze_h"].abs() == 5)]

    audit = {
        "created": utc_now(),
        "dataset_root": str(root),
        "manifest_fingerprint": manifest_fingerprint(man.records),
        "n_images": int(len(rec)),
        "n_invalid_files": int(len(invalid)),
        "invalid_files": invalid.to_dict(orient="records"),
        "n_unreadable": int((~rec["readable"]).sum()),
        "unreadable": rec.loc[~rec["readable"], "relpath"].tolist(),
        "n_subjects": int(rec["subject"].nunique()),
        "image_sizes": rec.groupby(["width", "height"]).size().rename("n").reset_index().to_dict(orient="records"),
        "formats": rec["format"].value_counts().to_dict(),
        "modes": rec["mode"].value_counts().to_dict(),
        "distances_m": sorted(int(d) for d in rec["distance"].unique()),
        "head_poses": [int(p) for p in poses],
        "gaze_v": [int(v) for v in gv],
        "gaze_h": [int(h) for h in gh],
        "n_gaze_conditions_per_pose": int(len(gv) * len(gh)),
        "expected_grid_size": int(rec["subject"].nunique() * len(poses) * len(gv) * len(gh)),
        "n_missing_grid_cells": len(missing),
        "missing_grid_cells": missing,
        "n_duplicate_content_groups": len(dup_groups),
        "duplicate_content_groups": dup_groups,
        "label_definition": cfg["labels"],
        "label_status_counts": labels["label_status"].value_counts().to_dict(),
        "per_head_pose_class_counts": rec_l.groupby(["head_pose", "label_status"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "subjects_with_single_class": subjects_one_class,
        "per_subject_positive_range": [
            int(per_subject.get("positive", pd.Series([0])).min()),
            int(per_subject.get("positive", pd.Series([0])).max()),
        ],
        "near_camera_negatives": {
            "definition": "negatives with gaze_v == 0 and |gaze_h| == 5 deg (closest non-camera condition)",
            "n": int(len(near_camera)),
            "note": "Smith et al. (2013) estimate their detector's tolerance at ~2.5 deg horizontally; these are the hardest negatives, kept as negatives by default.",
        },
        "identity_split_feasible": bool(rec["subject"].nunique() >= int(cfg["splits"]["n_folds"]) and not subjects_one_class),
        "identity_split_note": "Subject IDs are encoded in both folder names and filenames; they are cross-checked during manifest parsing.",
        "license_note": LICENSE_NOTE,
        "runtime": runtime_info(),
    }
    write_json(audit, out / "audit.json")
    man.records.to_csv(out / "manifest.csv", index=False)
    labels.to_csv(out / "labels.csv", index=False)
    per_subject.to_csv(out / "per_subject_label_counts.csv")
    write_summary(audit, per_subject, out / "audit_summary.md")
    make_composition_figure(rec_l, output_dir(cfg) / "figures")
    log.info("audit written to %s", out)
    blocking = audit["n_unreadable"] > 0.05 * len(rec) or not audit["identity_split_feasible"]
    return 3 if blocking else 0


def write_summary(a: dict, per_subject: pd.DataFrame, path: Path) -> None:
    sc = a["label_status_counts"]
    sizes = ", ".join("{}x{} (n={})".format(s["width"], s["height"], s["n"]) for s in a["image_sizes"])
    lines = [
        "# Dataset audit summary",
        "",
        f"Generated {a['created']} from `{a['dataset_root']}` (manifest fingerprint `{a['manifest_fingerprint']}`).",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Images parsed | {a['n_images']} |",
        f"| Files failing filename parsing | {a['n_invalid_files']} |",
        f"| Unreadable images | {a['n_unreadable']} |",
        f"| Subjects | {a['n_subjects']} |",
        f"| Head poses (deg) | {a['head_poses']} |",
        f"| Vertical gaze (deg) | {a['gaze_v']} |",
        f"| Horizontal gaze (deg) | {a['gaze_h']} |",
        f"| Capture distance (m) | {a['distances_m']} |",
        f"| Image sizes | {sizes} |",
        f"| Missing subject/pose/gaze cells | {a['n_missing_grid_cells']} of {a['expected_grid_size']} |",
        f"| Duplicate-content groups (SHA-256) | {a['n_duplicate_content_groups']} |",
        f"| Positive (camera-directed) | {sc.get('positive', 0)} |",
        f"| Negative | {sc.get('negative', 0)} |",
        f"| Excluded by config | {sc.get('excluded', 0)} |",
        f"| Unlabeled | {sc.get('unlabeled', 0)} |",
        f"| Near-camera negatives (V=0, abs(H)=5) | {a['near_camera_negatives']['n']} |",
        f"| Subjects with only one class | {len(a['subjects_with_single_class'])} |",
        f"| Positives per subject (min, max) | {a['per_subject_positive_range']} |",
        f"| Identity-based splitting feasible | {a['identity_split_feasible']} |",
        "",
        "## Label definition",
        "",
        f"`{a['label_definition']}`",
        "",
        "Positive = gaze directed at the camera (V = 0 deg, H = 0 deg), which Smith et al. (UIST 2013) call",
        "*gaze locking*; gaze angles are measured relative to each camera position, so this holds for every head pose.",
        "This is a controlled proxy for eye contact with a camera, not interpersonal eye contact.",
        "",
        "## Licensing",
        "",
        a["license_note"],
        "",
    ]
    if a["missing_grid_cells"]:
        lines += ["## Missing cells", "", *[f"- {m}" for m in a["missing_grid_cells"][:50]], ""]
    if a["invalid_files"]:
        lines += ["## Invalid files", "", *[f"- `{f['relpath']}`: {f['reason']}" for f in a["invalid_files"][:50]], ""]
    if a["duplicate_content_groups"]:
        lines += ["## Duplicate content", "", *[f"- {v}" for v in list(a["duplicate_content_groups"].values())[:50]], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_composition_figure(rec: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from fewgaze.plots import apply_style, save

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    lab = rec[rec["label"] >= 0]
    counts = lab.groupby(["head_pose", "label"]).size().unstack(fill_value=0)
    x = np.arange(len(counts))
    axes[0].bar(x, counts.get(0, 0), color="#9aa0a6", label="negative")
    axes[0].bar(x, counts.get(1, 0), bottom=counts.get(0, 0), color="#c2410c", label="positive (camera-directed)")
    axes[0].set_xticks(x, [f"{p:+d}°" if p else "0°" for p in counts.index])
    axes[0].set_xlabel("head pose relative to camera")
    axes[0].set_ylabel("images")
    axes[0].set_title("Images per head pose")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    grid = lab.pivot_table(index="gaze_v", columns="gaze_h", values="sample_id", aggfunc="count").sort_index(ascending=False)
    axes[1].imshow(np.zeros(grid.shape), cmap="Greys", vmin=0, vmax=1)  # uniform background: the grid is complete
    axes[1].set_xticks(range(grid.shape[1]), [f"{h}" for h in grid.columns])
    axes[1].set_yticks(range(grid.shape[0]), [f"{v}" for v in grid.index])
    axes[1].set_xlabel("horizontal gaze H (deg, relative to camera)")
    axes[1].set_ylabel("vertical gaze V (deg)")
    axes[1].set_title("Gaze conditions: labelled images per cell")
    axes[1].grid(False)
    r0 = list(grid.index).index(0)
    c0 = list(grid.columns).index(0)
    axes[1].add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), 1, 1, fill=True, fc="#c2410c", ec="#c2410c", lw=0))
    for (i, j), v in np.ndenumerate(grid.to_numpy()):
        axes[1].text(j, i, int(v), ha="center", va="center", fontsize=7, color="white" if (i, j) == (r0, c0) else "black")
    for k in range(grid.shape[1] + 1):
        axes[1].axvline(k - 0.5, color="white", lw=2)
    for k in range(grid.shape[0] + 1):
        axes[1].axhline(k - 0.5, color="white", lw=2)
    per_subj = lab.groupby("subject")["label"].agg(["sum", "count"]).sort_index()
    xs = np.arange(len(per_subj))
    axes[2].bar(xs, per_subj["count"] - per_subj["sum"], color="#9aa0a6", width=0.8, label="negative")
    axes[2].bar(xs, per_subj["sum"], bottom=per_subj["count"] - per_subj["sum"], color="#c2410c", width=0.8, label="positive")
    axes[2].set_xticks([])
    axes[2].set_xlabel(f"subject (n = {len(per_subj)})")
    axes[2].set_ylabel("labelled images")
    axes[2].set_title(
        f"Per subject: {int(per_subj['sum'].min())} positives, {int((per_subj['count'] - per_subj['sum']).min())}-{int((per_subj['count'] - per_subj['sum']).max())} negatives"
    )
    fig.suptitle(f"Columbia Gaze composition: {len(lab)} labelled images, {lab['subject'].nunique()} subjects", y=1.02)
    save(fig, fig_dir / "fig2_dataset_composition")


if __name__ == "__main__":
    sys.exit(main())

"""Dataset manifest: discover images and parse the source annotations encoded in filenames.

The manifest holds ONLY original annotations (subject, head pose, gaze angles) plus file metadata.
Derived binary labels live in ``fewgaze.labels`` and are stored separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fewgaze.config import resolve_path
from fewgaze.utils import stable_hash

ANNOTATION_FIELDS = ["subject", "distance", "head_pose", "gaze_v", "gaze_h"]
INT_FIELDS = ["distance", "head_pose", "gaze_v", "gaze_h"]


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be found or parsed."""


@dataclass
class ManifestResult:
    records: pd.DataFrame  # one row per valid, parsed image
    invalid: pd.DataFrame  # files that matched an image extension but failed parsing


def parse_filename(name: str, pattern: re.Pattern[str]) -> dict[str, Any] | None:
    m = pattern.match(name)
    if m is None:
        return None
    out: dict[str, Any] = dict(m.groupdict())
    for f in INT_FIELDS:
        if f in out and out[f] is not None:
            out[f] = int(out[f])
    return out


def build_manifest(root: str | Path, filename_regex: str, image_extensions: list[str]) -> ManifestResult:
    root = resolve_path(root)
    if not root.exists():
        raise DatasetError(
            f"Dataset root does not exist: {root}\nSee data/README.md for how to obtain the Columbia Gaze Dataset and where to unpack it."
        )
    pattern = re.compile(filename_regex)
    missing = [g for g in ["subject", "head_pose", "gaze_v", "gaze_h"] if g not in pattern.groupindex]
    if missing:
        raise DatasetError(f"filename_regex lacks required named groups: {missing}")
    exts = {e.lower() for e in image_extensions}
    rows, bad = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(root).as_posix()
        parsed = parse_filename(path.name, pattern)
        if parsed is None:
            bad.append({"relpath": rel, "reason": "filename_does_not_match_regex"})
            continue
        folder = path.parent.name
        rows.append(
            {
                "sample_id": path.stem,
                "relpath": rel,
                "folder": folder,
                "file_bytes": path.stat().st_size,
                **parsed,
            }
        )
    records = pd.DataFrame(rows)
    invalid = pd.DataFrame(bad, columns=["relpath", "reason"])
    if records.empty:
        raise DatasetError(f"No images matching the configured pattern were found under {root}")
    # Folder/filename subject disagreement is an annotation inconsistency: flag, don't guess.
    mismatch = records["folder"] != records["subject"]
    if mismatch.any() and records["folder"].nunique() > 1:
        extra = records.loc[mismatch, ["relpath"]].assign(reason="folder_subject_mismatch")
        invalid = pd.concat([invalid, extra], ignore_index=True)
        records = records.loc[~mismatch]
    dup = records["sample_id"].duplicated(keep=False)
    if dup.any():
        extra = records.loc[dup, ["relpath"]].assign(reason="duplicate_sample_id")
        invalid = pd.concat([invalid, extra], ignore_index=True)
        records = records.loc[~dup]
    records = records.sort_values("sample_id").reset_index(drop=True)
    return ManifestResult(records=records, invalid=invalid.reset_index(drop=True))


def load_manifest(cfg: dict[str, Any]) -> ManifestResult:
    ds = cfg["dataset"]
    return build_manifest(ds["root"], ds["filename_regex"], ds.get("image_extensions", [".jpg"]))


def manifest_fingerprint(records: pd.DataFrame) -> str:
    """Hash of the sample IDs, annotations and file sizes. Changes if the dataset changes."""
    cols = ["sample_id", *[c for c in ANNOTATION_FIELDS if c in records.columns], "file_bytes"]
    return stable_hash(records[cols].astype(str).values.tolist())


def image_path(cfg: dict[str, Any], relpath: str) -> Path:
    return resolve_path(cfg["dataset"]["root"]) / relpath

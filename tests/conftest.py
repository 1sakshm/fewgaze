"""Synthetic fixtures: a tiny Columbia-style dataset (real files on disk, 8x8 JPEGs) and
synthetic feature tables whose signal depends on the gaze annotation."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fewgaze.config import REPO_ROOT, load_config
from fewgaze.dataset import build_manifest
from fewgaze.extraction import MethodSpec
from fewgaze.features import FeatureTable
from fewgaze.labels import derive_labels, labelled_view

POSES = [-30, -15, 0, 15, 30]
GAZE_V = [-10, 0, 10]
GAZE_H = [-15, -10, -5, 0, 5, 10, 15]


def make_fake_dataset(root: Path, n_subjects: int = 10) -> Path:
    for s in range(1, n_subjects + 1):
        sid = f"{s:04d}"
        d = root / sid
        d.mkdir(parents=True, exist_ok=True)
        for p in POSES:
            for v in GAZE_V:
                for h in GAZE_H:
                    Image.new("RGB", (8, 8), (s * 10 % 255, 0, 0)).save(d / f"{sid}_2m_{p}P_{v}V_{h}H.jpg")
    return root


@pytest.fixture(scope="session")
def base_cfg() -> dict:
    return load_config(REPO_ROOT / "configs" / "experiment.yaml")


@pytest.fixture()
def fake_root(tmp_path: Path) -> Path:
    return make_fake_dataset(tmp_path / "columbia")


@pytest.fixture()
def cfg(base_cfg: dict, fake_root: Path, tmp_path: Path) -> dict:
    c = copy.deepcopy(base_cfg)
    c["dataset"]["root"] = str(fake_root)
    c["project"]["output_dir"] = str(tmp_path / "results")
    c["project"]["cache_dir"] = str(tmp_path / "cache")
    c["sampling"]["budgets"] = [5, 10, "all"]
    c["sampling"]["seeds"] = [0, 1]
    c["noise"]["budgets"] = [10]
    c["noise"]["rates"] = [0.2]
    c["statistics"]["n_bootstrap"] = 20
    c["labels"]["exclude_sample_ids"] = []  # real-data exclusions do not apply to synthetic data
    return c


@pytest.fixture()
def labelled(cfg: dict) -> pd.DataFrame:
    man = build_manifest(cfg["dataset"]["root"], cfg["dataset"]["filename_regex"], [".jpg"])
    return labelled_view(man.records, derive_labels(man.records, cfg["labels"]))


def synthetic_tables(data: pd.DataFrame, seed: int = 0, drop: int = 0) -> dict[str, FeatureTable]:
    """'geo': 3-d features where positives sit near the origin; 'emb': 16-d noisy embedding."""
    rng = np.random.default_rng(seed)
    ids = data["sample_id"].to_numpy()
    g = np.stack([data["gaze_h"] / 15, data["gaze_v"] / 10, data["head_pose"] / 30], axis=1).astype(float)
    geo = g + rng.normal(0, 0.15, g.shape)
    emb = np.concatenate([g, rng.normal(0, 1, (len(ids), 13))], axis=1) + rng.normal(0, 0.2, (len(ids), 16))
    valid = np.ones(len(ids), dtype=bool)
    if drop:
        valid[rng.choice(len(ids), drop, replace=False)] = False
    status = np.where(valid, "ok", "no_face")
    out = {}
    for name, X in (("geo", geo), ("emb", emb)):
        X = X.astype(np.float32).copy()
        X[~valid] = np.nan
        out[name] = FeatureTable(name, ids, X, valid.copy(), status.copy(), [f"f{i}" for i in range(X.shape[1])], {})
    return out


SYNTH_METHODS = {
    "geo_poly": MethodSpec("geo_poly", "geo", "trained", "geometry", True, 2),
    "emb_lin": MethodSpec("emb_lin", "emb", "trained", "dinov2", True, 1),
}

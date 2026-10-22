"""Feature-extraction orchestration and the method registry.

Cache dependency chain:  geometry (+ face crops)  ->  dinov2 / l2cs / clip
Each downstream cache key includes the geometry key, so changing crop parameters invalidates
every appearance feature automatically.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fewgaze.config import cache_dir, resolve_path
from fewgaze.features import FeatureCache, FeatureTable, batched, cache_key
from fewgaze.utils import get_logger, write_json

log = get_logger("fewgaze.extract")


@dataclass(frozen=True)
class MethodSpec:
    name: str
    cache: str  # feature cache name
    kind: str  # 'trained' (downstream LR) or 'zeroshot' (fixed score, no labels)
    family: str  # geometry | dinov2 | l2cs | clip
    primary: bool  # shown in the main comparison; otherwise an ablation
    poly: int = 1
    columns: tuple[int, ...] | None = None
    label: str = ""


def method_registry(cfg: dict[str, Any]) -> dict[str, MethodSpec]:
    m = cfg["methods"]
    specs = [
        MethodSpec(
            "geometry",
            "geometry",
            "trained",
            "geometry",
            True,
            int(m["geometry"].get("polynomial_degree", 2)),
            label="Geometry (MediaPipe, poly-2 LR)",
        ),
        MethodSpec("geometry_linear", "geometry", "trained", "geometry", False, 1, label="Geometry (linear LR)"),
        MethodSpec("dinov2", "dinov2_cls", "trained", "dinov2", True, 1, label="DINOv2-S CLS (linear probe)"),
        MethodSpec("dinov2_patchmean", "dinov2_patchmean", "trained", "dinov2", False, 1, label="DINOv2-S mean-patch (linear probe)"),
        MethodSpec(
            "l2cs", "l2cs_angles", "trained", "l2cs", True, int(m["l2cs"].get("polynomial_degree", 2)), label="L2CS-Net angles (poly-2 LR)"
        ),
        MethodSpec("l2cs_embed", "l2cs_embed", "trained", "l2cs", False, 1, label="L2CS-Net 2048-d features (linear probe)"),
        MethodSpec("clip_probe", "clip_embed", "trained", "clip", False, 1, label="CLIP ViT-B/32 (linear probe)"),
        MethodSpec("l2cs_zeroshot", "l2cs_angles", "zeroshot", "l2cs", True, label="L2CS-Net angle to camera (0 labels)"),
        MethodSpec("clip_zeroshot", "clip_zeroshot", "zeroshot", "clip", True, label="CLIP zero-shot prompts (0 labels)"),
    ]
    family_enabled = {
        "geometry": m["geometry"].get("enabled", True),
        "dinov2": m["dinov2"].get("enabled", True),
        "l2cs": m["l2cs"].get("enabled", True),
        "clip": m["clip"].get("enabled", True),
    }
    out = {s.name: s for s in specs if family_enabled[s.family]}
    if not m.get("geometry_linear", {}).get("enabled", True):
        out.pop("geometry_linear", None)
    return out


def cache_keys(cfg: dict[str, Any], fingerprint: str) -> dict[str, str]:
    pre, m = cfg["preprocessing"], cfg["methods"]
    geo = cache_key({"method": "geometry", "model": m["geometry"]["landmarker_model_url"]}, pre, fingerprint)
    keys = {"geometry": geo}
    dino = cache_key({"method": "dinov2", **{k: m["dinov2"][k] for k in ("model_id", "input_size")}, "crops": geo}, pre, fingerprint)
    keys["dinov2_cls"] = keys["dinov2_patchmean"] = dino
    l2 = cache_key(
        {"method": "l2cs", **{k: v for k, v in m["l2cs"].items() if k not in ("batch_size", "enabled", "polynomial_degree")}, "crops": geo},
        pre,
        fingerprint,
    )
    keys["l2cs_angles"] = keys["l2cs_embed"] = l2
    cl = cache_key(
        {"method": "clip", **{k: v for k, v in m["clip"].items() if k not in ("batch_size", "enabled")}, "crops": geo}, pre, fingerprint
    )
    keys["clip_embed"] = keys["clip_zeroshot"] = cl
    return keys


def crop_dir(cfg: dict[str, Any], keys: dict[str, str]) -> Path:
    return cache_dir(cfg) / f"crops-{keys['geometry']}"


def open_cache(cfg: dict[str, Any], keys: dict[str, str], name: str) -> FeatureCache:
    return FeatureCache(cache_dir(cfg), name, keys[name])


def load_tables(cfg: dict[str, Any], keys: dict[str, str], names: set[str], ids: list[str]) -> dict[str, FeatureTable]:
    return {n: open_cache(cfg, keys, n).load(ids) for n in sorted(names)}


# --------------------------------------------------------------------------------------------
# Extraction stages
# --------------------------------------------------------------------------------------------


def run_geometry(cfg: dict[str, Any], records: pd.DataFrame, keys: dict[str, str], chunk: int = 64) -> FeatureTable:
    from fewgaze import geometry as G

    cache = open_cache(cfg, keys, "geometry")
    ids = records["sample_id"].tolist()
    if cache.is_complete():
        log.info("geometry: cache complete (%s)", cache.dir)
        return cache.load(ids)
    model_path = ensure_landmarker(cfg)
    crops = crop_dir(cfg, keys)
    crops.mkdir(parents=True, exist_ok=True)
    done = cache.done_ids()
    todo = records[~records["sample_id"].isin(done)]
    log.info("geometry: %d done, %d to process", len(done), len(todo))
    ext = G.GeometryExtractor(model_path, cfg["preprocessing"])
    root = resolve_path(cfg["dataset"]["root"])
    prior = _prior_timing(cache)
    t0, n = time.perf_counter(), 0
    try:
        for batch in batched(list(todo.itertuples(index=False)), chunk):
            bid, bX, bst = [], [], []
            for r in batch:
                res = ext(root / r.relpath)
                if res.crop is not None:
                    res.crop.save(crops / f"{r.sample_id}.jpg", quality=95)
                bid.append(r.sample_id)
                bX.append(res.features)
                bst.append(res.status)
            cache.write_chunk(bid, np.stack(bX), bst)
            n += len(batch)
            _save_timing(cache, prior, time.perf_counter() - t0, n)
            el = time.perf_counter() - t0
            log.info("geometry: %d/%d (%.2f s/img)", n, len(todo), el / n)
    finally:
        ext.close()
    elapsed = time.perf_counter() - t0
    meta = {"timing": _timing(prior, elapsed, n), "landmarker": cfg["methods"]["geometry"]["landmarker_model_url"]}
    return cache.finalize(ids, G.FEATURE_NAMES, meta)


def ensure_landmarker(cfg: dict[str, Any]) -> Path:
    from fewgaze.geometry import ensure_model

    url = cfg["methods"]["geometry"]["landmarker_model_url"]
    return ensure_model(url, cache_dir(cfg) / "weights" / "face_landmarker.task")


def _timing(prior: dict[str, float], elapsed: float, n: int) -> dict[str, float]:
    """Cumulative timing over all (possibly resumed) extraction sessions."""
    total_t = prior.get("seconds", 0.0) + elapsed
    total_n = int(prior.get("n_images", 0)) + n
    return {"seconds": total_t, "n_images": total_n, "seconds_per_image": total_t / max(total_n, 1)}


def _prior_timing(cache: FeatureCache) -> dict[str, float]:
    from fewgaze.utils import read_json

    p = cache.dir / "timing_cumulative.json"
    return read_json(p) if p.exists() else {}


def _save_timing(cache: FeatureCache, prior: dict[str, float], elapsed: float, n: int) -> None:
    write_json(_timing(prior, elapsed, n), cache.dir / "timing_cumulative.json")


def run_appearance(
    cfg: dict[str, Any],
    records: pd.DataFrame,
    keys: dict[str, str],
    geometry: FeatureTable,
    outputs: dict[str, str],  # extractor output key -> cache name
    factory: Callable[[], Callable[[list[Any]], dict[str, np.ndarray]]],
    batch_size: int,
    model_info: Callable[[Any], dict[str, Any]],
) -> dict[str, FeatureTable]:
    from fewgaze.embeddings import load_crop

    caches = {out: open_cache(cfg, keys, name) for out, name in outputs.items()}
    ids = records["sample_id"].tolist()
    if all(c.is_complete() for c in caches.values()):
        log.info("%s: cache complete", list(outputs.values()))
        return {c.name: c.load(ids) for c in caches.values()}
    done = set.intersection(*(c.done_ids() for c in caches.values()))
    valid = dict(zip(geometry.sample_ids, geometry.valid))
    geo_status = dict(zip(geometry.sample_ids, geometry.status))
    todo = [s for s in ids if s not in done]
    log.info("%s: %d done, %d to process", list(outputs.values()), len(done), len(todo))
    model = factory()
    crops = crop_dir(cfg, keys)
    dims: dict[str, int] = {}
    priors = {out: _prior_timing(c) for out, c in caches.items()}
    t0, n_img = time.perf_counter(), 0
    for batch in batched(todo, batch_size):
        ok = [s for s in batch if valid[s]]
        result = model([load_crop(crops, s) for s in ok]) if ok else {}
        n_img += len(ok)
        for out, cache in caches.items():
            if ok:
                dims[out] = result[out].shape[1]
            d = dims.get(out) or _infer_dim(model, out)
            X = np.full((len(batch), d), np.nan, dtype=np.float32)
            pos = {s: i for i, s in enumerate(ok)}
            for i, s in enumerate(batch):
                if s in pos:
                    X[i] = result[out][pos[s]]
            cache.write_chunk(batch, X, ["ok" if valid[s] else str(geo_status[s]) for s in batch])
            _save_timing(cache, priors[out], time.perf_counter() - t0, n_img)
        if n_img and (n_img // batch_size) % 10 == 0:
            log.info("%s: %d images (%.3f s/img)", list(outputs.values()), n_img, (time.perf_counter() - t0) / n_img)
    elapsed = time.perf_counter() - t0
    info = model_info(model)
    tables = {}
    for out, cache in caches.items():
        d = dims.get(out) or _infer_dim(model, out)
        meta = {"timing": _timing(priors[out], elapsed, n_img), **info}
        tables[cache.name] = cache.finalize(ids, [f"{out}_{i}" for i in range(d)], meta)
    return tables


def _infer_dim(model: Any, out: str) -> int:
    fixed = {"angles": 2, "zeroshot": 1}
    if out in fixed:
        return fixed[out]
    return int(getattr(model, "dim", 2048))


def param_count(module: Any) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def write_extraction_summary(cfg: dict[str, Any], tables: dict[str, FeatureTable], path: Path) -> None:
    rows = {}
    for name, t in tables.items():
        rows[name] = {
            "n_samples": int(len(t.sample_ids)),
            "n_valid": int(t.valid.sum()),
            "status_counts": pd.Series(t.status).value_counts().to_dict(),
            "dim": int(t.X.shape[1]),
            "timing": t.meta.get("timing"),
            "model": {k: v for k, v in t.meta.items() if k in ("model_id", "params", "device", "weights", "landmarker")},
        }
    write_json(rows, path)

"""Study 2, step 1: eye boxes on the cached face crops, then every-layer global/eye tokens per model.

Outputs (cache, never committed): cache/study2/eyeboxes.npz, cache/study2/<model>.npz
Summary (committed):             results/study2/extraction.json
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from fewgaze.config import cache_dir, load_config, output_dir
from fewgaze.dataset import load_manifest, manifest_fingerprint
from fewgaze.embeddings import load_crop
from fewgaze.extraction import cache_keys, crop_dir, ensure_landmarker
from fewgaze.features import batched
from fewgaze.probing import MODELS, LayerExtractor, eye_boxes_from_landmarks, eye_patch_masks
from fewgaze.probing import cache_dir as s2_dir
from fewgaze.utils import get_logger, read_json, write_json

log = get_logger("fewgaze.study2")


def eye_boxes(cfg, ids, crops, out_path):
    if out_path.exists():
        z = np.load(out_path)
        if list(z["sample_ids"]) == list(ids):
            return z["boxes"], z["ok"]
    from fewgaze.geometry import GeometryExtractor

    ext = GeometryExtractor(ensure_landmarker(cfg), cfg["preprocessing"])
    boxes = np.full((len(ids), 2, 4), np.nan)
    ok = np.zeros(len(ids), dtype=bool)
    t0 = time.perf_counter()
    try:
        for i, s in enumerate(ids):
            crop = load_crop(crops, s)
            res = ext.landmarker.detect(ext.mp.Image(image_format=ext.mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(crop))))
            if res.face_landmarks:
                pts = np.array([[p.x * crop.width, p.y * crop.height] for p in res.face_landmarks[0]])
                boxes[i] = eye_boxes_from_landmarks(pts)
                ok[i] = True
            if (i + 1) % 1000 == 0:
                log.info("eye boxes %d/%d (%.3f s/img)", i + 1, len(ids), (time.perf_counter() - t0) / (i + 1))
    finally:
        ext.close()
    np.savez(out_path, sample_ids=np.array(ids), boxes=boxes, ok=ok)
    return boxes, ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", default=[m.key for m in MODELS])
    ap.add_argument("--batch-size", type=int, default=24)
    args = ap.parse_args()
    cfg = load_config(args.config)
    man = load_manifest(cfg)
    ids = man.records["sample_id"].tolist()
    keys = cache_keys(cfg, manifest_fingerprint(man.records))
    crops = crop_dir(cfg, keys)
    d = s2_dir(cache_dir(cfg))
    boxes, ok = eye_boxes(cfg, ids, crops, d / "eyeboxes.npz")
    log.info("eye boxes found for %d/%d crops", int(ok.sum()), len(ids))

    summary_path = output_dir(cfg) / "study2" / "extraction.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    summary["eye_boxes"] = {"n_ok": int(ok.sum()), "n": len(ids)}
    for spec in [m for m in MODELS if m.key in args.models]:
        path = d / f"{spec.key}.npz"
        if path.exists():
            log.info("%s: cached", spec.key)
            continue
        ext = LayerExtractor(spec)
        n_eye_patches = []
        G, E = None, None
        t0 = time.perf_counter()
        for bi, batch in enumerate(batched(list(range(len(ids))), args.batch_size)):
            valid = [i for i in batch if ok[i]]
            if not valid:
                continue
            masks = [eye_patch_masks(boxes[i], ext.grid, spec.patch) for i in valid]
            n_eye_patches += [len(m[0]) + len(m[1]) for m in masks]
            g, e = ext([load_crop(crops, ids[i]) for i in valid], masks)
            if G is None:
                G = np.full((len(ids), g.shape[1], g.shape[2]), np.nan, dtype=np.float16)
                E = np.full((len(ids), e.shape[1], e.shape[2]), np.nan, dtype=np.float16)
            G[valid], E[valid] = g.astype(np.float16), e.astype(np.float16)
            if bi % 40 == 0:
                log.info("%s: %d/%d", spec.key, batch[-1] + 1, len(ids))
        secs = time.perf_counter() - t0
        np.savez(path, sample_ids=np.array(ids), glob=G, eyes=E, ok=ok)
        summary[spec.key] = {
            "hf_id": spec.hf_id,
            "label": spec.label,
            "params": ext.params,
            "layers": int(G.shape[1]),
            "dim": int(G.shape[2]),
            "grid": ext.grid,
            "patch": spec.patch,
            "seconds": secs,
            "seconds_per_image": secs / max(int(ok.sum()), 1),
            "eye_patches_per_image_mean": float(np.mean(n_eye_patches)),
            "eye_patches_per_image_min": int(np.min(n_eye_patches)),
        }
        write_json(summary, summary_path)
        log.info("%s: done in %.0f s", spec.key, secs)
        del ext
        import torch

        torch.cuda.empty_cache()
    write_json(summary, summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Extract and cache features for every enabled method family.

Stages run in dependency order: geometry (MediaPipe + shared face crops) -> dinov2 -> l2cs -> clip.
Each stage is resumable and skipped when its cache is complete. A failing optional stage (e.g.
missing L2CS weights) is reported and the remaining stages still run.

Exit codes: 0 all requested stages complete, 2 dataset problem, 4 one or more stages failed.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from fewgaze.config import cache_dir, load_config, output_dir, resolve_path
from fewgaze.dataset import DatasetError, load_manifest, manifest_fingerprint
from fewgaze.extraction import cache_keys, param_count, run_appearance, run_geometry, write_extraction_summary
from fewgaze.utils import get_logger

log = get_logger("fewgaze.extract")
STAGES = ["geometry", "dinov2", "l2cs", "clip", "eyes"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES[:4], help="'eyes' = post-hoc P2 stage")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N images (smoke test)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    try:
        man = load_manifest(cfg)
    except DatasetError as exc:
        log.error("%s", exc)
        return 2
    records = man.records if args.limit is None else man.records.head(args.limit)
    keys = cache_keys(cfg, manifest_fingerprint(records))
    m = cfg["methods"]
    failures, tables = [], {}

    geo = None
    if "geometry" in args.stages or any(s in args.stages for s in STAGES[1:]):
        geo = run_geometry(cfg, records, keys)
        tables["geometry"] = geo
        log.info("geometry: %d/%d images with a detected face", int(geo.valid.sum()), len(geo.valid))

    def stage(name, outputs, factory, bs, info):
        if name not in args.stages or not m[name].get("enabled", True):
            return
        try:
            tables.update(run_appearance(cfg, records, keys, geo, outputs, factory, bs, info))
        except Exception as exc:  # optional stages must not abort the others
            failures.append(name)
            log.error("stage %s failed: %s\n%s", name, exc, traceback.format_exc(limit=3))

    def dino():
        from fewgaze.embeddings import DinoV2Extractor

        return DinoV2Extractor(m["dinov2"]["model_id"], int(m["dinov2"]["input_size"]))

    stage(
        "dinov2",
        {"cls": "dinov2_cls", "patchmean": "dinov2_patchmean"},
        dino,
        int(m["dinov2"]["batch_size"]),
        lambda x: {"model_id": m["dinov2"]["model_id"], "params": param_count(x.model), "device": x.device},
    )

    def l2cs():
        from fewgaze.gaze_estimator import L2CSExtractor

        c = m["l2cs"]
        return L2CSExtractor(
            resolve_path(c["weights_path"]),
            int(c["num_bins"]),
            float(c["bin_width_deg"]),
            float(c["angle_offset_deg"]),
            int(c["input_size"]),
        )

    stage(
        "l2cs",
        {"angles": "l2cs_angles", "embedding": "l2cs_embed"},
        l2cs,
        int(m["l2cs"]["batch_size"]),
        lambda x: {
            "model_id": "L2CS-Net ResNet-50 (Gaze360)",
            "weights": str(m["l2cs"]["weights_path"]),
            "params": param_count(x.model),
            "device": x.device,
        },
    )

    def clip():
        from fewgaze.embeddings import ClipExtractor

        c = m["clip"]
        return ClipExtractor(c["model_id"], c["positive_prompt"], c["negative_prompt"])

    stage(
        "clip",
        {"embedding": "clip_embed", "zeroshot": "clip_zeroshot"},
        clip,
        int(m["clip"]["batch_size"]),
        lambda x: {"model_id": m["clip"]["model_id"], "params": param_count(x.model), "device": x.device},
    )

    if "eyes" in args.stages and m.get("dinov2_eyes", {}).get("enabled", False):
        try:
            from fewgaze.embeddings import DinoV2Extractor
            from fewgaze.extraction import run_eye_crops

            eyes_table, eyes_dir = run_eye_crops(cfg, records, keys, geo)
            size = (int(m["dinov2_eyes"]["size"][0]), int(m["dinov2_eyes"]["size"][1]))
            tables.update(
                run_appearance(
                    cfg,
                    records,
                    keys,
                    eyes_table,
                    {"cls": "dinov2_eyes"},
                    lambda: DinoV2Extractor(m["dinov2"]["model_id"], size),
                    int(m["dinov2"]["batch_size"]),
                    lambda x: {"model_id": m["dinov2"]["model_id"], "params": param_count(x.model), "device": x.device},
                    crops=eyes_dir,
                )
            )
        except Exception as exc:
            failures.append("eyes")
            log.error("stage eyes failed: %s %s", exc, traceback.format_exc(limit=3))

    if args.limit is None:
        write_extraction_summary(cfg, tables, output_dir(cfg) / "tables" / "extraction_summary.json")
    log.info("caches under %s", cache_dir(cfg))
    if failures:
        log.error("failed stages: %s", failures)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())

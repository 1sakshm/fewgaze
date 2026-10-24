"""Study 3 (docs/methodology.md): a generative VLM (Qwen2.5-VL-3B-Instruct, 4-bit) asked whether the person looks at
the camera, zero-shot on every image (S3-a) and with 2 in-context examples per class on a fixed subset (S3-b).

Run in the separate `fewgaze-vlm` environment (transformers 4.51.3, bitsandbytes 0.45.5):
    python scripts/study3_vlm.py --config configs/experiment.yaml --stage zeroshot
    python scripts/study3_vlm.py --config configs/experiment.yaml --stage fewshot
    python scripts/study3_vlm.py --config configs/experiment.yaml --stage analyse
Scores are appended per image, so an interrupted run resumes where it stopped.
Outputs: results/study3/scores_zeroshot.csv, scores_fewshot.csv, summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fewgaze.config import load_config, output_dir
from fewgaze.dataset import load_manifest, manifest_fingerprint
from fewgaze.extraction import cache_keys, crop_dir
from fewgaze.labels import derive_labels, labelled_view
from fewgaze.splits import make_folds, sample_budget
from fewgaze.utils import get_logger

log = get_logger("fewgaze.study3")
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QUESTION = "Is this person looking directly at the camera? Answer with Yes or No."
PIXELS = 448 * 448
NEG_PER_POS = 4
SUBSET_SEED = 20261108
CANONICAL_RUN = "20261023-005930-20d2e4"


class VLM:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, quantization_config=q, device_map="cuda:0", torch_dtype=torch.float16
        ).eval()
        self.proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=PIXELS, max_pixels=PIXELS)
        tok = self.proc.tokenizer
        self.yes = tok.encode("Yes", add_special_tokens=False)[0]
        self.no = tok.encode("No", add_special_tokens=False)[0]

    def score(self, image, examples=()):
        """logit('Yes') - logit('No') at the first answer position. examples: [(PIL image, 'Yes'|'No'), ...]."""
        msgs, images = [], []
        for im, ans in examples:
            msgs += [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]},
                {"role": "assistant", "content": [{"type": "text", "text": ans}]},
            ]
            images.append(im)
        msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]})
        images.append(image)
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=images, return_tensors="pt").to("cuda:0")
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        return float(logits[self.yes] - logits[self.no])


def load_image(cdir: Path, sid: str):
    from PIL import Image

    return Image.open(cdir / f"{sid}.jpg").convert("RGB")


def fewshot_plan(data: pd.DataFrame, folds) -> pd.DataFrame:
    rows = []
    for f in folds:
        test = data[data["sample_id"].isin(f.test_ids)]
        pos = sorted(test.loc[test["label"] == 1, "sample_id"])
        neg = np.array(sorted(test.loc[test["label"] == 0, "sample_id"]))
        rng = np.random.default_rng([SUBSET_SEED, f.index])
        pick = rng.choice(len(neg), size=min(len(neg), NEG_PER_POS * len(pos)), replace=False)
        train = data[data["sample_id"].isin(f.train_ids)]
        shots = sample_budget(train, 2, seed=0, fold_index=f.index)
        lab = dict(zip(train["sample_id"], train["label"]))
        p = [s for s in shots if lab[s] == 1]
        n = [s for s in shots if lab[s] == 0]
        order = [p[0], n[0], p[1], n[1]]
        for sid in pos + sorted(neg[pick].tolist()):
            rows.append({"fold": f.index, "sample_id": sid, "shots": " ".join(order)})
    return pd.DataFrame(rows)


def run_stage(stage: str, cfg, data, folds, out: Path) -> None:
    keys = cache_keys(cfg, manifest_fingerprint(load_manifest(cfg).records))
    cdir = crop_dir(cfg, keys)
    path = out / f"scores_{stage}.csv"
    done = set(pd.read_csv(path)["sample_id"]) if path.exists() else set()
    if stage == "zeroshot":
        todo = pd.DataFrame({"sample_id": sorted(set(data["sample_id"]) - done)})
        todo["shots"] = ""
    else:
        plan = fewshot_plan(data, folds)
        plan.to_csv(out / "fewshot_plan.csv", index=False)
        todo = plan[~plan["sample_id"].isin(done)]
    log.info("%s: %d images to score (%d done)", stage, len(todo), len(done))
    if todo.empty:
        return
    vlm = VLM()
    lab = dict(zip(data["sample_id"], data["label"]))
    t0, buf = time.perf_counter(), []
    for i, r in enumerate(todo.itertuples(index=False), 1):
        shots = [(load_image(cdir, s), "Yes" if lab[s] == 1 else "No") for s in r.shots.split()] if r.shots else []
        buf.append({"sample_id": r.sample_id, "score": vlm.score(load_image(cdir, r.sample_id), shots)})
        if i % 50 == 0 or i == len(todo):
            pd.DataFrame(buf).to_csv(path, mode="a", header=not path.exists(), index=False)
            buf = []
            el = time.perf_counter() - t0
            log.info("%s %d/%d  %.2f s/img  eta %.0f min", stage, i, len(todo), el / i, el / i * (len(todo) - i) / 60)


def analyse(cfg, data, folds, out: Path) -> dict:
    from fewgaze.metrics import auroc, balanced_accuracy
    from fewgaze.statistics import FoldBlock, bootstrap, metric_fn, percentile_ci

    n_boot, seed = int(cfg["statistics"]["n_bootstrap"]), int(cfg["statistics"]["bootstrap_seed"])
    zs = pd.read_csv(out / "scores_zeroshot.csv").drop_duplicates("sample_id").set_index("sample_id")["score"]
    pred = pd.read_csv(output_dir(cfg) / "runs" / CANONICAL_RUN / "predictions.csv.gz", low_memory=False)
    pred["budget"] = pred["budget"].astype(str)
    ref = {
        "clip_zeroshot": pred[(pred.method == "clip_zeroshot") & (pred.condition == "clean")],
        "l2cs_zeroshot": pred[(pred.method == "l2cs_zeroshot") & (pred.condition == "clean")],
        "geometry_k5_seed0": pred[(pred.method == "geometry") & (pred.condition == "clean") & (pred.budget == "5") & (pred.seed == 0)],
    }
    lab = data.set_index("sample_id")["label"]
    subj = data.set_index("sample_id")["subject"]
    res: dict = {"model": MODEL_ID, "quantisation": "4-bit NF4, fp16 compute"}

    def blocks_for(ids_per_fold, series):
        bl = []
        for ids in ids_per_fold:
            ids = [i for i in ids if all(i in s.index for s in series.values())]
            bl.append(
                FoldBlock(subj.loc[ids].to_numpy(), lab.loc[ids].to_numpy(), {k: s.loc[ids].to_numpy()[None] for k, s in series.items()})
            )
        return bl

    def summarise(blocks, names, tag):
        fn = metric_fn("auroc")
        dist = bootstrap(blocks, names, fn, n_boot, seed)
        o = {}
        for m in names:
            per_fold = [auroc(b.labels, b.scores[m][0]) for b in blocks]
            o[m] = {"auroc": float(np.mean(per_fold)), "ci95": percentile_ci(dist[m]), "per_fold": per_fold}
        for a in names[1:]:
            d = dist[names[0]] - dist[a]
            o[f"{names[0]}_minus_{a}"] = {
                "delta": o[names[0]]["auroc"] - o[a]["auroc"],
                "ci95": percentile_ci(d),
                "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            }
        res[tag] = o

    ref_s = {k: v.drop_duplicates("sample_id").set_index("sample_id")["score"] for k, v in ref.items()}
    fold_ids = [list(f.test_ids) for f in folds]
    summarise(
        blocks_for(fold_ids, {"vlm_zeroshot": zs, "clip_zeroshot": ref_s["clip_zeroshot"], "l2cs_zeroshot": ref_s["l2cs_zeroshot"]}),
        ["vlm_zeroshot", "clip_zeroshot", "l2cs_zeroshot"],
        "zeroshot_all",
    )
    ba = [balanced_accuracy(lab.loc[ids].to_numpy(), zs.loc[ids].to_numpy() > 0) for ids in fold_ids]
    res["zeroshot_all"]["vlm_zeroshot"]["balanced_accuracy"] = float(np.mean(ba))
    res["zeroshot_all"]["vlm_zeroshot"]["says_yes_rate"] = float((zs > 0).mean())
    fs_path = out / "scores_fewshot.csv"
    if fs_path.exists():
        fs = pd.read_csv(fs_path).drop_duplicates("sample_id").set_index("sample_id")["score"]
        plan = pd.read_csv(out / "fewshot_plan.csv")
        sub_ids = [plan.loc[plan.fold == f.index, "sample_id"].tolist() for f in folds]
        summarise(
            blocks_for(sub_ids, {"vlm_fewshot_k2": fs, "vlm_zeroshot": zs, "geometry_k5_seed0": ref_s["geometry_k5_seed0"]}),
            ["vlm_fewshot_k2", "vlm_zeroshot", "geometry_k5_seed0"],
            "subset",
        )
        res["subset"]["n_images"] = int(sum(len(i) for i in sub_ids))
        res["subset"]["vlm_fewshot_k2"]["says_yes_rate"] = float((fs > 0).mean())
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["zeroshot", "fewshot", "analyse"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="score only this many images (timing test; not saved)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    man = load_manifest(cfg)
    data = labelled_view(man.records, derive_labels(man.records, cfg["labels"]))
    folds = make_folds(data, int(cfg["splits"]["n_folds"]), int(cfg["splits"]["seed"]))
    out = output_dir(cfg) / "study3"
    out.mkdir(parents=True, exist_ok=True)
    if args.limit:
        keys = cache_keys(cfg, manifest_fingerprint(man.records))
        cdir = crop_dir(cfg, keys)
        vlm = VLM()
        plan = fewshot_plan(data, folds).head(args.limit)
        lab = dict(zip(data["sample_id"], data["label"]))
        for r in plan.itertuples(index=False):
            t0 = time.perf_counter()
            z = vlm.score(load_image(cdir, r.sample_id))
            t1 = time.perf_counter()
            f = vlm.score(load_image(cdir, r.sample_id), [(load_image(cdir, s), "Yes" if lab[s] else "No") for s in r.shots.split()])
            log.info(
                "%s label=%d zs=%.2f (%.2f s) fs=%.2f (%.2f s)", r.sample_id, lab[r.sample_id], z, t1 - t0, f, time.perf_counter() - t1
            )
        return 0
    if args.stage == "analyse":
        print(json.dumps(analyse(cfg, data, folds, out), indent=1))
    else:
        run_stage(args.stage, cfg, data, folds, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

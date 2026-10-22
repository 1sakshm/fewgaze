# FewGaze

**How little labelled data does eye-contact detection really need?**

FewGaze is a reproducible study of how classifiers for **camera-directed gaze** improve as the number of
labelled training examples grows, evaluated on **people the model has never seen**. It compares four kinds
of frozen representation: interpretable facial and iris geometry (MediaPipe), generic self-supervised features
(DINOv2), a pretrained gaze estimator (L2CS-Net) and a vision-language model (CLIP). All four use
identical subject-disjoint folds, identical training samples and subject-level confidence intervals. The
analysis plan was committed to git before the first experiment ran.

📄 Paper: [`report/main.pdf`](report/main.pdf) · 🌐 Project page: [`site/`](site/) · 📐 Protocol: [`docs/methodology.md`](docs/methodology.md)

![Pipeline](results/figures/fig1_pipeline.png)

## Research question

> How does the number of labelled examples per class affect held-out-subject performance of different
> computer-vision representations for identifying images in which a person looks toward the camera?

**What is classified.** In the Columbia Gaze Data Set (Smith et al., UIST 2013), each subject's head rests in a
chin rest while a camera at five positions records them looking at targets defined **relative to that camera**.
*Positive* means a gaze offset of (0°, 0°), the paper's "gaze locking"; *negative* covers the other 20 offsets, the
nearest of which is 5° to the side. This is a controlled proxy for eye contact with a camera, **not
interpersonal eye contact**.

## Key findings

Verified results from run `20261023-005930-20d2e4` (clean commit `3fbfc55`; it reproduces the earlier run `20261023-000805-20d2e4` bit-for-bit). Values are mean AUROC over 5 subject folds × 5 seeds,
with 95 % subject-bootstrap CIs. Sources: `results/tables/learning_curves.csv` and `paired_differences.csv`.

1. **Simple geometry was the most label-efficient method at every budget.** Nine MediaPipe eye and head
   features with a degree-2 logistic regression reached AUROC 0.537 [0.529, 0.545] at 5 labels per class,
   0.601 [0.592, 0.613] at 25, 0.730 [0.715, 0.751] at 100 and 0.810 [0.791, 0.833] with all training data.
2. **A linear probe on frozen DINOv2-S stayed at chance** (0.499–0.513) even with about 4,700 training images.
   A post-hoc diagnostic shows that its embeddings decode *head pose* from unseen people (R² = 0.95) but
   not gaze direction (R² < 0). An eye-region crop helped only a little (0.562 at ALL, post-hoc).
3. **The pretrained gaze estimator mainly helped with zero labels.** L2CS-Net's angle-to-camera score
   gave 0.595 [0.566, 0.630] without any FewGaze label, better than every method trained on ≤ 10 labels
   per class. A head trained on its two angles matched but never improved on this (0.596 at ALL). Its
   predictions are biased on this dataset, and per-subject AUROC ranged from 0.32 to 0.94.
4. **The 5° boundary is unsolved.** At k = 25, geometry predicted "camera" for 71.8 % of true positives
   but also for 67.7 % of the 5°-off negatives. Its discrimination comes from the larger offsets.
5. **Head pose matters.** Geometry's AUROC (ALL) fell from 0.891 at a frontal pose to 0.718–0.766 at ±30°.
6. **Label noise** reduced geometry's AUROC roughly in proportion to the flip rate (k = 100: 0.730 →
   0.695 / 0.656 / 0.602 at 10 / 20 / 30 % flips). CLIP zero-shot was weak (0.547).

![Learning curves](results/figures/fig3_learning_curves_auroc.png)

## Live demo

The project page ([`site/`](site/)) includes an in-browser demo of the best method. A visitor picks a face photo,
and MediaPipe plus the exported geometry classifier return P(looking at the camera), the measured eye
landmarks and warnings when the photo is outside the training conditions. The photo never leaves the browser.
The browser pipeline matches the Python pipeline to within 4 × 10⁻⁴ in probability, with 0 of 24 decisions
differing ([`site/demo/README.md`](site/demo/README.md)). The model was trained on 56 people in one lab, so it is
a research demo and not an eye-contact detector.

## Methods

| Method | Representation (frozen) | Trained head | Prior supervision |
|---|---|---|---|
| **Geometry** | MediaPipe Face Landmarker → 9 features: iris offset along/across each eye, lid aperture, head rotation | standardise → degree-2 polynomial → logistic regression | landmark model only, no gaze labels |
| **DINOv2-S** | `facebook/dinov2-small`, CLS token of the 224² face crop | linear probe | self-supervised (LVD-142M) |
| **L2CS-Net** | ResNet-50, Gaze360 weights → (pitch, yaw); also a 0-label "angle to camera" score | degree-2 polynomial → logistic regression | **supervised gaze labels (Gaze360)** |
| **CLIP** (exploratory) | ViT-B/32 with two fixed prompts | none (zero-shot) | 400 M image–text pairs |
| Ablations | geometry without the polynomial; DINOv2 mean-patch; L2CS 2048-d features; CLIP linear probe | linear probe | |

The logistic regression is fixed in advance (C = 1, balanced class weights, threshold 0.5) and never tuned. Every
learned transform is fit on the training selection only.

## Protocol in one paragraph

The 56 subjects are dealt into **5 folds by identity** (automated tests check that no person appears in both
train and test). In each fold, for each of 5 seeds, **k ∈ {5, 10, 25, 50, 100, ALL} examples per class** are
drawn from the training subjects. Budgets are nested, and **every method gets the same examples**. Metrics are
AUROC (primary) and balanced accuracy on all held-out images. The 95 % CIs come from a fold-stratified
**subject** cluster bootstrap with 1,000 replicates, using the same resamples for every method so that
differences are paired. Label-noise conditions flip exactly 10 / 20 / 30 % of each class's training labels;
test labels stay clean. Full details, including every deviation and the two post-hoc analyses: [`docs/methodology.md`](docs/methodology.md).

## Dataset and licensing

The Columbia Gaze Data Set is **not included and must not be redistributed**. Its terms are non-commercial
use, with a citation of Smith et al. 2013. See [`data/README.md`](data/README.md) for how to obtain it. The audit
([`results/audit/audit_summary.md`](results/audit/audit_summary.md)) found 5,880 readable images of 56 subjects in a
complete grid, plus **one byte-identical pair with conflicting gaze annotations**, which was excluded. No
dataset images appear in this repository, the paper or the site.

## Installation

Tested with Python 3.11.17 on Windows 11 with an RTX 3050 (CUDA 12.1). Linux and WSL use the same commands. The
core package (splits, training, metrics, statistics, plots, tests) needs only the lightweight dependencies.

```bash
conda create -n fewgaze python=3.11 -y && conda activate fewgaze     # or python -m venv .venv
# GPU (optional; CPU works, only slower for DINOv2/L2CS/CLIP extraction):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .            # optional extras: pip install -e ".[geometry,models,dev]"
pytest                      # 42 tests on synthetic data; no dataset or weights needed
```

L2CS-Net weights: the upstream Google Drive folder returned 404 on 2026-10-22. FewGaze uses the same Gaze360
checkpoint re-hosted as safetensors by Py-Feat:

```bash
mkdir -p cache/weights
curl -L -o cache/weights/l2cs_gaze360_resnet50.safetensors \
  https://huggingface.co/py-feat/l2cs/resolve/main/l2cs_gaze360_resnet50.safetensors
```

The MediaPipe, DINOv2 and CLIP models download automatically on first use.

## Reproducing the experiments

```bash
python scripts/audit_dataset.py          --config configs/experiment.yaml   # ~1 min
python scripts/extract_features.py       --config configs/experiment.yaml   # ~20 min (12 min CPU geometry + GPU models)
python scripts/run_experiments.py        --config configs/experiment.yaml   # ~10 min incl. 1,000-replicate bootstrap
python scripts/analyze_errors.py         --config configs/experiment.yaml
python scripts/generate_report_assets.py --config configs/experiment.yaml   # figures, LaTeX tables, site data
python scripts/export_demo_model.py                                          # demo model (site/demo/model.json)
cd report && latexmk -pdf main.tex                                          # paper

# post-hoc analyses (P1, P2; clearly separated in results/posthoc/)
python scripts/extract_features.py --config configs/posthoc_eyes.yaml --stages eyes
python scripts/run_experiments.py  --config configs/posthoc_eyes.yaml --no-noise
python scripts/posthoc_analyses.py
```

Every setting lives in [`configs/experiment.yaml`](configs/experiment.yaml). Feature caches are keyed by
model, preprocessing and dataset fingerprint, can resume after an interruption, and are rejected if stale. Each run
writes `results/runs/<run_id>/` with per-run metrics (`results.jsonl`), per-image scores, the exact training-sample
IDs and flip masks, eligibility, and provenance (config, git commit, dataset fingerprint, library versions). The
schema is in [`results/README.md`](results/README.md).

**Exit codes:** 0 ok · 2 dataset missing or unparseable · 3 audit found blocking problems / no results yet ·
4 an extraction stage failed · 5 no features cached.

## Results and figures

| Figure | File | Script |
|---|---|---|
| 1 Pipeline | `results/figures/fig1_pipeline.png` | `generate_report_assets.py` |
| 2 Dataset composition | `fig2_dataset_composition.png` | `audit_dataset.py` |
| 3 / 4 AUROC and balanced-accuracy learning curves | `fig3_*`, `fig4_*` | `generate_report_assets.py` |
| 5 Label noise | `fig5_label_noise.png` | 〃 |
| 6 Head pose | `fig6_head_pose.png` | 〃 |
| 7 Confusion matrices (k = 25) | `fig7_confusion.png` | 〃 |
| 8 Efficiency | `fig8_efficiency.png` | 〃 |
| 9 Error examples | `figures/local_only/` (**never committed**: dataset images) | `analyze_errors.py` |
| S1–S3 Ablations, per-gaze-cell prediction rates, per-subject spread | `figS*` | `generate_report_assets.py` |
| P1–P2 Post-hoc diagnostics | `results/posthoc/figures/` | `posthoc_analyses.py` |

Result categories: everything under `results/tables` and `results/figures` is a **verified experimental result**
from the run named in `results/runs/LATEST`. Everything under `results/posthoc/` is **post-hoc exploratory**.
Synthetic numbers appear only in `tests/`.

## Repository structure

```text
configs/        experiment.yaml (primary), posthoc_eyes.yaml (P2)
src/fewgaze/    dataset, labels, splits, geometry, embeddings, gaze_estimator, extraction, features (cache),
                train, evaluate, metrics, statistics (subject bootstrap), summary, errors, plots, utils
scripts/        audit, extract, run, analyze errors, report assets, post-hoc analyses
tests/          42 tests: leakage, labels, sampling, noise isolation, metrics vs sklearn, closed-form bootstrap,
                cache invalidation/resume, result schema, determinism
results/        audit/, runs/, tables/, figures/, posthoc/
report/         main.tex, references.bib (DOI-verified), generated/ tables, figures/
site/           static project page rendered from site/data/results.json
docs/           methodology.md (pre-registered plan + deviations), dataset_card.md, model_card.md
```

## Limitations

* One controlled dataset: 56 people, one lab, 2 m, yaw-only head pose, a chin rest. Camera-directed gaze ≠ eye contact.
* L2CS-Net was trained with gaze labels, so its *k* undercounts its supervision. The comparison is between off-the-shelf systems.
* Heads and hyperparameters were fixed in advance. Non-linear probes on embeddings could change their ranking.
* CIs capture subject-sampling uncertainty with models held fixed. The SD across runs is reported separately.
* No demographic labels are available, so no fairness analysis was done. Not for surveillance or attention monitoring.

## References

Key sources: Smith et al. 2013 (Columbia Gaze, gaze locking); Müller et al. 2018; Zhang et al. 2017; Oquab et al. 2024
(DINOv2); Abdelrahman et al. 2023 (L2CS-Net); Kellnhofer et al. 2019 (Gaze360); Radford et al. 2021 (CLIP);
Lugaresi et al. 2019 / Kartynnik et al. 2019 / Ablavatski et al. 2020 (MediaPipe). Full, DOI-verified BibTeX is in
[`report/references.bib`](report/references.bib).

## Citation

See [`CITATION.cff`](CITATION.cff). If you use the dataset, cite Smith et al. (UIST 2013).

## Licence

Code: MIT ([`LICENSE`](LICENSE)). It does **not** cover the dataset, derived crops and features, or third-party model weights.

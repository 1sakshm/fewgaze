# FewGaze methodology

This document fixes the analysis plan. It was committed to git **before the first experiment was run**.
Check the commit history: the commit containing this file predates every file in `results/runs/`.
Any later change to the protocol is listed under *Deviations* at the end, with a date and a reason.

## 1. Question and target

**Question.** How does the number of labelled training examples per class, *k*, affect how well
different frozen representations identify camera-directed gaze in images of **people not seen during training**?

**Target (operational definition).** *Camera-directed gaze under the Columbia Gaze capture protocol.*
Smith et al. (UIST 2013, §"Collection Procedure") moved the camera to five positions while the subject's
head stayed fixed in a chin rest. Each camera position had its own 7 × 3 grid of gaze targets, so the
gaze angles (H, V) are **relative to the camera**, and (H, V) = (0°, 0°) is the paper's *gaze-locking*
condition at every head pose. The paper reports "280 gaze locking images and 5,600 non-gaze-locking images".

* positive: V = 0° and H = 0° (all head poses)
* negative: every other gaze condition (H ∈ {0, ±5, ±10, ±15}°, V ∈ {0, ±10}°)
* excluded: the two byte-identical files with conflicting annotations found by the audit (§3)

This is a **controlled proxy**. The dataset contains no interpersonal eye-contact annotations. Subjects
looked at dots or at a camera lens in a lab, so results say nothing directly about eye contact in
conversation. The paper defines no sign conventions for H, V or P. They do not affect the binary label,
which depends only on whether H = V = 0.

The hardest negatives are V = 0°, |H| = 5° (560 images). They are kept as negatives, because the paper's
own label definition does so. They are analysed separately.

## 2. Data split

* **Outer folds.** 5 folds by subject. The 56 subjects are sorted, permuted with seed 20261108, and
  dealt round-robin, giving 11 or 12 test subjects per fold. Every subject contributes 5 positives and
  100 negatives (98 for subject 0007 after the exclusion), so every test fold contains both classes.
  Tests in `tests/test_splits.py` verify subject-disjointness, full coverage, both classes and determinism.
* **The same folds are used for every method.**

## 3. Eligibility

Every method uses the MediaPipe Face Landmarker crop: geometry features come from its landmarks, and
the appearance models consume its 448 × 448 face crop (landmark bounding box + 25 % margin). A sample is
*commonly eligible* if every enabled method produced finite features for it. **All primary comparisons
use the commonly eligible set.** The count of excluded samples per method is written to
`results/runs/<id>/eligibility.csv` and reported.

## 4. Label budgets and selections

* *k* ∈ {5, 10, 25, 50, 100, ALL} labelled examples **per class**. ALL means every commonly eligible
  training-subject sample: about 225 positives and about 4,480 negatives, so it is imbalanced.
* For each (fold, seed ∈ {0..4}, class), one permutation of that class's eligible training samples is
  drawn, and the first *k* are used. Budgets are therefore **nested**: the *k* = 5 set ⊂ the *k* = 10 set ⊂ …
* **One selection is shared by every trained method.** Selected IDs are stored in `selections.json.gz`.
* A budget a fold cannot supply is recorded as `unavailable`, never oversampled. With 44–45 training
  subjects × 5 positives, *k* = 100 is always available.

## 5. Methods (all backbones frozen)

| Method | Representation | Downstream |
|---|---|---|
| **Geometry** (primary) | 9 features from MediaPipe: per eye, the iris-centre offset along and across the eye-corner axis (÷ eye width) and the lid aperture (÷ eye width); head rotation (3 Euler angles from the face transformation matrix) | standardise → degree-2 polynomial → standardise → LR |
| Geometry-linear (ablation) | same 9 features | standardise → LR |
| **DINOv2-S** (primary) | `facebook/dinov2-small`, crop resized to 224², ImageNet normalisation, layer-normed CLS (`pooler_output`), 384-d | standardise → LR |
| DINOv2-S mean-patch (ablation) | mean of the final patch tokens, 384-d | standardise → LR |
| **L2CS-Net angles** (primary) | ResNet-50 L2CS-Net, Gaze360 weights, 448² crop; (pitch, yaw) in degrees from the expectation over 90 bins | standardise → degree-2 polynomial → standardise → LR |
| L2CS-Net features (ablation) | 2048-d pooled ResNet features | standardise → LR |
| **L2CS-Net zero-shot** (primary, 0 labels) | score = −(angle between predicted gaze and the camera axis) | none; AUROC only (no threshold without labels) |
| CLIP probe (ablation) | `openai/clip-vit-base-patch32` image embedding (L2-normalised, 512-d) | standardise → LR |
| **CLIP zero-shot** (primary, 0 labels) | softmax over two fixed prompts, P("…looking directly toward the camera.") | none; decision = argmax of the two prompts |

**Why the polynomial for low-dimensional gaze features?** The positive class is a small region at the
centre of gaze space, with negatives on every side. No linear function of a monotone gaze proxy can
separate a centre from both sides. A degree-2 expansion is the smallest change that can represent
"close to zero". This choice was fixed before seeing results. The linear geometry ablation tests it.

**Classifier.** scikit-learn `LogisticRegression` with C = 1.0, L2, `class_weight="balanced"` and
lbfgs (max_iter 5000). **No hyperparameter tuning.** Every learned transform sits inside one `Pipeline`
fit on the training selection only, so the test subjects never influence scaling.

**Decision threshold.** 0.5 on the balanced-weight probability, prespecified. Balanced class weights
make 0.5 the natural operating point for both balanced and imbalanced (ALL) training sets. The
threshold is never chosen with test labels.

**Pretrained knowledge.** DINOv2 (LVD-142M), CLIP (400M image–text pairs) and especially L2CS-Net
(trained with gaze labels on Gaze360) bring very different prior supervision. *k* counts only the
FewGaze labels. The comparison is between *off-the-shelf systems under the same downstream label
budget*, not between architectures.

## 6. Metrics

Computed on the held-out subjects of each fold:

* **AUROC** (primary, threshold-free; Mann–Whitney with ties counted as ½; NaN if a class is absent)
* **Balanced accuracy** = (sensitivity + specificity) / 2 at the prespecified threshold
* Sensitivity, specificity, and confusion counts

**Point estimate:** the mean of the per-(fold, seed) metric. Zero-shot methods have one score per fold.

**95 % CI:** a fold-stratified **subject cluster bootstrap** with 1,000 replicates. In each replicate the
test subjects of each fold are resampled with replacement, together with all their images, and the
metric is recomputed for every seed and averaged. **The same resamples are used for every method,** so
differences are paired.

*What the interval covers:* uncertainty from which subjects were tested, with the trained models held
fixed. *What it does not cover:* variability from retraining on other training subjects. The SD across
the 25 fold × seed runs is reported separately for that.

**Prespecified paired comparisons** (ΔAUROC with paired-bootstrap 95 % CI, at every budget):
DINOv2 − Geometry, L2CS − Geometry, L2CS − DINOv2. No p-values are computed. A CI that excludes 0 is
described as *a difference beyond subject-sampling uncertainty*, not as a significant result, and other
comparisons are descriptive only.

## 7. Label-noise experiment (simulated)

For *k* ∈ {10, 25, 50, 100} and rate ∈ {0.1, 0.2, 0.3}: exactly round(rate · k) labels **in each class**
of the training selection are flipped. This is class-balanced symmetric noise, and it keeps the
training class ratio. The flip mask depends on (seed, fold, k, rate) and is shared by all methods.
Flipped IDs are stored. Test labels are never touched; `tests/test_reproducibility.py` checks this.

ALL is excluded from this experiment. With 1 : 20 imbalance, flipping 10 % of negatives would create
about twice as many false positives as there are true positives, and the experiment would measure the
imbalance rather than robustness. This is a simulation, not an estimate of real annotation error in
Columbia Gaze.

## 8. Subgroup and error analysis

* AUROC per head pose (−30 … +30°), at *k* = 25 and ALL, with bootstrap CIs and image counts. The
  bootstrap uses 500 replicates. The positive count per pose is 56.
* Predicted-positive rate per (V, H) gaze cell, at *k* = 25.
* Per-subject AUROC (5 positives per subject, so this is coarse).
* Example selection, prespecified: for each trained primary method, the most confident false positives
  (highest mean score) and the most confident false negatives (lowest), at *k* = 25.
* Metadata categories only (near-camera negative, vertical-offset-only, extreme pose, other). These
  describe an image's condition, not a verified cause.

## 9. Reporting rules

* Every number in the report, README and site is read from `results/tables/*` files generated by
  the scripts.
* Failed or unavailable runs are kept in `results.jsonl` with their status.
* Dataset images and crops are never committed or published.

## Deviations from this plan

All deviations below were made **before the main experiment run** (`results/runs/LATEST`). None changes
which comparisons are made or how a metric is defined.

1. **2026-10-22, L2CS-Net weight source.** The upstream Google Drive folder returned HTTP 404. We
   used the same Gaze360 checkpoint re-hosted as safetensors by Py-Feat (`py-feat/l2cs`). The checkpoint
   loads into the re-declared official architecture with strict key matching (324 tensors).
2. **2026-10-23, bootstrap implementation.** Subject resamples are now drawn as count vectors, and the
   metric is evaluated in closed form (AUROC = cᵀAc / ((c·P)(c·N)); balanced accuracy from per-subject
   TP/TN counts). This is mathematically identical to explicit resampling, which
   `tests/test_metrics.py::test_closed_form_bootstrap_equals_explicit_resampling` checks to 1e-12. The
   only reason for the change was speed (about 25× faster), so the prespecified 1,000 replicates are
   feasible. Because the random draws are consumed differently, CI endpoints differ from a naive
   implementation by Monte Carlo error only.
3. **Observation recorded before any classifier result was seen** (from inspecting the cached L2CS
   outputs): on Columbia, mean L2CS yaw falls monotonically with H, and pitch rises with V. At the
   gaze-locked condition, however, the mean prediction is about (pitch −7°, yaw −6°) rather than (0, 0),
   and mean yaw also shifts with head pose. The zero-shot "angle to camera" score was **not** changed in
   response, because it is prespecified. This observation is reported to help interpret its result.

## Post-hoc exploratory analyses (added after seeing the main results)

These were **not** in the original plan. They are reported separately, labelled *post-hoc*, and never
replace a prespecified result. Each was written down here before it was run.

* **P1, 2026-10-23: linear decoding diagnostic.** After seeing the DINOv2 linear probe at chance, we
  checked for an implementation error by measuring how well each cached representation linearly decodes
  the continuous annotations H, V and head pose on unseen subjects. We used a ridge regression with
  5-fold GroupKFold by subject; the ridge α was chosen by RidgeCV's internal leave-one-out, which can only
  bias R² upward. Script: `scripts/posthoc_analyses.py`.
* **P2, 2026-10-23: DINOv2 on an eye-region crop.** This asks whether the DINOv2 result reflects the
  representation or the face-scale crop, in which each eye covers only a few 14-px patches. MediaPipe is
  re-run on the cached 448² face crop, and a strip around both eyes is cut out: centred on the four eye
  corners, width 1.5 × the outer-corner distance, aspect 8 : 3. The strip is resized to 336 × 126
  (24 × 9 patches) and passed through the same frozen `facebook/dinov2-small`, using the CLS token
  with a linear probe exactly as in the primary DINOv2 method. It is evaluated with the identical
  folds, selections and statistics, in a separate run alongside Geometry and DINOv2 (face) for paired
  comparison. Outputs: `results/posthoc/`.

## Reproducibility check (2026-10-23)

The primary run `20261023-000805-20d2e4` was made from a working tree with uncommitted documentation and
bootstrap changes (its provenance records `d26428d…-dirty`). After everything was committed, the protocol was
re-run from the clean commit `3fbfc55` as run `20261023-005930-20d2e4`. All 3,160 per-run records and all 255
summary rows (estimates, CI bounds, SDs) were **identical** (max |difference| = 0.0). `results/runs/LATEST`
points to the clean run, and every reported number comes from it.

## Study 2 (pre-registered 2026-10-23, before any extraction): where does gaze live inside foundation models?

**Motivation.** Post-hoc analysis P1 found that the final DINOv2-S CLS token decodes head pose (R² 0.95)
but not gaze (R² < 0). Study 2 asks whether gaze information exists **anywhere** in frozen vision foundation
models: in which layer, and in which tokens.

**Models (frozen, Hugging Face weights):** `facebook/dinov2-small`, `facebook/dinov2-base`,
`openai/clip-vit-base-patch16` (vision tower), `google/siglip-base-patch16-224` (vision tower),
`facebook/vit-mae-base` (with `mask_ratio = 0` and an identity patch order, so all tokens are kept in
place). Input: the same 448² MediaPipe face crop, resized to 224² with each model's own normalisation
constants. DINOv2 gives a 16×16 grid with 14 px patches; the others give 14×14 grids with 16 px patches.

**Representations per layer** (the embedding output plus every transformer block; 13 layers):
* *global*: the CLS token, or, for SigLIP, which has no CLS token, the mean of all patch tokens;
* *eye*: the mean of the patch tokens whose centres fall inside each eye's box (right and left eyes
  concatenated). Eye boxes come from MediaPipe landmarks re-detected on the face crop: the eye corners and
  lids, padded by 25 % of eye width. If no patch centre falls inside a box, the nearest patch is used.

**Probes.** All use the five subject-disjoint FewGaze folds; every learned step is fit on training subjects only.
1. *Decoding (primary):* a ridge regression (standardised features; α from RidgeCV's grid 10⁻²…10⁵) predicts
   H, V and head pose P. We report the pooled held-out-subject R² for each model × layer × representation.
   α selection uses within-training leave-one-out, which ignores subject grouping and can only make R²
   optimistic.
2. *Eye-contact readout (secondary):* (a) a linear logistic probe (C = 1, balanced weights) on all training
   data, scored by held-out AUROC averaged over folds; (b) a geometric readout that uses the ridge
   predictions Ĥ, V̂ (trained with the continuous gaze annotations, which is **more supervision** than the
   binary labels of Study 1, stated as such): score = −[(Ĥ/σ_H)² + (V̂/σ_V)²], with σ estimated on
   training subjects.

**Decisions fixed now.** No layer or model is selected on test folds. All 5 × 13 × 2 cells are reported.
Uncertainty: SD across the five folds. Comparisons with Study 1 are descriptive. Outputs: `results/study2/`.

### Study 2: correction and added sensitivity analysis (2026-10-23, after the first probe layers)

* **Correction of a statement in the pre-registration (procedure unchanged).** The plan said that choosing
  the ridge α by RidgeCV's leave-one-*image*-out "can only make R² optimistic". That is wrong. Leave-one-image-out
  keeps other images of the same person in the inner training set, so it can favour weak regularisation that fits
  person-specific structure. On held-out people that structure does not transfer, and R² can come out
  **pessimistic**, even below 0. The first probe layers showed exactly this (e.g. DINOv2-S CLS, layer 12:
  R²(H) = −0.15). The same correction applies to post-hoc analysis P1.
* **Sensitivity analysis S2-b (added after observing the above; reported separately).** We re-run the decoding
  probe for the eye-patch representation at every layer of every model, plus the geometry reference, choosing α by
  **3-fold person-grouped inner cross-validation** on the training people. The grid is 10⁰…10⁶, 7 values; the
  solution is computed once per fold by SVD. Outer folds, targets and metric are unchanged. Output:
  `results/study2/decoding_grouped_alpha.csv`. The primary pre-registered table is not replaced. If the two
  analyses disagree, the paper reports both.

### Study 2: post-hoc analysis P3, readout with the layer chosen without test data (added 2026-10-23, after all probes)

The primary Study 2 results report all 13 layers; quoting the best layer of a model is selection on test folds and is
optimistic. This matters most for the readout, where the best CLIP eye-patch layer (0.834) exceeds the geometry
reference (0.813). P3 picks the layer and the ridge α's inside each outer fold, using 3-fold person-grouped inner CV on
the training people. The criterion is inner squared error of H/σ_H plus V/σ_V. P3 then scores the readout once on the
held-out people. Geometry goes through the same procedure. Outputs: `results/study2/nested_layer.csv` and
`nested_layer_summary.csv` (fold-mean AUROC, fold SD, and the per-fold paired difference against geometry). Script:
`scripts/study2_nested_layer.py`. This analysis is post hoc and labelled as such.

# Model card: FewGaze classifiers and the frozen models they use

FewGaze trains **only small logistic-regression heads**. Every backbone is frozen and used exactly as
released. The trained heads are research artefacts for measuring sample efficiency. They are **not
deployable eye-contact detectors**.

## Frozen components

| Component | Version / source | Licence | Prior supervision | Params |
|---|---|---|---|---|
| MediaPipe Face Landmarker | `face_landmarker.task`, float16 v1, from `storage.googleapis.com/mediapipe-models` (mediapipe 0.10.21) | Apache-2.0 (MediaPipe repo) | Google-internal face/iris landmark data | – |
| DINOv2 ViT-S/14 | `facebook/dinov2-small` (Hugging Face) | Apache-2.0 | self-supervised on LVD-142M | 22.06 M |
| L2CS-Net ResNet-50 | official `L2CSNet_gaze360.pkl`, re-hosted as safetensors by Py-Feat (`py-feat/l2cs`) because the upstream Google Drive folder returned HTTP 404 on 2026-10-22 | code MIT; weights trained on Gaze360, whose licence excludes commercial use of trained models | **supervised gaze regression on Gaze360** | 23.88 M (incl. vestigial head) |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | code MIT (openai/CLIP); the HF model card lists no licence field | 400 M image–text pairs | 151.3 M (image + text towers) |

**L2CS-Net integration notes.** The architecture is re-declared in `fewgaze/gaze_estimator.py` to
match the official `l2cs/model.py`, and the checkpoint loads with strict key matching. Decoding follows
the official demo: softmax over 90 bins, then expectation × 4° − 180°. The upstream `Pipeline` labels its
outputs inconsistently: `model.forward` returns (yaw, pitch), but `predict_gaze` unpacks them as (pitch,
yaw). FewGaze names the outputs after the heads they come from. The zero-shot score uses
cos(pitch)·cos(yaw), which is symmetric in the two angles, so it is unaffected. Two documented
deviations from the upstream demo: the face crop comes from MediaPipe instead of RetinaFace, and the crop
is resized directly to 448² instead of 224² → 448².

## Downstream heads

`StandardScaler → [PolynomialFeatures(2) → StandardScaler] → LogisticRegression(C=1, class_weight=balanced)`.
The polynomial step is used only for the 9-d geometry and 2-d L2CS angle features. All hyperparameters
were fixed in advance, and the decision threshold is 0.5.

## Intended use

Research on annotation budgets and on the generalisation of camera-directed-gaze classifiers to unseen
identities in controlled imagery.

## Out-of-scope use

Surveillance; monitoring of attention or engagement of students, employees, drivers or the public; any
high-stakes decision; inferring social behaviour or interpersonal eye contact. The heads were trained on
56 people in one lab, at one distance, with yaw-only head poses. Performance outside those conditions is
unknown.

## Evaluation

See `docs/methodology.md`, `results/tables/learning_curves.csv` and the report. Results are reported
by subject-independent cross-validation with subject-bootstrap confidence intervals.

## Bias and fairness

The dataset has no per-subject demographic labels, so subgroup fairness was **not evaluated**. The
pretrained backbones carry the biases of their training data. Gaze360, for example, has its own
demographic composition. Glasses, which about 21–22 of 56 subjects wear, are a plausible source of
error for iris-based geometry, but no per-subject glasses label is available to test this.

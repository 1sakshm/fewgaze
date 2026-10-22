# FewGaze live demo

The "Demo" section of the project page runs the study's best method, the geometry classifier, on a photo the
viewer chooses. **Everything runs in the browser; no image is uploaded.**

| File | Role |
|---|---|
| `demo.js` | UI: lazy-loads the face model only when a photo is chosen, runs it, draws the eye region, shows the probability, range checks and the nine features |
| `fewgaze-core.js` | Port of `fewgaze.geometry` (eye features, head rotation) and of the exported scikit-learn pipeline |
| `model.json` | Geometry pipeline fit on all 5,878 Columbia images (`scripts/export_demo_model.py`); includes the cross-validated performance of the run it came from and the 1st/99th-percentile feature ranges |
| `vendor/` | MediaPipe tasks-vision 0.10.21 and the same `face_landmarker.task` used in Python (see `vendor/NOTICE.md`) |
| `parity.html` | Browser-vs-Python parity test (needs local reference data, see below) |

## Parity with the Python pipeline (verified 2026-10-23)

`scripts/demo_parity_reference.py` writes 24 Columbia images, resized exactly as in Python, with their Python
features and probabilities to `site/demo/parity/`. This folder is gitignored because the images may not be
redistributed. Serve `site/demo/` over HTTP and open `parity.html`. Results in Edge (headless, CPU delegate):

| Quantity | Max abs. difference |
|---|---|
| Landmark coordinates (normalised) | 5.5 × 10⁻⁶ |
| Facial transformation matrix | 6.0 × 10⁻⁴ |
| Nine features | 2.0 × 10⁻³ |
| P(camera-directed) | 4.3 × 10⁻⁴ |
| Decisions at 0.5 that differ | 0 of 24 |

The test also settled an undocumented detail: in tasks-vision 0.10.21, `Matrix.data` is **column-major**.
Reading it as row-major flipped 13 of 24 decisions. Separately, the exported parameters reproduce
scikit-learn's probabilities on all training images to 3 × 10⁻¹⁶, a check that `export_demo_model.py` runs
before it writes `model.json`.

## Limits shown to the viewer

The model learned from 56 people in one lab, photographed at 2 m with a chin rest and yaw-only head poses. Its
cross-validated AUROC on unseen people is 0.81. Photos outside the training ranges of head angle and eye
measurements get an explicit warning. It is a research demo, not an eye-contact detector, and must not be
used for monitoring.

## Regenerate

```bash
python scripts/export_demo_model.py          # after run_experiments.py; writes site/demo/model.json
python scripts/demo_parity_reference.py      # optional parity check (local only)
```

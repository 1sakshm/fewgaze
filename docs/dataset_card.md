# Dataset card: Columbia Gaze Data Set, as used by FewGaze

## Source

* **Name:** Columbia Gaze Data Set. **Creators:** Brian A. Smith, Qi Yin, Steven K. Feiner, Shree K. Nayar (Columbia University).
* **Reference:** *Gaze Locking: Passive Eye Contact Detection for Human–Object Interaction*, ACM UIST 2013, pp. 271–280, doi:10.1145/2501988.2501994.
* **Access:** e-mail form on <https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/>. Our copy was downloaded on 2026-10-22: `columbia_gaze_data_set.zip`, 2,378,561,401 bytes, which passed `unzip -t`.
* **Terms:** non-commercial use only; the paper must be cited. **FewGaze does not redistribute images, crops or any per-image visual data.**

## Capture protocol

Quoted or paraphrased from Smith et al. 2013, pp. 273–274:

* 56 people (32 male, 24 female, aged 18–36). The paper gives both 21 and 22 as the number wearing glasses. Ethnicities listed on the project page: Asian, White, South Asian, Black, Hispanic/Latino. **No per-subject demographic metadata ships with the data**, so FewGaze cannot do any fairness analysis.
* The subject sat with their head in a chin rest in front of a black background. **The camera moved** between five positions on the floor, 2 m from the subject, which gives head poses of 0°, ±15° and ±30° *relative to the camera*.
* Each camera position had its own 7 × 3 grid of dots: horizontal 0°, ±5°, ±10°, ±15° and vertical 0°, ±10°. The gaze angles are therefore **relative to the camera**. "We collected five gaze locking images (0° vertical and horizontal gaze direction) for each subject, one for each head pose."
* Canon EOS Rebel T3i, 5,184 × 3,456 px. The best of 3–6 shots was kept for each condition.

## What FewGaze verified (`scripts/audit_dataset.py`)

| Check | Result |
|---|---|
| Images matching the filename pattern | 5,880, all readable JPEGs, all 5184×3456 |
| Subjects | 56 (folder names match filename IDs) |
| Grid completeness (56 × 5 poses × 21 gaze) | complete, 0 missing cells |
| Non-image files | `Thumbs.db`, `.DS_Store`, `.picasa.ini` (ignored) |
| Byte-identical files | **1 pair**: `0007_2m_-15P_-10V_0H.jpg` = `0007_2m_-15P_-10V_5H.jpg` |
| Positives (H = V = 0) | 280, 5 per subject |
| Negatives | 5,598 after excluding the duplicate pair |

The duplicate pair means at least one of the two gaze annotations is wrong. Both files are negatives
(V = −10°), so binary labels would not change, but the true condition of either image is unknown.
Both are excluded through `labels.exclude_sample_ids`, a decision made before any modelling.

## Conventions that could not be verified

* **Signs of H, V and P.** The paper does not define them. Our MediaPipe features give *empirical*
  evidence: estimated head yaw tracks P (≈ −15° for P = −15), the iris offset rises monotonically with H,
  and lid aperture is larger for V = +10 (consistent with +V = upward). The binary label does not depend
  on signs.
* **Fixation accuracy.** Subjects were asked to fixate on targets, but individual fixation errors are
  unknown. The paper estimates its own detector's tolerance at about ±2.5° horizontally and ±5°
  vertically, so the ±5° horizontal negatives are the hardest cases and are analysed separately.

## Label derivation (`fewgaze.labels`)

The original annotations (`manifest.csv`) and the derived labels (`labels.csv`: `label_status` ∈
{positive, negative, excluded, unlabeled}) are stored separately. The mapping is configured in
`configs/experiment.yaml → labels`.

## Known limitations for this study

* Controlled lab captures: one background, fixed lighting, chin rest, 2 m, and yaw-only head poses.
* Camera-directed gaze is a proxy. There is no interpersonal or conversational eye-contact annotation.
* With 56 identities, subject-level uncertainty is large.
* A chin rest is visible in every image. It is constant across classes, so it should not leak label
  information, but it is a domain artefact.

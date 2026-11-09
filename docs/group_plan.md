# Natural group conversations: analysis plan

This plan is committed before any MPIIGroupInteraction data has been requested, downloaded or viewed. Every later
change is added to the deviation log at the end, with its date and reason. The paper's results (tag `paper-v1`) are
not changed by this work.

## Questions

1. **Transfer.** How much does the FewGaze geometry detector, trained on the Columbia Gaze lab images and applied
   unchanged, lose when it is used on people in natural group conversations?
2. **Group behaviour.** Over a whole meeting, does the person who talks most also receive the most looks from the
   others, even while they are silent, and does either match who the group rates as its leader?

Müller et al. (ETRA 2018) already showed that, moment to moment, the current speaker is the most likely person to be
looked at. Question 2 is about meeting-level shares and peer-rated leadership, not about that moment-to-moment effect.

## Data

MPIIGroupInteraction (Müller, Huang and Bulling, IUI 2018): 22 discussions of about 20 minutes, 12 with four people
and 10 with three, 78 German-speaking participants who did not know each other. Two cameras are placed behind each
participant, slightly above head height, giving a near-frontal view of every other participant. Annotations used:

* eye contact, labelled every 15 s per participant: the face of another participant (which one) or no face;
* speaking turns for all recordings;
* questionnaires: each participant rated the others on leadership, dominance, competence and liking (7-point scales).

**Licence.** Non-commercial research use under the EULA signed by a supervising faculty member. Anything derived
from the corpus counts as part of it, so only code, configuration and aggregate results are committed: no frames,
crops, landmarks, features or per-frame scores. No image of a participant appears in any figure, page or video.
Publications include the acknowledgement sentence and citations the EULA requires.

## Model (frozen)

* **Primary:** the geometry classifier fitted on all 56 Columbia subjects by `scripts/export_demo_model.py` at tag
  `paper-v1`. MediaPipe features, the degree-2 logistic regression and the 0.5 threshold are used exactly as in the paper.
* **Secondary:** L2CS-Net's zero-label angle-to-camera score, as in the paper.

Nothing is refitted, recalibrated or re-thresholded on MPIIGroupInteraction before T1 and T2 are reported.

## Splits

All MPIIGroupInteraction people are unseen by the frozen models. Any later model trained on this dataset (E1) uses
**meeting-disjoint** folds, so no person and no group appears in both training and test data.

## T1: out-of-the-box accuracy (primary)

* **Unit:** one annotated frame × gazer A × one camera placed behind another participant B.
* **Positive:** the annotators say A looks at B's face. **Negative:** any other label, including "no face".
* **Score:** P(looking at the camera) from the frozen model on A's face in that camera. If MediaPipe finds no face,
  the score is the lowest possible value (counted as a miss, not dropped). AUROC on detected faces only is also reported.
* **Metric:** AUROC with a 95 % CI from 1,000 bootstrap resamples of meetings. It is reported next to the Columbia
  result (0.810 [0.791, 0.833]).
* **Head-pose subset:** the same AUROC restricted to frames where A's head pose toward that camera is within ±30°,
  the Columbia range. This separates the new conditions from the larger head turns.

## T2: who looks at whom (primary)

* **Rule, fixed in advance:** for gazer A, take the camera with the highest P(camera). If that value is below 0.5,
  predict "no face"; otherwise predict the participant that camera sits behind.
* **Metrics:** accuracy and macro-F1 against the annotators, with meeting-bootstrap CIs.
* **Baselines:** (a) always the most frequent label overall; (b) "looks at the current speaker", using the speaking
  annotations, which predicts "no face" when nobody else is speaking.

## Group measures

Per person and meeting:

* **Speaking share:** time annotated as speaking ÷ meeting length.
* **Looks share:** the fraction of all face-directed looks in the meeting that land on this person. The fair share is
  1/3 in four-person meetings and 1/2 in three-person meetings, because nobody looks at themselves.
* **Silent looks share:** the same, counting only looks given while the receiver is not speaking.
* **Perceived leadership:** the mean of the other members' ratings on the leadership sub-scale.

Group analyses use the human eye-contact and speaking annotations, so they do not depend on the detector.

## Hypotheses and tests

The unit of analysis is the meeting. P-values are two-sided, from 10,000 permutations that shuffle values between
people within each meeting.

* **H1 (primary):** speaking share and looks share are positively related within meetings. Statistic: the mean of the
  per-meeting Spearman correlations.
* **H2 (primary):** the top talker is also the person who receives the most looks in more meetings than chance.
  Expected by chance: 12 × 1/4 + 10 × 1/3 ≈ 6.3 of 22. Test: exact test against that Poisson-binomial distribution.
* **H3 (secondary):** perceived leadership is related to speaking share and to looks share. Both mean within-meeting
  correlations are reported, and their difference gets a paired meeting-bootstrap CI.
* **C1 (secondary control):** H1 repeated with the silent looks share. If the link survives, it is not only the
  moment-to-moment habit of looking at whoever is speaking.

With 22 meetings the power is limited, so effect sizes and CIs are reported for every test, including null results.
No correction for multiple tests is applied; all tests listed here are reported, and none is added afterwards
without being marked post hoc.

## A1: fully automatic pipeline

The group measures are recomputed with the frozen detector (T2 rule, frames at 1 fps) and pyannote speaker
diarization, with speakers matched to people by the loudest personal microphone. Reported: the Spearman correlation
between automatic and annotated shares across people, and whether H1 and H2 reach the same conclusion.

## E1: label efficiency in conversations (exploratory, only if time allows)

A who-looks-at-whom classifier on the same features, trained with k labels per class as in Study 1, and a version
that uses speaking turns as free weak labels. Meeting-disjoint folds. Reported as exploratory.

## What would count as failure

* T1 AUROC whose CI includes 0.5: the lab detector does not transfer. This is reported as the result.
* The readme gives no usable camera-to-seat map: T2 and A1 cannot be run as specified, and T1 is limited to cameras
  whose position can be verified. This would be logged as a deviation.
* Leadership ratings are not in the released files: H3 is dropped and the rest is unchanged.

## Deviation log

| Date | Change | Reason |
|---|---|---|
| | | |

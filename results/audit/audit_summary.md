# Dataset audit summary

Generated 2026-10-22T18:14:29Z from `D:\Cactus\fewgaze\data\raw\Columbia Gaze Data Set` (manifest fingerprint `335876e2b2182213`).

| Quantity | Value |
|---|---|
| Images parsed | 5880 |
| Files failing filename parsing | 0 |
| Unreadable images | 0 |
| Subjects | 56 |
| Head poses (deg) | [-30, -15, 0, 15, 30] |
| Vertical gaze (deg) | [-10, 0, 10] |
| Horizontal gaze (deg) | [-15, -10, -5, 0, 5, 10, 15] |
| Capture distance (m) | [2] |
| Image sizes | 5184x3456 (n=5880) |
| Missing subject/pose/gaze cells | 0 of 5880 |
| Duplicate-content groups (SHA-256) | 0 |
| Positive (camera-directed) | 280 |
| Negative | 5598 |
| Excluded by config | 2 |
| Unlabeled | 0 |
| Near-camera negatives (V=0, abs(H)=5) | 560 |
| Subjects with only one class | 0 |
| Positives per subject (min, max) | [5, 5] |
| Identity-based splitting feasible | True |

## Label definition

`{'positive': {'gaze_v': [0], 'gaze_h': [0]}, 'negative': 'all_other', 'exclude': [], 'exclude_sample_ids': ['0007_2m_-15P_-10V_0H', '0007_2m_-15P_-10V_5H']}`

Positive = gaze directed at the camera (V = 0 deg, H = 0 deg), which Smith et al. (UIST 2013) call
*gaze locking*; gaze angles are measured relative to each camera position, so this holds for every head pose.
This is a controlled proxy for eye contact with a camera, not interpersonal eye contact.

## Licensing

Columbia Gaze Data Set: obtained via the CAVE download form (email required). The project page states the data may be used for non-commercial purposes only and that the UIST 2013 paper must be cited. No redistribution permission is stated, so FewGaze never commits or publishes dataset images.

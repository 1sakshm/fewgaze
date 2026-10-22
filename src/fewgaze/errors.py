"""Error analysis helpers.

Automatic failure categories are assigned ONLY from reliable metadata:
  * landmark_failure        MediaPipe found no face (sample excluded from every method)
  * near_camera_negative    negative with V = 0 and |H| = 5 deg (closest non-camera condition)
  * extreme_head_pose       |head pose| = 30 deg
  * vertical_offset_only    negative with H = 0 and V != 0
  * other                   none of the above (a genuine classifier error or an unexplained case)
Categories describe the condition of the image, not a verified cause. Causal labels require
manual review, which is recorded separately and marked as manual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_scores(preds: pd.DataFrame, method: str, budget: str) -> pd.DataFrame:
    q = preds[(preds["method"] == method) & (preds["condition"] == "clean")]
    q = q[q["budget"] == ("0" if (q["budget"] == "0").all() else budget)]
    return q.groupby("sample_id", as_index=False).agg(score=("score", "mean"), score_sd=("score", "std"), n_seeds=("seed", "nunique"))


def metadata_category(row: pd.Series) -> str:
    if row["label"] == 0 and row["gaze_v"] == 0 and abs(row["gaze_h"]) == 5:
        return "near_camera_negative"
    if row["label"] == 0 and row["gaze_h"] == 0 and row["gaze_v"] != 0:
        return "vertical_offset_only"
    if abs(row["head_pose"]) == 30:
        return "extreme_head_pose"
    return "other"


def error_table(preds: pd.DataFrame, meta: pd.DataFrame, method: str, budget: str, threshold: float) -> pd.DataFrame:
    s = mean_scores(preds, method, budget).merge(meta, on="sample_id")
    s["pred"] = (s["score"] >= threshold).astype(int)
    s["error"] = np.where((s["pred"] == 1) & (s["label"] == 0), "FP", np.where((s["pred"] == 0) & (s["label"] == 1), "FN", ""))
    s["category"] = s.apply(metadata_category, axis=1)
    s["method"] = method
    return s


def select_examples(err: pd.DataFrame, n_per_type: int) -> pd.DataFrame:
    """Prespecified selection: the n most confident false positives (highest score) and the n most
    confident false negatives (lowest score). No manual filtering."""
    fp = err[err["error"] == "FP"].sort_values("score", ascending=False).head(n_per_type)
    fn = err[err["error"] == "FN"].sort_values("score", ascending=True).head(n_per_type)
    return pd.concat([fp, fn], ignore_index=True)


def rates_by_condition(err: pd.DataFrame) -> pd.DataFrame:
    """Predicted-positive rate per (head pose, V, H) cell: FP rate for negatives, TP rate for positives."""
    return err.groupby(["head_pose", "gaze_v", "gaze_h", "label"], as_index=False).agg(
        predicted_positive_rate=("pred", "mean"), n=("pred", "size")
    )

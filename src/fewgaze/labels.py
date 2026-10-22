"""Derive binary camera-directed-gaze labels from the source annotations.

Label statuses:
  positive  -> label 1
  negative  -> label 0
  excluded  -> removed from the study by configuration (kept for accounting)
  unlabeled -> matched neither the positive rule nor an explicit negative list
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

STATUSES = ("positive", "negative", "excluded", "unlabeled")


class LabelError(ValueError):
    pass


def _match(df: pd.DataFrame, cond: dict[str, list[int]]) -> pd.Series:
    if not cond:
        raise LabelError("Empty condition")
    mask = pd.Series(True, index=df.index)
    for field, values in cond.items():
        if field not in df.columns:
            raise LabelError(f"Label condition refers to unknown field '{field}'")
        if not isinstance(values, (list, tuple)):
            values = [values]
        mask &= df[field].isin(values)
    return mask


def derive_labels(records: pd.DataFrame, label_cfg: dict[str, Any]) -> pd.DataFrame:
    """Return a table [sample_id, label_status, label] aligned with ``records``.

    Original annotations are not modified or copied here, so the mapping can be audited.
    """
    pos = _match(records, label_cfg["positive"])
    excl = pd.Series(False, index=records.index)
    for cond in label_cfg.get("exclude", []) or []:
        excl |= _match(records, cond)
    # individual samples excluded for a documented reason (e.g. duplicate content with conflicting annotations)
    excl |= records["sample_id"].isin(label_cfg.get("exclude_sample_ids", []) or [])
    if (pos & excl).any():
        raise LabelError("A condition is both positive and excluded; fix the label configuration")

    neg_rule = label_cfg["negative"]
    if neg_rule == "all_other":
        neg = ~pos & ~excl
    elif isinstance(neg_rule, list):
        neg = pd.Series(False, index=records.index)
        for cond in neg_rule:
            neg |= _match(records, cond)
        if (neg & pos).any():
            raise LabelError("A condition is both positive and negative")
        neg &= ~excl
    else:
        raise LabelError(f"Unknown negative rule: {neg_rule!r}")

    status = np.full(len(records), "unlabeled", dtype=object)
    status[neg.to_numpy()] = "negative"
    status[pos.to_numpy()] = "positive"
    status[excl.to_numpy()] = "excluded"
    label = np.where(status == "positive", 1, np.where(status == "negative", 0, -1))
    return pd.DataFrame({"sample_id": records["sample_id"].to_numpy(), "label_status": status, "label": label})


def labelled_view(records: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join annotations with derived labels, keeping only positive/negative samples."""
    merged = records.merge(labels, on="sample_id", how="inner", validate="one_to_one")
    return merged[merged["label"] >= 0].reset_index(drop=True)

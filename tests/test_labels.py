import pandas as pd
import pytest

from fewgaze.labels import LabelError, derive_labels, labelled_view


def _records():
    rows = [{"sample_id": f"s{v}_{h}", "gaze_v": v, "gaze_h": h, "head_pose": 0} for v in (-10, 0, 10) for h in (-5, 0, 5)]
    return pd.DataFrame(rows)


def test_default_mapping_positive_is_v0_h0():
    rec = _records()
    lab = derive_labels(rec, {"positive": {"gaze_v": [0], "gaze_h": [0]}, "negative": "all_other"})
    assert lab.loc[lab["label"] == 1, "sample_id"].tolist() == ["s0_0"]
    assert (lab["label"] == 0).sum() == 8
    assert list(rec.columns) == ["sample_id", "gaze_v", "gaze_h", "head_pose"]  # annotations untouched


def test_exclusion_of_ambiguous_near_camera_conditions():
    rec = _records()
    cfg = {"positive": {"gaze_v": [0], "gaze_h": [0]}, "negative": "all_other", "exclude": [{"gaze_v": [0], "gaze_h": [-5, 5]}]}
    lab = derive_labels(rec, cfg)
    assert set(lab.loc[lab["label_status"] == "excluded", "sample_id"]) == {"s0_-5", "s0_5"}
    view = labelled_view(rec, lab)
    assert len(view) == 7 and "s0_5" not in set(view["sample_id"])


def test_explicit_negative_list_leaves_unlabeled():
    rec = _records()
    lab = derive_labels(rec, {"positive": {"gaze_v": [0], "gaze_h": [0]}, "negative": [{"gaze_v": [10]}]})
    assert (lab["label_status"] == "unlabeled").sum() == 5
    assert (lab["label"] == 0).sum() == 3


def test_conflicting_rules_raise():
    rec = _records()
    with pytest.raises(LabelError):
        derive_labels(rec, {"positive": {"gaze_v": [0], "gaze_h": [0]}, "negative": "all_other", "exclude": [{"gaze_h": [0]}]})
    with pytest.raises(LabelError):
        derive_labels(rec, {"positive": {"gaze_v": [0]}, "negative": [{"gaze_h": [0]}]})
    with pytest.raises(LabelError, match="unknown field"):
        derive_labels(rec, {"positive": {"yaw": [0]}, "negative": "all_other"})


def test_fake_dataset_class_counts(labelled):
    # 10 subjects x 5 poses: one positive gaze condition each, 20 negatives each
    assert (labelled["label"] == 1).sum() == 50
    assert (labelled["label"] == 0).sum() == 1000
    assert labelled.groupby("subject")["label"].nunique().min() == 2


def test_exclude_individual_sample_ids():
    rec = _records()
    lab = derive_labels(rec, {"positive": {"gaze_v": [0], "gaze_h": [0]}, "negative": "all_other", "exclude_sample_ids": ["s10_5"]})
    assert lab.set_index("sample_id").loc["s10_5", "label_status"] == "excluded"
    assert (lab["label_status"] == "excluded").sum() == 1

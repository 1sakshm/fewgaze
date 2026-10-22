import re
from pathlib import Path

import pytest
from PIL import Image

from fewgaze.dataset import DatasetError, build_manifest, manifest_fingerprint, parse_filename


def test_parse_filename(base_cfg):
    pat = re.compile(base_cfg["dataset"]["filename_regex"])
    r = parse_filename("0042_2m_-15P_10V_-5H.jpg", pat)
    assert r == {"subject": "0042", "distance": 2, "head_pose": -15, "gaze_v": 10, "gaze_h": -5}
    assert parse_filename("0042_2m_-15P_10V.jpg", pat) is None
    assert parse_filename("notes.txt", pat) is None


def test_manifest_counts(cfg):
    man = build_manifest(cfg["dataset"]["root"], cfg["dataset"]["filename_regex"], [".jpg"])
    assert len(man.records) == 10 * 5 * 21
    assert man.records["subject"].nunique() == 10
    assert man.invalid.empty
    assert man.records["sample_id"].is_unique


def test_invalid_and_mismatched_files_are_reported(cfg, fake_root: Path):
    Image.new("RGB", (8, 8)).save(fake_root / "0001" / "0001_badname.jpg")
    Image.new("RGB", (8, 8)).save(fake_root / "0002" / "0003_2m_0P_0V_0H.jpg")  # file in the wrong folder
    (fake_root / "0001" / "readme.txt").write_text("not an image")
    man = build_manifest(cfg["dataset"]["root"], cfg["dataset"]["filename_regex"], [".jpg"])
    reasons = set(man.invalid["reason"])
    assert "filename_does_not_match_regex" in reasons
    assert "folder_subject_mismatch" in reasons
    assert not man.records["relpath"].str.contains("badname").any()
    assert not (man.records["relpath"] == "0002/0003_2m_0P_0V_0H.jpg").any()


def test_missing_root_gives_actionable_error(tmp_path):
    with pytest.raises(DatasetError, match="data/README.md"):
        build_manifest(tmp_path / "nope", r"(?P<subject>\d+)_(?P<head_pose>\d+)_(?P<gaze_v>\d+)_(?P<gaze_h>\d+)", [".jpg"])


def test_regex_without_required_groups(tmp_path):
    with pytest.raises(DatasetError, match="named groups"):
        build_manifest(tmp_path, r"(?P<subject>\d+)\.jpg", [".jpg"])


def test_fingerprint_changes_with_data(cfg, fake_root):
    man = build_manifest(cfg["dataset"]["root"], cfg["dataset"]["filename_regex"], [".jpg"])
    fp1 = manifest_fingerprint(man.records)
    (fake_root / "0001" / "0001_2m_0P_0V_0H.jpg").unlink()
    man2 = build_manifest(cfg["dataset"]["root"], cfg["dataset"]["filename_regex"], [".jpg"])
    assert manifest_fingerprint(man2.records) != fp1

"""Configuration loading. All experimental constants come from a YAML file."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when the configuration is missing a required field or is inconsistent."""


REQUIRED = {
    "dataset": ["root", "filename_regex", "subject_field"],
    "labels": ["positive", "negative"],
    "splits": ["n_folds", "seed"],
    "sampling": ["budgets", "seeds"],
    "classifier": ["C", "threshold"],
}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    cfg = copy.deepcopy(cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ConfigError("Config must be a mapping")
    for section, keys in REQUIRED.items():
        if section not in cfg:
            raise ConfigError(f"Config is missing section '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise ConfigError(f"Config section '{section}' is missing '{key}'")
    for b in cfg["sampling"]["budgets"]:
        if not (b == "all" or (isinstance(b, int) and b > 0)):
            raise ConfigError(f"Invalid budget {b!r}: use positive integers or 'all'")
    if int(cfg["splits"]["n_folds"]) < 2:
        raise ConfigError("n_folds must be at least 2")


def resolve_path(p: str | Path, root: Path = REPO_ROOT) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def output_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg.get("project", {}).get("output_dir", "results"))


def cache_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg.get("project", {}).get("cache_dir", "cache"))

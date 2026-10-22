"""Feature cache.

A cache entry lives in ``<cache_dir>/features/<name>-<key>/`` where ``key`` hashes the extractor
configuration, the preprocessing configuration and the dataset manifest fingerprint. Extraction
writes numbered chunk files, so an interrupted run resumes from the last complete chunk. When all
samples are processed, the chunks are consolidated into ``features.npz`` and ``meta.json`` gets
``complete: true``. Loading validates the key and the sample IDs; a stale cache is an error, never
silently reused.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fewgaze.utils import read_json, stable_hash, utc_now, write_json


class CacheError(RuntimeError):
    pass


@dataclass
class FeatureTable:
    name: str
    sample_ids: np.ndarray  # (n,) str
    X: np.ndarray  # (n, d) float32; rows for failed samples are NaN
    valid: np.ndarray  # (n,) bool
    status: np.ndarray  # (n,) str, e.g. 'ok', 'no_face', 'read_error'
    feature_names: list[str]
    meta: dict[str, Any]

    def subset(self, ids: Iterable[str]) -> np.ndarray:
        index = {s: i for i, s in enumerate(self.sample_ids)}
        return self.X[[index[s] for s in ids]]


def cache_key(extractor_cfg: dict[str, Any], preprocessing_cfg: dict[str, Any], fingerprint: str) -> str:
    return stable_hash({"extractor": extractor_cfg, "preprocessing": preprocessing_cfg, "manifest": fingerprint})


class FeatureCache:
    def __init__(self, root: Path, name: str, key: str):
        self.name, self.key = name, key
        self.dir = Path(root) / "features" / f"{name}-{key}"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def final_path(self) -> Path:
        return self.dir / "features.npz"

    def is_complete(self) -> bool:
        if not (self.meta_path.exists() and self.final_path.exists()):
            return False
        meta = read_json(self.meta_path)
        return bool(meta.get("complete")) and meta.get("key") == self.key

    def done_ids(self) -> set[str]:
        done: set[str] = set()
        for chunk in sorted(self.dir.glob("chunk_*.npz")):
            try:
                with np.load(chunk, allow_pickle=False) as z:
                    done.update(z["sample_ids"].tolist())
            except (OSError, ValueError, KeyError):
                chunk.unlink()  # truncated chunk from an interrupted write: discard and redo
        return done

    def write_chunk(self, sample_ids: list[str], X: np.ndarray, status: list[str]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        n = len(list(self.dir.glob("chunk_*.npz")))
        path = self.dir / f"chunk_{n:05d}.npz"
        tmp = self.dir / f"chunk_{n:05d}.tmp.npz"
        np.savez(tmp, sample_ids=np.array(sample_ids, dtype=str), X=X.astype(np.float32), status=np.array(status, dtype=str))
        tmp.replace(path)

    def finalize(self, all_ids: list[str], feature_names: list[str], meta: dict[str, Any]) -> FeatureTable:
        ids, Xs, sts = [], [], []
        for chunk in sorted(self.dir.glob("chunk_*.npz")):
            with np.load(chunk, allow_pickle=False) as z:
                ids.extend(z["sample_ids"].tolist())
                Xs.append(z["X"])
                sts.extend(z["status"].tolist())
        if not Xs:
            raise CacheError(f"No chunks to consolidate in {self.dir}")
        X = np.concatenate(Xs)
        pos = {s: i for i, s in enumerate(ids)}
        missing = [s for s in all_ids if s not in pos]
        if missing:
            raise CacheError(f"{len(missing)} samples were never extracted (e.g. {missing[:3]})")
        order = [pos[s] for s in all_ids]
        X, status = X[order], np.array(sts)[order]
        valid = status == "ok"
        np.savez(self.final_path, sample_ids=np.array(all_ids, dtype=str), X=X, status=status, valid=valid)
        full_meta = {
            **meta,
            "name": self.name,
            "key": self.key,
            "feature_names": feature_names,
            "n_samples": len(all_ids),
            "n_valid": int(valid.sum()),
            "created": utc_now(),
            "complete": True,
        }
        write_json(full_meta, self.meta_path)
        for chunk in self.dir.glob("chunk_*.npz"):
            chunk.unlink()
        return self.load(all_ids)

    def load(self, expected_ids: list[str] | None = None) -> FeatureTable:
        if not self.is_complete():
            raise CacheError(
                f"Feature cache '{self.name}' (key {self.key}) is missing or incomplete at {self.dir}. "
                "Run scripts/extract_features.py first."
            )
        meta = read_json(self.meta_path)
        with np.load(self.final_path, allow_pickle=False) as z:
            ids, X, status, valid = z["sample_ids"], z["X"], z["status"], z["valid"]
        if expected_ids is not None and list(ids) != list(expected_ids):
            raise CacheError(f"Cache '{self.name}' sample IDs do not match the current manifest (stale cache)")
        return FeatureTable(self.name, ids, X, valid, status, meta["feature_names"], meta)


def batched(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]

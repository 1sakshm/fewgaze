import numpy as np
import pytest

from fewgaze.features import CacheError, FeatureCache, cache_key


def _write(cache, ids, d=3):
    for chunk in (ids[:5], ids[5:]):
        cache.write_chunk(chunk, np.arange(len(chunk) * d, dtype=float).reshape(len(chunk), d), ["ok"] * len(chunk))


def test_cache_key_depends_on_every_input():
    base = cache_key({"m": "a"}, {"size": 448}, "fp1")
    assert base == cache_key({"m": "a"}, {"size": 448}, "fp1")
    assert base != cache_key({"m": "b"}, {"size": 448}, "fp1")
    assert base != cache_key({"m": "a"}, {"size": 224}, "fp1")
    assert base != cache_key({"m": "a"}, {"size": 448}, "fp2")


def test_roundtrip_and_resume(tmp_path):
    ids = [f"s{i}" for i in range(12)]
    c = FeatureCache(tmp_path, "feat", "k1")
    c.write_chunk(ids[:5], np.zeros((5, 2)), ["ok"] * 5)
    assert c.done_ids() == set(ids[:5]) and not c.is_complete()
    c.write_chunk(ids[5:], np.ones((7, 2)), ["ok"] * 6 + ["no_face"])
    t = c.finalize(list(reversed(ids)), ["a", "b"], {"x": 1})
    assert list(t.sample_ids) == list(reversed(ids))  # stored in requested order
    assert t.valid.sum() == 11 and t.status[0] == "no_face"
    assert c.is_complete()
    assert np.array_equal(t.subset(["s0", "s11"]), np.array([[0, 0], [1, 1]], dtype=np.float32))


def test_truncated_chunk_is_discarded(tmp_path):
    c = FeatureCache(tmp_path, "feat", "k1")
    c.write_chunk(["a", "b"], np.zeros((2, 2)), ["ok", "ok"])
    (c.dir / "chunk_00001.npz").write_bytes(b"garbage")
    assert c.done_ids() == {"a", "b"}
    assert not (c.dir / "chunk_00001.npz").exists()


def test_incomplete_or_stale_cache_raises(tmp_path):
    ids = [f"s{i}" for i in range(10)]
    c = FeatureCache(tmp_path, "feat", "k1")
    with pytest.raises(CacheError, match="extract_features"):
        c.load()
    c.write_chunk(ids[:4], np.zeros((4, 2)), ["ok"] * 4)
    with pytest.raises(CacheError, match="never extracted"):
        c.finalize(ids, ["a", "b"], {})
    c.write_chunk(ids[4:], np.zeros((6, 2)), ["ok"] * 6)
    c.finalize(ids, ["a", "b"], {})
    with pytest.raises(CacheError, match="stale"):
        c.load(ids[:-1])
    # a different key never sees this cache
    assert not FeatureCache(tmp_path, "feat", "k2").is_complete()

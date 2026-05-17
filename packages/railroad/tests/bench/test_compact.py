"""Tests for the benchmark compaction cache (railroad.bench.compact).

These focus on the per-experiment stamp-file fingerprint introduced to stop
unrelated benchmark runs from invalidating every experiment's cache, plus the
purge-on-corruption and cache-format behaviors.
"""

import pandas as pd
import pytest

from railroad.bench import compact


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    """Isolate the cache dir and point the artifact root at a tmp dir."""
    cache_dir = tmp_path / ".benchmark_cache"
    artifact_dir = tmp_path / "mlruns" / "1"
    artifact_dir.mkdir(parents=True)

    monkeypatch.setattr(compact, "_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(compact, "_artifact_root", lambda exp: artifact_dir)
    return tmp_path, artifact_dir


def _df():
    return pd.DataFrame({"status": ["FINISHED", "FINISHED"], "x": [1, 2]})


def test_stamp_drives_invalidation(cache_env):
    """Cache hits while the stamp is unchanged, misses once it's touched."""
    compact.touch_stamp("exp")
    assert compact.save("exp", _df(), {"m": 1}, {"total_runs": 2})

    loaded = compact.load("exp")
    assert loaded is not None
    df, meta, summary = loaded
    assert len(df) == 2 and meta == {"m": 1} and summary["total_runs"] == 2

    # An unrelated run does NOT touch this experiment's stamp -> still a hit.
    assert compact.load("exp") is not None

    # This experiment's data changes -> stamp bumps -> cache miss.
    compact.touch_stamp("exp")
    assert compact.load("exp") is None


def test_corrupt_meta_purges_cache(cache_env):
    compact.touch_stamp("exp")
    assert compact.save("exp", _df(), {}, {})
    runs_path, meta_path, _ = compact._cache_paths("exp")
    meta_path.write_text("{ not json")

    assert compact.load("exp") is None
    # Whole experiment cache dir purged so the next load rebuilds cleanly.
    assert not meta_path.parent.exists()


def test_corrupt_figures_purges_cache(cache_env):
    compact.touch_stamp("exp")
    assert compact.save("exp", _df(), {}, {}, figures={})
    _runs, _meta, figures_path = compact._cache_paths("exp")
    figures_path.write_text("{ not json")

    assert compact.load_figures("exp") is None
    assert not figures_path.parent.exists()


def test_cache_format_bump_invalidates_without_purge(cache_env, monkeypatch):
    compact.touch_stamp("exp")
    assert compact.save("exp", _df(), {}, {})

    monkeypatch.setattr(compact, "CACHE_FORMAT_VERSION", compact.CACHE_FORMAT_VERSION + 1)
    # A format bump is a normal fingerprint mismatch: miss, but not a purge
    # (save() overwrites it on the rebuild).
    assert compact.load("exp") is None
    runs_path, _meta, _fig = compact._cache_paths("exp")
    assert runs_path.exists()


def test_fallback_fingerprint_without_artifact_root(tmp_path, monkeypatch):
    """With no resolvable artifact root, the legacy fingerprint still works."""
    cache_dir = tmp_path / ".benchmark_cache"
    monkeypatch.setattr(compact, "_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(compact, "_artifact_root", lambda exp: None)

    fp = compact._source_fingerprint("exp")
    assert fp["cache_format"] == compact.CACHE_FORMAT_VERSION
    assert "stamp" not in fp

    assert compact.save("exp", _df(), {"m": 2}, {"total_runs": 2})
    loaded = compact.load("exp")
    assert loaded is not None and loaded[1] == {"m": 2}


def test_remove_stamp_roundtrip(cache_env):
    _tmp, artifact_dir = cache_env
    stamp = artifact_dir / compact.STAMP_FILENAME

    compact.touch_stamp("exp")
    assert stamp.exists()

    compact.remove_stamp("exp")
    assert not stamp.exists()
    # Idempotent.
    compact.remove_stamp("exp")

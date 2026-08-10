"""Where the per-seed scene cache is looked for.

The path used to be hardcoded relative to the working directory, so a run
started from anywhere but the directory holding ``resources/`` silently missed
every cached scene and generated it again -- slow at best, and on a headless
machine a hard failure, for a scene that was sitting on disk the whole time.

It follows the resources directory now, which is what ``PROCTHOR_RESOURCES_DIR``
sets and what every other ProcTHOR asset already used.
"""

import pickle

import pytest

from railroad.environment.procthor import resources
from railroad.environment.procthor.thor_interface import ThorInterface


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the resources base somewhere new, as the env var would.

    The env var is read once at import, so the test moves the value it
    produced rather than the variable.
    """
    monkeypatch.setattr(resources, "DEFAULT_RESOURCES_BASE", tmp_path / "elsewhere")
    return tmp_path / "elsewhere" / "procthor-10k" / "cache"


def _interface(seed: int) -> ThorInterface:
    """A ThorInterface without ``__init__``, which would load a whole scene."""
    thor = object.__new__(ThorInterface)
    thor.seed = seed
    return thor


def test_the_cache_follows_the_resources_directory(cache_dir):
    assert _interface(7)._cache_dir() == cache_dir


def test_a_cached_scene_is_found_from_any_working_directory(
    cache_dir, tmp_path, monkeypatch
):
    """The bug: this returned None, and the caller started Unity."""
    cache_dir.mkdir(parents=True)
    (cache_dir / "scene_7.pkl").write_bytes(pickle.dumps({"reachable_positions": []}))

    monkeypatch.chdir(tmp_path)  # anywhere but the directory holding resources/
    assert _interface(7)._load_cache() == {"reachable_positions": []}


def test_a_missing_scene_is_still_a_miss(cache_dir):
    cache_dir.mkdir(parents=True)
    assert _interface(7)._load_cache() is None


def test_an_explicit_path_still_wins(cache_dir, tmp_path):
    somewhere = tmp_path / "somewhere"
    somewhere.mkdir()
    (somewhere / "scene_3.pkl").write_bytes(pickle.dumps({"marker": True}))
    assert _interface(3)._load_cache(str(somewhere)) == {"marker": True}

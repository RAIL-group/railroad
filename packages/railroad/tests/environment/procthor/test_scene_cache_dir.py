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
    # NOTE: added to account for my custom modifications to thor_interface
    thor.object_seed = None
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


class TestTheCacheSurvivesAnInterruptedWrite:
    """Generating a scene is slow, so it gets interrupted.

    Dumping straight to the destination leaves a truncated pickle that every
    later run fails to load -- the scene is permanently broken rather than
    regenerated, and the cache is shared, so one bad file follows the host
    around.
    """

    @staticmethod
    def _cache(marker: str) -> dict:
        return {"reachable_positions": [], "marker": marker}

    def test_an_interrupted_write_leaves_the_previous_cache_intact(self, tmp_path):
        thor = _interface(5)
        target = tmp_path / "scene_5.pkl"
        thor._write_cache_atomically(self._cache("original"), target)

        def explode(obj, file):
            file.write(b"\x80\x04\x95truncated")
            raise KeyboardInterrupt

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pickle, "dump", explode)
            with pytest.raises(KeyboardInterrupt):
                thor._write_cache_atomically(self._cache("replacement"), target)

        assert thor._load_cache(str(tmp_path)) == self._cache("original")
        # And no half-written temporary file left lying beside it.
        assert [p.name for p in tmp_path.iterdir()] == ["scene_5.pkl"]

    def test_a_written_cache_is_readable_by_whoever_shares_the_directory(self, tmp_path):
        """mkstemp is 0600, but PROCTHOR_RESOURCES_DIR exists to point this
        somewhere shared -- and an entry nobody else can read is one they
        regenerate, launching Unity, on every run."""
        import os
        import stat

        target = tmp_path / "scene_5.pkl"
        _interface(5)._write_cache_atomically(self._cache("x"), target)

        umask = os.umask(0o777)
        os.umask(umask)
        assert stat.S_IMODE(target.stat().st_mode) == 0o666 & ~umask

    def test_an_unreadable_cache_reads_as_a_miss_rather_than_raising(self, tmp_path):
        """Recovers files written before the atomic swap existed."""
        (tmp_path / "scene_9.pkl").write_bytes(b"\x80\x04\x95 truncated garbage")
        with pytest.warns(RuntimeWarning, match="could not be read"):
            assert _interface(9)._load_cache(str(tmp_path)) is None

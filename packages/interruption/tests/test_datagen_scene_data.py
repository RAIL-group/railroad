import importlib.util
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "multiprocess_datagen.py"


class _Env:
    """Just enough of an environment for _generate_scene_data to walk a plan."""

    def get_actions(self):
        return []

    def act(self, action):
        pass

    def update_scene_graph(self, action):
        pass

    def fork(self):
        return _Env()


@pytest.fixture
def datagen(monkeypatch):
    spec = importlib.util.spec_from_file_location("multiprocess_datagen_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run = SimpleNamespace(data=SimpleNamespace(env=_Env()), start_state_recorded=False)
    plan = [SimpleNamespace(name=f"action{i}") for i in range(3)]
    monkeypatch.setattr(module, "_open_scene_run", lambda *args: run)
    monkeypatch.setattr(module, "_plan_next_task", lambda *args: plan)
    monkeypatch.setattr(module, "get_action_by_name", lambda actions, name: name)
    return module


def _recording_outcomes(datagen, monkeypatch, outcomes):
    """Makes _record_datum return the given outcomes in order; returns the call log."""
    outcomes = iter(outcomes)
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs["file_counter"])
        return next(outcomes)

    monkeypatch.setattr(datagen, "_record_datum", record)
    return calls


def _generate(datagen, target):
    return datagen._generate_scene_data(
        scene_seed=5, scene_key=123, task_distribution=([], []), relevant_objects=None,
        target=target,
    )


def test_gives_up_after_consecutive_failed_recordings(datagen, monkeypatch):
    limit = datagen.MAX_FAILED_RECORDINGS
    calls = _recording_outcomes(datagen, monkeypatch, [False] * limit)

    with pytest.raises(RuntimeError, match=f"{limit} consecutive states of scene 5"):
        _generate(datagen, target=1)
    assert len(calls) == limit


def test_a_recorded_datum_resets_the_failure_count(datagen, monkeypatch):
    limit = datagen.MAX_FAILED_RECORDINGS
    # just under the limit twice, separated by one success: never gives up
    outcomes = [False] * (limit - 1) + [True] + [False] * (limit - 1) + [True]
    calls = _recording_outcomes(datagen, monkeypatch, outcomes)

    assert _generate(datagen, target=2) == 2
    assert len(calls) == len(outcomes)


class _InlineExecutor:
    """Runs submitted work immediately, in-process, in place of the process pool."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as e:  # noqa: BLE001 - handed back like a real pool would
            future.set_exception(e)
        return future


def test_a_failing_scene_is_reported_without_stopping_the_others(datagen, monkeypatch):
    monkeypatch.setattr(datagen, "ProcessPoolExecutor", _InlineExecutor)

    def generate(scene_seed, scene_key, task_distribution, relevant_objects, target):
        if scene_seed == 9:
            raise RuntimeError("no plannable task in scene 9")
        return target

    monkeypatch.setattr(datagen, "_generate_scene_data", generate)

    written, failures = datagen._run_workers({5: 1, 9: 2, 32: 3}, ([], []), None)

    assert written == 2 * datagen.NUM_DATUM
    assert failures == {9: "RuntimeError: no plannable task in scene 9"}


def test_a_dead_worker_stops_the_run(datagen, monkeypatch):
    monkeypatch.setattr(datagen, "ProcessPoolExecutor", _InlineExecutor)

    def generate(*args):
        raise BrokenProcessPool("a worker died")

    monkeypatch.setattr(datagen, "_generate_scene_data", generate)

    with pytest.raises(RuntimeError, match="lower MAX_WORKERS"):
        datagen._run_workers({5: 1}, ([], []), None)

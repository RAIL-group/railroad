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


def _load_datagen():
    spec = importlib.util.spec_from_file_location("multiprocess_datagen_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def datagen(monkeypatch):
    """The script, with scene opening and planning stubbed out."""
    module = _load_datagen()
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

    created_with: dict = {}

    def __init__(self, *args, **kwargs):
        _InlineExecutor.created_with = kwargs

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


def _scene_with_tasks(datagen, monkeypatch, tasks):
    """Stubs scene construction: a scene with one object and one location."""
    from railroad.core import Fluent, LiteralGoal

    config = SimpleNamespace(interrupting_task_dist=([LiteralGoal(Fluent(t)) for t in tasks], []))
    scene = SimpleNamespace(objects={"apple_27"}, locations={"fridge_5": (0, 0)})
    env = SimpleNamespace(scene=scene, objects_by_type={"category": {"fridge"}})
    data = SimpleNamespace(env=env)
    monkeypatch.setattr(datagen, "initialize_experiment_config", lambda *args: config)
    monkeypatch.setattr(datagen, "initialize_experiment_data", lambda *args, **kwargs: data)


def test_opening_a_scene_with_an_unmappable_task_fails_immediately(monkeypatch):
    datagen = _load_datagen()
    _scene_with_tasks(
        datagen, monkeypatch, ["at-category apple_27 fridge", "at-category bowl fridge"]
    )

    with pytest.raises(RuntimeError, match=r"scene 5: tasks \[\(at-category bowl fridge\)\]"):
        datagen._open_scene_run(5, ([], []), None)


def test_opening_a_scene_whose_tasks_all_map_succeeds(monkeypatch):
    datagen = _load_datagen()
    _scene_with_tasks(datagen, monkeypatch, ["at-category apple_27 fridge"])

    run = datagen._open_scene_run(5, ([], []), None)

    assert run.data.env.scene.objects == {"apple_27"}


def test_every_scene_gets_a_fresh_worker_process(datagen, monkeypatch):
    monkeypatch.setattr(datagen, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(datagen, "_generate_scene_data", lambda *args: 0)

    datagen._run_workers({5: 1, 9: 2}, ([], []), None)

    assert _InlineExecutor.created_with["max_tasks_per_child"] == datagen.SCENES_PER_WORKER == 1


def test_memory_usage_reports_a_peak(datagen):
    current_gb, peak_gb = datagen._memory_usage_gb()

    assert 0 < peak_gb < 1024
    assert current_gb is None or 0 < current_gb < 1024


def test_memory_is_logged_when_a_scene_opens_and_finishes(datagen, monkeypatch, capsys):
    _recording_outcomes(datagen, monkeypatch, [True, True])

    _generate(datagen, target=2)

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[scene 5]")]
    assert [line.split(" (")[0] for line in lines] == ["[scene 5] opened", "[scene 5] finished"]
    assert "peak" in lines[-1]


def test_memory_is_logged_when_a_scene_fails(datagen, monkeypatch, capsys):
    _recording_outcomes(datagen, monkeypatch, [False] * datagen.MAX_FAILED_RECORDINGS)

    with pytest.raises(RuntimeError):
        _generate(datagen, target=1)

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[scene 5]")]
    assert [line.split(" (")[0] for line in lines] == ["[scene 5] opened", "[scene 5] failed"]


@pytest.mark.parametrize("seconds, logged", [(5.0, False), (75.0, True)])
def test_slow_datums_are_logged(seconds, logged, monkeypatch, capsys, tmp_path):
    datagen = _load_datagen()
    monkeypatch.setattr(datagen, "get_procthor_10k_dir", lambda: tmp_path)
    clock = iter([0.0, seconds])
    monkeypatch.setattr(datagen.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(datagen, "convert_state_to_positive_preconditions", lambda *args: None)
    def compute(*args, search_stats):
        searches = (("(at-category cup_3 fridge)", 1200), ("(at-category bowl_2 sofa)", 30000))
        for goal, expanded in searches:
            search_stats.append(datagen.SearchStats(goal, expanded, 30000, expanded < 30000))
        return -1, None

    monkeypatch.setattr(datagen, "compute_interruption_value", compute)
    data = SimpleNamespace(
        search_problem=SimpleNamespace(interrupting_task_dist=([], []), actions=[]),
        planner_parameters=SimpleNamespace(heuristic_fn=0), neg_to_pos_mapping={},
    )

    recorded = datagen._record_datum(
        SimpleNamespace(state=None), data, ([], []), scene_seed=5, scene_key=1, file_counter=7
    )

    assert recorded is False
    out = capsys.readouterr().out
    expected = (
        "[scene 5] slow datum 7: 75s (unsolvable task, not recorded); "
        "largest search 30000/30000 expansions (100%) for (at-category bowl_2 sofa); "
        "all 2 searches 31200/60000 (52%)"
    )
    assert (expected in out) == logged

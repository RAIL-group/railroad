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
def datagen(monkeypatch, tmp_path):
    """The script, with scene opening and planning stubbed out, writing under tmp_path."""
    module = _load_datagen()
    monkeypatch.setattr(module, "get_procthor_10k_dir", lambda: tmp_path)
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
    def expected_task_cost(env, data, search_stats):
        searches = (("(at-category cup_3 fridge)", 1200), ("(at-category bowl_2 sofa)", 30000))
        for goal, expanded in searches:
            search_stats.append(datagen.SearchStats(goal, expanded, 30000, expanded < 30000))
        return -1, None

    monkeypatch.setattr(datagen, "_expected_task_cost", expected_task_cost)
    data = SimpleNamespace(
        search_problem=SimpleNamespace(interrupting_task_dist=([], []), actions=[]),
        planner_parameters=SimpleNamespace(heuristic_fn=0), neg_to_pos_mapping={},
    )

    recorded = datagen._record_datum(
        SimpleNamespace(state=None, scene=SimpleNamespace(objects=set())), data, ([], []),
        scene_seed=5, scene_key=1, file_counter=7,
    )

    assert recorded is False
    out = capsys.readouterr().out
    expected = (
        "[scene 5] slow datum 7: 75s (unsolvable task, not recorded); "
        "largest search 30000/30000 expansions (100%) for (at-category bowl_2 sofa); "
        "all 2 searches 31200/60000 (52%)"
    )
    assert (expected in out) == logged


def _prewarm_with(monkeypatch, tmp_path, present, failing=()):
    """Stubs cache paths (present ones exist) and scene loading; returns the scenes built."""
    datagen = _load_datagen()
    for seed in present:
        (tmp_path / f"scene_{seed}.pkl").touch()
    monkeypatch.setattr(datagen, "scene_cache_path", lambda seed: tmp_path / f"scene_{seed}.pkl")
    built = []

    def build(seed, **kwargs):
        if seed in failing:
            raise RuntimeError(f"Unity failed on {seed}")
        built.append(seed)

    monkeypatch.setattr(datagen, "ThorInterface", build)
    monkeypatch.setenv("DISPLAY", ":2")
    return datagen, built


def test_prewarm_builds_only_the_missing_caches(monkeypatch, tmp_path, capsys):
    datagen, built = _prewarm_with(monkeypatch, tmp_path, present=[5])

    datagen._prewarm_scene_caches([5, 9, 32], None)

    assert built == [9, 32]
    assert "Scene caches: 1/3 present, building 2" in capsys.readouterr().out


def test_prewarm_lists_every_scene_whose_cache_fails(monkeypatch, tmp_path):
    datagen, built = _prewarm_with(monkeypatch, tmp_path, present=[], failing=(9, 32))

    with pytest.raises(RuntimeError, match=r"could not build 2 scene cache\(s\)") as error:
        datagen._prewarm_scene_caches([5, 9, 32], None)

    assert built == [5]
    assert "Unity failed on 9" in str(error.value) and "Unity failed on 32" in str(error.value)


def test_prewarm_needs_a_display_only_when_caches_are_missing(monkeypatch, tmp_path):
    datagen, built = _prewarm_with(monkeypatch, tmp_path, present=[5])
    monkeypatch.setattr(datagen.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY")

    datagen._prewarm_scene_caches([5], None)  # nothing to build: no display needed
    with pytest.raises(RuntimeError, match="needs a display") as error:
        datagen._prewarm_scene_caches([5, 9], None)
    assert built == []
    assert "ai2thor-xorg --height 4096 --width 4096 start 1" in str(error.value)
    assert "export DISPLAY=:1" in str(error.value)
    assert "ai2thor-xorg stop 1" in str(error.value)


def test_releasing_memory_collects_and_trims_where_glibc_exists(monkeypatch):
    datagen = _load_datagen()
    calls = []
    monkeypatch.setattr(datagen.gc, "collect", lambda: calls.append("collect"))
    libc = SimpleNamespace(malloc_trim=lambda pad: calls.append(pad))
    monkeypatch.setattr(datagen, "_LIBC", libc)

    datagen._release_freed_memory()
    assert calls == ["collect", 0]

    monkeypatch.setattr(datagen, "_LIBC", None)  # macOS: no glibc, collection only
    datagen._release_freed_memory()
    assert calls == ["collect", 0, "collect"]


@pytest.mark.parametrize("enabled", [True, False])
def test_memory_is_released_after_each_computed_datum(enabled, monkeypatch, tmp_path):
    datagen = _load_datagen()
    monkeypatch.setattr(datagen, "get_procthor_10k_dir", lambda: tmp_path)
    monkeypatch.setattr(datagen, "RELEASE_MEMORY_AFTER_DATUM", enabled)
    released = []
    monkeypatch.setattr(datagen, "_release_freed_memory", lambda: released.append(True))
    monkeypatch.setattr(datagen, "_expected_task_cost", lambda env, data, stats: (-1, None))
    data = SimpleNamespace(search_problem=SimpleNamespace(interrupting_task_dist=([], [])))

    datagen._record_datum(
        SimpleNamespace(state=None), data, ([], []), scene_seed=5, scene_key=1, file_counter=0
    )

    assert released == ([True] if enabled else [])


def test_unfinished_scenes_go_first_by_size_then_finished_ones_by_duration(datagen):
    order = datagen._longest_first(
        [5, 9, 32, 33], durations={9: 448.0, 33: 20.0}, estimates={5: 1300.0, 32: 1100.0}
    )
    assert order == [5, 32, 9, 33]


def test_scenes_are_submitted_longest_first_and_their_durations_recorded(datagen, monkeypatch):
    monkeypatch.setattr(datagen, "ProcessPoolExecutor", _InlineExecutor)
    started = []
    monkeypatch.setattr(
        datagen, "_generate_scene_data", lambda seed, *args: started.append(seed) or 2
    )
    scene_keys = {5: 1, 9: 2, 32: 3}

    written, failures = datagen._run_workers(
        scene_keys, ([], []), None, scene_costs={5: 10.0, 9: 30.0, 32: 20.0}
    )

    assert (written, failures, started) == (6, {}, [9, 32, 5])
    path = datagen._run_metadata_path("scene_durations", datagen._run_key(scene_keys))
    assert set(datagen._load_scene_durations(path)) == {5, 9, 32}


def test_a_later_run_orders_scenes_by_their_recorded_durations(datagen, monkeypatch):
    monkeypatch.setattr(datagen, "ProcessPoolExecutor", _InlineExecutor)
    started = []
    monkeypatch.setattr(
        datagen, "_generate_scene_data", lambda seed, *args: started.append(seed) or 2
    )
    scene_keys = {5: 1, 9: 2}
    path = datagen._run_metadata_path("scene_durations", datagen._run_key(scene_keys))
    datagen._write_scene_durations(path, {5: 500.0, 9: 4.0})

    datagen._run_workers(scene_keys, ([], []), None, scene_costs={5: 1.0, 9: 99.0})

    assert started == [5, 9]  # the recorded durations win over the size estimate


def test_scene_log_lines_include_the_elapsed_time(datagen, monkeypatch, capsys):
    _recording_outcomes(datagen, monkeypatch, [True, True])

    _generate(datagen, target=2)

    finished = [line for line in capsys.readouterr().out.splitlines() if "finished" in line]
    assert finished and " after " in finished[0] and finished[0].split(" after ")[1][0].isdigit()


@pytest.mark.parametrize("fraction, expected", [(0.0, 0), (1.0, 2000)])
def test_the_audited_fraction_is_honoured_at_the_extremes(
    datagen, monkeypatch, fraction, expected
):
    monkeypatch.setattr(datagen, "LABEL_AUDIT_FRACTION", fraction)
    assert sum(datagen._is_audited(5, counter) for counter in range(2000)) == expected


def test_about_the_configured_fraction_of_datums_is_audited(datagen, monkeypatch):
    monkeypatch.setattr(datagen, "LABEL_AUDIT_FRACTION", 0.05)
    audited = sum(datagen._is_audited(5, counter) for counter in range(4000))
    assert 140 < audited < 260  # ~200, and the same every run (seeded)
    assert audited == sum(datagen._is_audited(5, counter) for counter in range(4000))


def _audit_data():
    from railroad.core import Fluent, LiteralGoal

    goals = [
        LiteralGoal(Fluent("at-category cup_3 fridge")),
        LiteralGoal(Fluent("at-category pan_2 sofa")),
    ]
    problem = SimpleNamespace(interrupting_task_dist=(goals, [0.5, 0.5]))
    return SimpleNamespace(search_problem=problem)


def test_the_audit_records_both_costs_and_logs_only_differences(
    datagen, monkeypatch, capsys, tmp_path
):
    unfiltered = iter([10.0, 12.5])
    monkeypatch.setattr(datagen, "_task_cost", lambda *args, **kwargs: next(unfiltered))

    datagen._audit_labels(
        None, _audit_data(), [10.0, 14.0], scene_seed=5, scene_key=1, file_counter=7
    )

    rows = (tmp_path / "label_audit_5_1.csv").read_text().splitlines()
    assert rows == [
        "file_counter,task,filtered_cost,unfiltered_cost",
        "7,(at-category cup_3 fridge),10.0,10.0",
        "7,(at-category pan_2 sofa),14.0,12.5",
    ]
    out = capsys.readouterr().out
    assert "[scene 5] audit datum 7: 1/2 task cost(s) differ" in out and "pan_2" in out


def test_the_audit_summary_counts_differences_across_scenes(datagen, monkeypatch, capsys):
    unfiltered = iter([10.0, 12.5, None, 8.0])
    monkeypatch.setattr(datagen, "_task_cost", lambda *args, **kwargs: next(unfiltered))
    for scene_seed, scene_key, filtered in ((5, 1, [10.0, 14.0]), (9, 2, [9.0, 8.0])):
        datagen._audit_labels(
            None, _audit_data(), filtered, scene_seed=scene_seed, scene_key=scene_key,
            file_counter=0,
        )
    capsys.readouterr()

    datagen._summarize_label_audit({5: 1, 9: 2, 32: 3})  # scene 32 has no audit file

    assert capsys.readouterr().out.strip() == (
        "Label audit: 4 task costs rechecked without relaxed-plan filtering; "
        "1 differ (worst +12.0%); 1 unsolved unfiltered"
    )


@pytest.mark.parametrize("filtering, audited, expected_audits", [
    (True, True, 1), (True, False, 0), (False, True, 0),
])
def test_record_datum_audits_only_selected_datums_with_filtering_on(
    datagen, monkeypatch, filtering, audited, expected_audits
):
    audits = []
    monkeypatch.setattr(datagen, "_expected_task_cost", lambda env, data, stats: (5.0, [5.0]))
    monkeypatch.setattr(datagen, "_is_audited", lambda seed, counter: audited)
    monkeypatch.setattr(datagen, "_audit_labels", lambda *args: audits.append(args))
    monkeypatch.setattr(datagen, "write_datum_to_file", lambda *args: None)
    monkeypatch.setattr(datagen, "WRITE_OUT_INDIVIDUAL_TASK_COSTS", False)
    data = SimpleNamespace(
        search_problem=SimpleNamespace(interrupting_task_dist=([], [])),
        planner_parameters=SimpleNamespace(relaxed_plan_filtering=filtering),
    )
    env = SimpleNamespace(scene=SimpleNamespace(scene_graph=None))

    assert datagen._record_datum(env, data, ([], []), scene_seed=5, scene_key=1, file_counter=3)
    assert len(audits) == expected_audits

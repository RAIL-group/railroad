"""
Data-generation script for expected value over interrupting task distribution for
ProcTHOR environments.
NOTE: this file contains a fix for the objects_seed misrecording issue that is not currently
updated in datagen.py
"""
from functools import partial
from typing import Sequence
import hashlib
import json
import multiprocessing
import os
import random
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
# from functools import partial
from pathlib import Path

from interruption.environments import (
    KitchenProcTHOREnvironment,
    construct_procthor_kitchen_environment,
    get_alfred_task_distribution,
    get_example_procthor_goal,
)
from interruption.experiments import (
    ExperimentConfig,
    ExperimentData,
    ExperimentSeeds,
    initialize_experiment_data,
)
from interruption.planning_framework import PlannerMode
from interruption.constants import (
    NUM_TASKS, PROCTHOR_SEED, FILTER_OBJECTS, NUM_ROOMS_FILTER, ONE_ROOM_FILTER,
)
from interruption.learning.data import write_compressed_pickle
from interruption.planner import SearchStats, astar_search, compute_interruption_value
from interruption.utilities import (
    RandomVariableType, get_task_arrival_prob, extract_relevant_objects,
    filter_procthor_scenes, find_exemplar_scene, remap_scene_objects_and_locations
)
from railroad.core import (
    Action,
    Fluent,
    Goal,
    convert_state_to_positive_preconditions,
    get_action_by_name,
)
from railroad.environment.procthor.resources import REMAP_DIR_ENV_VAR, get_procthor_10k_dir
from railroad.environment.procthor.scenegraph import SceneGraph
from railroad.environment.procthor.thor_interface import ThorInterface, scene_cache_path

NUM_DATUM = 800  # datums to generate for each scene seed
DATA_GENERATION_SEED = 37
REMOVE_DUPLICATE_OBJECTS = True
H_MULTIPLIER = 1
WRITE_OUT_INDIVIDUAL_TASK_COSTS = True

# Worker processes, each generating one scene's data at a time. Only affects
# speed and memory, never the data: a scene's data depends only on the scene.
# Concurrent AI2-THOR Controller instances this machine sustains without
# throughput degrading (see benchmark_thor_concurrency.py: 8 is the
# peak-throughput point, 16+ regresses, 24 produced an outright hang).
MAX_WORKERS = 8
# Scenes a worker process generates before it is replaced by a fresh one. Python
# keeps memory a process has freed rather than returning it to the OS, so a
# long-lived worker would carry its largest search's footprint into every later
# scene; a fresh process starts each scene from the import baseline instead.
SCENES_PER_WORKER = 1
# How often (in datums) a worker logs its memory use while generating a scene;
# it also logs at the start and end of every scene.
MEMORY_LOG_EVERY = 25
# A datum whose labels take longer than this (seconds) to compute is logged, with
# whether it was recorded, so slow searches stay visible without progress bars.
SLOW_DATUM_SECONDS = 60

# Folded into the per-(scene, count) random.seed() used for task sampling, so
# different scenes don't draw identical task sequences at the same count.
SEED_STRIDE = 100_000

# How many times a scene visit may re-deal a scene's objects (see
# _plan_next_task) before giving up on that visit.
MAX_RESHUFFLES_PER_VISIT = 5
# consecutive visits to one scene that find nothing to plan, even after
# reshuffling, before _generate_scene_data gives up on it
MAX_STALLED_VISITS = 2
# consecutive states of one scene whose datum can't be recorded (some task in
# the distribution can't be solved from them) before _generate_scene_data gives
# up on it, rather than walking unrecordable states forever
MAX_FAILED_RECORDINGS = 50


# --------------------------------------------------------------------------- #
# Data types shared by the pipeline stages below
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SceneSelection:
    matching_seeds: list[int]
    task_objects: set[str]
    task_locations: set[str]
    remapped_scenes: dict[int, dict]  # empty when the plain scene is used as-is


@dataclass
class _SceneRun:
    """A scene's persistent environment, plus its RNG bookkeeping."""
    config: ExperimentConfig
    data: ExperimentData
    attempt: int = 0       # goal samples drawn so far; never reused, so no sample repeats
    num_shuffles: int = 0  # reshuffles so far; seeds the next one
    # whether the datum for the environment's current state was already written
    # by the previous visit's walk (which passed through it), so it isn't
    # written twice
    start_state_recorded: bool = False


# --------------------------------------------------------------------------- #
# Pipeline (in the order main() drives them)
# --------------------------------------------------------------------------- #
def main(prewarm_only: bool = False):
    """
    Data-generation entrypoint: select scenes -> build the task distribution ->
    key each scene -> build any missing scene caches -> generate each scene's data
    in a pool of workers. prewarm_only stops after the caches are built
    (`python scripts/multiprocess_datagen.py --prewarm-only`), so they can be built
    ahead of a run.
    """
    start = time.perf_counter()

    # remaps are opt-in per run: start from a known state rather than
    # inheriting whatever an earlier run or the calling shell left set, so
    # the environments built during scene selection are plain
    os.environ.pop(REMAP_DIR_ENV_VAR, None)

    selection = _select_scenes(ONE_ROOM_FILTER)
    task_distribution = get_alfred_task_distribution(
        selection.task_objects,
        selection.task_locations,
        size=NUM_TASKS,
        one_object_per_taskdist=True,
    )
    scene_keys = _compute_scene_keys(selection)
    run_key = _run_key(scene_keys)
    _write_scene_selection(selection, run_key)
    _write_task_distribution(task_distribution, run_key)
    relevant_objects = extract_relevant_objects(task_distribution[0]) if FILTER_OBJECTS else None

    with _remaps_enabled(selection.remapped_scenes, scene_keys):
        _prewarm_scene_caches(list(scene_keys), relevant_objects)
        if prewarm_only:
            return
        total_written, failures = _run_workers(scene_keys, task_distribution, relevant_objects)

    print(f"Data Generation took: {time.perf_counter() - start: .4f} seconds")
    print(
        f"Wrote {total_written} data points ({NUM_DATUM} for each of "
        f"{len(scene_keys) - len(failures)}/{len(scene_keys)} scene(s))"
    )
    if failures:
        details = "\n".join(f"  scene {seed}: {error}" for seed, error in sorted(failures.items()))
        raise RuntimeError(f"{len(failures)} scene(s) failed:\n{details}")


def _select_scenes(scene_filter: dict[str, int | set[str]]) -> SceneSelection:
    """
    num_scenes == 1: just [PROCTHOR_SEED], the plain un-remapped scene.

    num_scenes > 1: filter the dataset, pick an exemplar whose vocabulary
    defines the shared object/location set, and remap every matching scene
    onto it (once, in the parent process) so one task_distribution is valid
    for all of them. num_scenes only bounds how many scenes get remapped; it
    is not a filter_procthor_scenes criterion.
    """
    if scene_filter["num_scenes"] == 1:
        env = construct_procthor_kitchen_environment(
            PROCTHOR_SEED, remove_duplicate_objects=REMOVE_DUPLICATE_OBJECTS
        )
        return SceneSelection([PROCTHOR_SEED], env.scene.objects, set(env.scene.locations), {})

    scene_filter_kwargs = {k: v for k, v in scene_filter.items() if k != "num_scenes"}
    filtered_scenes = filter_procthor_scenes(num_rooms=NUM_ROOMS_FILTER, **scene_filter_kwargs)

    assert (
        isinstance(scene_filter["num_pickupable_objects"], int) and
        isinstance(scene_filter["num_valid_locations"], int) and
        isinstance(scene_filter["num_scenes"], int)
    )
    exemplar_seed, target_objects, target_locations = find_exemplar_scene(
        filtered_scenes,
        num_objects=scene_filter["num_pickupable_objects"],
        num_locations=scene_filter["num_valid_locations"],
    )
    env = construct_procthor_kitchen_environment(
        exemplar_seed, remove_duplicate_objects=REMOVE_DUPLICATE_OBJECTS
    )
    matching_seeds = list(filtered_scenes)[:scene_filter["num_scenes"]]

    # the exemplar already has the vocabulary, but remapping it too is a
    # harmless no-op. Written to disk later, once the run's keys are known.
    remapped_scenes = remap_scene_objects_and_locations(
        {seed: filtered_scenes[seed] for seed in matching_seeds},
        target_objects, target_locations,
        seed=DATA_GENERATION_SEED,
    )
    return SceneSelection(
        matching_seeds,
        {obj for obj in env.scene.objects if obj.split("_")[0] in target_objects},
        {loc for loc in env.scene.locations if loc.split("_")[0] in target_locations},
        remapped_scenes,
    )


def _compute_scene_keys(selection: SceneSelection) -> dict[int, int]:
    """
    Content-addressed key per scene (see _stable_hash): a hash of the scene's
    actual content and every generation setting that changes what a datum
    contains. It names the pickles, the index CSVs and the remap directory, so
    restart-safety can only match data generated from identical inputs.
    """
    # generic categories (not instance names like fork_21), so the key is the
    # same however the scene's nodes happen to be numbered
    generation_settings = {
        "task_objects": sorted({obj.split("_")[0] for obj in selection.task_objects}),
        "task_locations": sorted({loc.split("_")[0] for loc in selection.task_locations}),
        "num_tasks": NUM_TASKS,
        "remove_duplicate_objects": REMOVE_DUPLICATE_OBJECTS,
        "h_multiplier": H_MULTIPLIER,
        "filter_objects": FILTER_OBJECTS,
        "data_generation_seed": DATA_GENERATION_SEED,
        # a datum now depends on the scene's trajectory, not just the scene
        "environment": "persistent; a visit walks the whole plan but the scene advances one action; reshuffle over original slots and held objects",
        "max_reshuffles_per_visit": MAX_RESHUFFLES_PER_VISIT,
        # datums generated before this fix carry stale positions for objects
        # the robot carried across a move (see _update_held_objects_position)
        "held_object_position_tracks_robot": True,
        # datum filenames count per scene now, not across the worker's scenes
        "datum_counter": "per scene",
        # each scene is one trajectory of NUM_DATUM states, generated by one
        # worker, rather than one shorter trajectory per worker
        "trajectories_per_scene": 1,
        # duplicate objects are removed but every container is kept, and the
        # remap never relies on objects inside ignored containers
        "duplicate_removal": "objects only",
        "remap_skips_ignored_containers": True,
    }
    # a plain scene is fully determined by its seed (the dataset is static),
    # so it contributes None; a remapped one contributes its actual content
    return {
        seed: _stable_hash(
            {"scene": selection.remapped_scenes.get(seed), "settings": generation_settings}
        )
        for seed in selection.matching_seeds
    }


def _run_key(scene_keys: dict[int, int]) -> int:
    """
    Stable identifier for this run's set of scenes (see _stable_hash): the
    same scene_keys always produce the same run_key, so it names the remap
    directory and the run-level metadata files (scene selection, task
    distribution) consistently across a run.
    """
    return _stable_hash(sorted(scene_keys.items()))


def _write_scene_selection(selection: SceneSelection, run_key: int) -> None:
    """
    Persists the scenes chosen for this run - matching_seeds, task_objects,
    task_locations - to disk before any worker is spawned, so the selection
    that produced a run's task distribution is inspectable independent of the
    generated data.
    """
    path = _run_metadata_path("scene_selection", run_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "matching_seeds": selection.matching_seeds,
        "task_objects": sorted(selection.task_objects),
        "task_locations": sorted(selection.task_locations),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Scene selection for this run: {path}")


def _write_task_distribution(
    task_distribution: tuple[Sequence[Goal], list[float]], run_key: int
) -> None:
    """
    Persists this run's task distribution (goals + sampling weights) to disk
    before any worker is spawned, so the tasks a run sampled from are
    inspectable independent of the generated data.
    """
    path = _run_metadata_path("task_distribution", run_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    goals, weights = task_distribution
    payload = {
        "goals": [_goal_to_dict(goal) for goal in goals],
        "weights": list(weights),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Task distribution for this run: {path}")


@contextmanager
def _remaps_enabled(remapped_scenes: dict[int, dict], scene_keys: dict[int, int]):
    """
    Writes this run's remaps to a directory named for its keys (so runs never
    see each other's remaps; an identical config reuses the directory) and
    points REMAP_DIR_ENV_VAR at it for the spawned workers. The variable is
    cleared on exit so the remaps don't leak into anything else this process
    runs. A no-op when there are no remaps.
    """
    if not remapped_scenes:
        yield
        return
    remap_dir = Path(get_procthor_10k_dir()) / "remapped_scenes" / str(_run_key(scene_keys))
    for seed, remapped_scene in remapped_scenes.items():
        _write_remapped_scene_cache(remap_dir, seed, remapped_scene)
    os.environ[REMAP_DIR_ENV_VAR] = str(remap_dir)
    print(f"Remapped scenes for this run: {remap_dir}")
    try:
        yield
    finally:
        os.environ.pop(REMAP_DIR_ENV_VAR, None)


def _prewarm_scene_caches(scene_seeds: list[int], relevant_objects: list[str] | None) -> None:
    """
    Builds the cached navigation data and renders for every scene that lacks
    them, before any worker starts. Building one starts AI2-THOR and holds a
    machine-wide lock, so workers that hit a missing cache would wait on each
    other; doing it up front keeps that out of the worker pool. Built the same
    way a worker loads a scene, so the cache is the one a worker would have made.
    Raises, listing every scene, if any cache couldn't be built.
    """
    missing = [seed for seed in scene_seeds if not scene_cache_path(seed).exists()]
    print(
        f"Scene caches: {len(scene_seeds) - len(missing)}/{len(scene_seeds)} present, "
        f"building {len(missing)}",
        flush=True,
    )
    if missing and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        # the X server ai2thor ships, from this environment's own bin directory
        xorg = Path(sys.executable).parent / "ai2thor-xorg"
        raise RuntimeError(
            f"{len(missing)} scene caches need AI2-THOR, which needs a display. Start one:\n"
            f"  sudo {xorg} --height 4096 --width 4096 start 1\n"
            "  export DISPLAY=:1\n"
            "and stop it once data generation has finished:\n"
            f"  sudo {xorg} stop 1"
        )
    failed: dict[int, str] = {}
    for done, seed in enumerate(missing, start=1):
        start = time.perf_counter()
        try:
            ThorInterface(
                seed, relevant_objects=relevant_objects,
                remove_duplicate_objects=REMOVE_DUPLICATE_OBJECTS,
            )
        except Exception as e:  # noqa: BLE001 - report every failing scene, not just the first
            failed[seed] = f"{type(e).__name__}: {e}"
            print(f"[cache {done}/{len(missing)}] scene {seed} failed: {failed[seed]}", flush=True)
        else:
            elapsed = time.perf_counter() - start
            print(f"[cache {done}/{len(missing)}] scene {seed} built in {elapsed:.0f}s", flush=True)
    if failed:
        details = "\n".join(f"  scene {seed}: {error}" for seed, error in sorted(failed.items()))
        raise RuntimeError(f"could not build {len(failed)} scene cache(s):\n{details}")


def _run_workers(
    scene_keys: dict[int, int],
    task_distribution: tuple[Sequence[Goal], list[float]],
    relevant_objects: list[str] | None,
) -> tuple[int, dict[int, str]]:
    """
    Generates NUM_DATUM data points for every scene, one task per scene in a
    pool of MAX_WORKERS processes: a free worker takes the next scene and
    generates all of its data. A scene that fails doesn't stop the others.
    Returns the total data points written and the failed scenes' errors.
    """
    ctx = multiprocessing.get_context("spawn")
    num_workers = min(MAX_WORKERS, len(scene_keys))
    total_written = 0
    failures: dict[int, str] = {}
    with ProcessPoolExecutor(
        max_workers=num_workers, mp_context=ctx, max_tasks_per_child=SCENES_PER_WORKER
    ) as executor:
        futures = {
            executor.submit(
                _generate_scene_data, scene_seed, scene_key, task_distribution,
                relevant_objects, NUM_DATUM,
            ): scene_seed
            for scene_seed, scene_key in scene_keys.items()
        }
        for done, future in enumerate(as_completed(futures), start=1):
            scene_seed = futures[future]
            try:
                total_written += future.result()
            except BrokenProcessPool as e:
                # a worker died outright (often the out-of-memory killer), which
                # takes the whole pool down with it
                raise RuntimeError(
                    f"a worker process died while generating scene {scene_seed}; "
                    "if it was killed for memory, lower MAX_WORKERS"
                ) from e
            except Exception as e:  # noqa: BLE001 - one bad scene shouldn't stop the run
                failures[scene_seed] = f"{type(e).__name__}: {e}"
                print(f"[{done}/{len(futures)}] scene {scene_seed} failed: {failures[scene_seed]}")
            else:
                index_csv = _csv_path("procthor_data", scene_seed, scene_keys[scene_seed])
                print(f"[{done}/{len(futures)}] scene {scene_seed} done: {index_csv}")
    return total_written, failures


def _generate_scene_data(
    scene_seed: int,
    scene_key: int,
    task_distribution: tuple[Sequence[Goal], list[float]],
    relevant_objects: list[str] | None,
    target: int,
) -> int:
    """
    Generates `target` data points for one scene, one plan per visit. The
    scene's environment is built once, persists across visits, and is released
    when this returns. A visit plans a task sampled from the scene's current
    state (see _plan_next_task), then walks a copy of the environment through
    the whole plan, writing a datum for every state on the way (expected value
    over the task distribution, then the action). The scene itself advances by
    only the plan's first action: a fork taken after it becomes the scene's
    environment for the next visit, and the walked one is discarded. The next
    visit then starts at a state the walk already passed through, so that datum
    isn't written again. Every seed involved is derived from the scene and its
    own counters, so a scene's data doesn't depend on which worker generates it
    or when. Returns the number of data points written.
    """
    run = _open_scene_run(scene_seed, task_distribution, relevant_objects)
    data = run.data
    count = 0
    stalled_visits = 0
    failed_recordings = 0
    _log_memory(scene_seed, "opened", count, target)
    try:
        while count < target:
            plan = _plan_next_task(run, scene_seed)
            if plan is None:
                # nothing plannable even after reshuffling - the next visit reshuffles
                # again, but fail loudly rather than spin forever
                stalled_visits += 1
                if stalled_visits >= MAX_STALLED_VISITS:
                    raise RuntimeError(f"no plannable task in scene {scene_seed}")
                continue
            stalled_visits = 0

            walk_env = data.env
            next_env = None
            second_state_recorded = False
            for step, converted_action in enumerate(plan):
                if count >= target:
                    break

                if step == 0 and run.start_state_recorded:
                    recorded = True  # the previous visit's walk already wrote this state
                else:
                    recorded = _record_datum(
                        walk_env, data, task_distribution, scene_seed, scene_key,
                        file_counter=count,
                    )
                    if recorded:
                        count += 1
                        failed_recordings = 0
                        if count % MEMORY_LOG_EVERY == 0 and count < target:
                            _log_memory(scene_seed, "generating", count, target)
                    else:
                        failed_recordings += 1
                        if failed_recordings >= MAX_FAILED_RECORDINGS:
                            raise RuntimeError(
                                f"{failed_recordings} consecutive states of scene {scene_seed} "
                                "have a task that can't be solved, so no datum could be "
                                f"recorded ({count}/{target} written)"
                            )
                if step == 1:
                    second_state_recorded = recorded

                action = get_action_by_name(walk_env.get_actions(), converted_action.name)
                walk_env.act(action)
                walk_env.update_scene_graph(action)
                if step == 0:
                    next_env = walk_env.fork()

            assert next_env is not None
            data.env = next_env
            run.start_state_recorded = second_state_recorded
    except BaseException:
        # a failing scene is the one whose memory you want to see
        _log_memory(scene_seed, "failed", count, target)
        raise
    _log_memory(scene_seed, "finished", count, target)
    return count


def _log_slow_datum(
    scene_seed: int, file_counter: int, elapsed: float, recorded: bool,
    search_stats: list[SearchStats],
) -> None:
    """One line: the datum's time and outcome, and how much of the expansion
    budget its searches used - the largest one, and in total."""
    outcome = "recorded" if recorded else "unsolvable task, not recorded"
    line = f"[scene {scene_seed}] slow datum {file_counter}: {elapsed:.0f}s ({outcome})"
    if search_stats:
        largest = max(search_stats, key=lambda stats: stats.expanded)
        expanded = sum(stats.expanded for stats in search_stats)
        budget = sum(stats.budget for stats in search_stats)
        line += (
            f"; largest search {largest.expanded}/{largest.budget} expansions "
            f"({largest.fraction:.0%}) for {largest.goal}; "
            f"all {len(search_stats)} searches {expanded}/{budget} ({expanded / budget:.0%})"
        )
    print(line, flush=True)


def _memory_usage_gb() -> tuple[float | None, float]:
    """
    (current, peak) resident memory of this process, in GB. Current comes from
    /proc and is None where that doesn't exist (macOS); peak is the process's
    high-water mark, which with SCENES_PER_WORKER = 1 is this scene's peak.
    """
    # ru_maxrss is in kilobytes on Linux but bytes on macOS
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / 2**30 if sys.platform == "darwin" else peak / 2**20
    try:
        with open("/proc/self/statm", encoding="utf-8") as f:
            resident_pages = int(f.read().split()[1])
        current_gb = resident_pages * os.sysconf("SC_PAGE_SIZE") / 2**30
    except OSError:
        current_gb = None
    return current_gb, peak_gb


def _log_memory(scene_seed: int, stage: str, count: int, target: int) -> None:
    current_gb, peak_gb = _memory_usage_gb()
    current = "n/a" if current_gb is None else f"{current_gb:.2f} GB"
    print(
        f"[scene {scene_seed}] {stage} ({count}/{target} datums): "
        f"memory {current}, peak {peak_gb:.2f} GB (pid {os.getpid()})",
        flush=True,
    )


def _record_datum(
    env: KitchenProcTHOREnvironment,
    data: ExperimentData,
    task_distribution: tuple[Sequence[Goal], list[float]],
    scene_seed: int,
    scene_key: int,
    file_counter: int,
) -> bool:
    """
    Writes the datum for `env`'s current state (its scene graph and expected
    value over the task distribution). True if the datum now exists on disk:
    written here, or already there from a previous run (restart-safety).
    False if some task can't be solved from this state, so no datum exists.
    """
    if datum_pickle_path(scene_seed, scene_key, file_counter).exists():
        return True
    assert data.search_problem.interrupting_task_dist is not None
    start = time.perf_counter()
    search_stats: list[SearchStats] = []
    expected_value, task_costs = compute_interruption_value(
        convert_state_to_positive_preconditions(env.state, data.neg_to_pos_mapping),
        data.search_problem.actions,
        data.search_problem.interrupting_task_dist,
        data.planner_parameters.heuristic_fn,
        search_stats=search_stats,
    )
    elapsed = time.perf_counter() - start
    if elapsed > SLOW_DATUM_SECONDS:
        _log_slow_datum(scene_seed, file_counter, elapsed, expected_value != -1, search_stats)
    if expected_value == -1:
        return False
    write_datum_to_file(
        scene_seed, scene_key, (env.scene.scene_graph, expected_value), file_counter
    )
    if WRITE_OUT_INDIVIDUAL_TASK_COSTS:
        assert task_costs is not None
        write_out_individual_task_costs(
            scene_seed, scene_key, env.scene.scene_graph,
            (task_distribution[0], task_costs), file_counter,
        )
    return True


# --------------------------------------------------------------------------- #
# Helpers shared across pipeline stages
# --------------------------------------------------------------------------- #
def _stable_hash(obj) -> int:
    """
    40-bit int from a canonical-JSON sha256. Unlike hash(), which is
    randomized per process for strings, this is identical across processes
    and machines - which is what lets a datum key double as a restart-safety
    key.
    """
    payload = json.dumps(obj, sort_keys=True).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:10], 16)


# --------------------------------------------------------------------------- #
# Helpers for _write_scene_selection / _write_task_distribution
# --------------------------------------------------------------------------- #
def _run_metadata_path(prefix: str, run_key: int) -> Path:
    """
    Location of a run-level (not per-scene) JSON metadata file, named for run_key (see _run_key).
    """
    return Path(get_procthor_10k_dir()) / "run_metadata" / f"{prefix}_{run_key}.json"


def _fluent_to_dict(fluent: Fluent) -> dict:
    return {"name": fluent.name, "args": fluent.args, "negated": fluent.negated}


def _goal_to_dict(goal: Goal) -> dict:
    """
    A human-readable repr alongside the goal's literals (name/args/negated),
    sorted by repr for a deterministic file across runs with identical goals.
    """
    literals = sorted(goal.get_all_literals(), key=repr)
    return {"repr": repr(goal), "literals": [_fluent_to_dict(f) for f in literals]}


# --------------------------------------------------------------------------- #
# Helpers for _remaps_enabled
# --------------------------------------------------------------------------- #
def _write_remapped_scene_cache(remap_dir: Path, scene_seed: int, remapped_scene: dict) -> None:
    """
    Persists a remapped scene into remap_dir, the directory this run points
    REMAP_DIR_ENV_VAR at, so construct_procthor_kitchen_environment
    (seed=scene_seed, object_seed=None) loads it instead of the raw
    ProcTHOR-10k scene.
    """
    remap_dir.mkdir(parents=True, exist_ok=True)
    with open(remap_dir / f"scene_{scene_seed}.json", "w", encoding="utf-8") as f:
        json.dump(remapped_scene, f)


# --------------------------------------------------------------------------- #
# Helpers for _generate_scene_data
# --------------------------------------------------------------------------- #
def _open_scene_run(
    scene_seed: int,
    task_distribution: tuple[Sequence[Goal], list[float]],
    relevant_objects: list[str] | None,
) -> _SceneRun:
    config = initialize_experiment_config(
        get_example_procthor_goal(), task_distribution, scene_seed, None
    )
    data = initialize_experiment_data(
        config, PlannerMode.MYOPIC, relevant_objects, REMOVE_DUPLICATE_OBJECTS,
        h_multiplier=H_MULTIPLIER,
    )
    # a task naming a category the loaded scene lacks stays generic and can
    # never be solved, so every datum would fail after a full search
    assert config.interrupting_task_dist is not None
    scene_names = (
        data.env.scene.objects | set(data.env.scene.locations)
        | data.env.objects_by_type["category"]
    )
    unmapped = [
        goal for goal in config.interrupting_task_dist[0]
        if not {arg for literal in goal.get_all_literals() for arg in literal.args} <= scene_names
    ]
    if unmapped:
        raise RuntimeError(
            f"scene {scene_seed}: tasks {unmapped} name categories the loaded scene "
            "doesn't have, so they can never be solved"
        )
    return _SceneRun(config, data)


def _plan_next_task(run: _SceneRun, scene_seed: int) -> list[Action] | None:
    """
    A plan for a task sampled from the environment's current state. When every
    task is already satisfied, or none can be planned, re-deals the objects
    over the scene's original container occupancy (held objects included) and
    tries again, up to MAX_RESHUFFLES_PER_VISIT times. The shuffle is seeded by
    (scene, shuffle count), so replaying a scene from a fresh environment
    reproduces the same trajectory.
    """
    for reshuffle in range(MAX_RESHUFFLES_PER_VISIT + 1):
        if reshuffle:
            rng = random.Random(
                _stable_hash([DATA_GENERATION_SEED, scene_seed, run.num_shuffles])
            )
            shuffled = run.data.env.fork()
            shuffled.randomize_object_locations(rng)
            run.data.env = shuffled
            run.start_state_recorded = False  # a state nothing has passed through yet
            run.num_shuffles += 1
        plan = _sample_plan(run, scene_seed)
        if plan is not None:
            return plan
    return None


def _sample_plan(run: _SceneRun, scene_seed: int) -> list[Action] | None:
    """
    Samples a task from the goals not yet satisfied in the environment's
    current state and plans it from that state. Each failure narrows the
    candidate pool, so this terminates. None if every goal is already
    satisfied (a saturated scene) or none can be planned.
    """
    data = run.data
    assert run.config.interrupting_task_dist is not None
    assert data.search_problem.interrupting_task_dist is not None

    # scene_goals are the scene-mapped goals before negative-fluent
    # conversion, so they evaluate against the raw env state; they are
    # index-parallel with the converted goals used for search
    scene_goals, _ = run.config.interrupting_task_dist
    candidate_indices = [
        i for i, goal in enumerate(scene_goals) if not goal.evaluate(data.env.state.fluents)
    ]
    interrupting_goals, _ = data.search_problem.interrupting_task_dist
    initial_state = convert_state_to_positive_preconditions(
        data.env.state, data.neg_to_pos_mapping
    )
    while candidate_indices:
        random.seed(DATA_GENERATION_SEED + scene_seed * SEED_STRIDE + run.attempt)
        run.attempt += 1
        chosen_index = random.choice(candidate_indices)
        data.search_problem.goal = interrupting_goals[chosen_index]
        plan, _, success, _ = astar_search(
            (initial_state, None), data.search_problem, data.planner_parameters
        )
        if success and plan:
            return plan
        candidate_indices.remove(chosen_index)
    return None


def initialize_experiment_config(
    goal: Goal,
    task_distribution: tuple[Sequence[Goal], list[float]],
    procthor_seed: int,
    objects_seed: int | None,
) -> ExperimentConfig:
    """
    Initialize a ExperimentConfig object for randomized object scene generation.
    """
    # in this case, the task_arrival_model won't be used in any meaningful way, because from
    # each state a single task will be solved without considering any possible interruption events.
    task_arrival_fn = partial(
        get_task_arrival_prob, RandomVariableType.CONTINUOUS, -1, float("inf")
    )
    return ExperimentConfig(
        ExperimentSeeds(procthor_seed, object_placement_seed=objects_seed),
        goal,
        task_distribution,
        task_arrival_fn
    )


# --------------------------------------------------------------------------- #
# Helpers for _record_datum
# --------------------------------------------------------------------------- #
def datum_pickle_path(
    scene_seed: int, object_randomization_seed: int, counter: int
) -> Path:
    """
    Canonical on-disk location of a single training-datum pickle. Kept as one
    function so the existence check in `_record_datum` and the write in
    `write_datum_to_file` can never disagree about the filename.

    `object_randomization_seed` is a real object seed on the legacy
    get_randomized_procthor_data path; on the path `_generate_scene_data`
    uses, it carries the scene's content-addressed key instead (see `main`),
    so the existence check only matches data generated from identical inputs.
    """
    return (
        Path(get_procthor_10k_dir()) / "pickles"
        / f"dat_{scene_seed}_{object_randomization_seed}_{counter}.pgz"
    )


def write_datum_to_file(
    scene_seed: int,
    object_randomization_seed: int,
    datum: tuple[SceneGraph, float],
    counter: int,
) -> None:
    """
    Helper function for writing out the training data.
    Writes out the datum as a zipped pickle file and
    adds an entry to the csv file used for tracking all the
    datum files generated. Only the one worker generating a scene writes its
    CSV, so writes never interleave.
    """
    data_filepath = datum_pickle_path(scene_seed, object_randomization_seed, counter)
    data_filepath.parent.mkdir(parents=True, exist_ok=True)
    write_compressed_pickle(data_filepath, datum)
    csv_filepath = _csv_path("procthor_data", scene_seed, object_randomization_seed)
    with open(csv_filepath, 'a', encoding="utf-8") as f:
        f.write(f'{data_filepath}\n')


def write_out_individual_task_costs(
    scene_seed: int,
    object_randomization_seed: int,
    scene_graph: SceneGraph,
    tasks_with_costs: tuple[Sequence[Goal], list[float]],
    counter: int,
) -> None:
    """
    Helper function for writing out the costs of completing tasks from the task
    distribution for a particular procthor scene.
    """
    for idx, (task, task_cost) in enumerate(zip(*tasks_with_costs)):
        datum = (scene_graph, task, task_cost)
        data_filepath = _task_datum_pickle_path(scene_seed, object_randomization_seed, counter, idx)
        data_filepath.parent.mkdir(parents=True, exist_ok=True)
        write_compressed_pickle(data_filepath, datum)
        csv_filepath = _csv_path(
            "procthor_individual_task_data", scene_seed, object_randomization_seed
        )
        with open(csv_filepath, 'a', encoding="utf-8") as f:
            f.write(f'{data_filepath}\n')


def _task_datum_pickle_path(
    scene_seed: int, object_randomization_seed: int, counter: int, task_idx: int
) -> Path:
    return (
        Path(get_procthor_10k_dir()) / "pickles" / "task_costs"
        / f"dat_{scene_seed}_{object_randomization_seed}_{counter}_{task_idx}.pgz"
    )


# --------------------------------------------------------------------------- #
# Index CSVs
# --------------------------------------------------------------------------- #
def _csv_path(prefix: str, scene_seed: int, key: int) -> Path:
    """
    Index-CSV location: {prefix}_{scene_seed}_{key}.csv. `key` is the same
    content-addressed key that names the datum pickles (see `main`), so a CSV
    only ever lists pickles generated from identical inputs; a run with
    different inputs writes a different CSV instead of appending to this one.
    """
    return Path(get_procthor_10k_dir()) / f"{prefix}_{scene_seed}_{key}.csv"


# --------------------------------------------------------------------------- #
# Legacy / standalone - not on main()'s path, kept for reference and possible
# reinstatement (see SEED_STRIDE and datum_pickle_path above)
# --------------------------------------------------------------------------- #
def get_randomized_procthor_data(
    goal: Goal,
    task_distribution: tuple[Sequence[Goal], list[float]],
    procthor_seed: int,
    start_seed: int,
    num_objects: int,
    num_locations: int
) -> tuple[ExperimentData, int]:
    """
    Helper function for ensuring that the randomized object procthor environment
    has all objects/locations of the original scene.
    """
    while True:
        data = initialize_experiment_data(
            initialize_experiment_config(
                goal,
                task_distribution,
                procthor_seed,
                start_seed
            ),
            PlannerMode.MYOPIC,
            extract_relevant_objects(task_distribution[0])if FILTER_OBJECTS else None,
            REMOVE_DUPLICATE_OBJECTS,
            h_multiplier=H_MULTIPLIER
        )

        if (
            len(data.env.scene.objects) == num_objects and
            len(data.env.scene.locations) == num_locations
        ):
            return data, start_seed

        start_seed+=1


if __name__ == "__main__":
    main(prewarm_only="--prewarm-only" in sys.argv[1:])

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
import time
from concurrent.futures import ProcessPoolExecutor
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
from interruption.constants import NUM_TASKS, PROCTHOR_SEED, FILTER_OBJECTS
from interruption.learning.data import write_compressed_pickle
from interruption.planner import astar_search, compute_interruption_value
from interruption.utilities import (
    RandomVariableType, get_task_arrival_prob, extract_relevant_objects,
    filter_procthor_scenes, find_exemplar_scene, remap_scene_objects_and_locations
)
from railroad.core import (
    Action,
    Goal,
    convert_state_to_positive_preconditions,
    get_action_by_name,
)
from railroad.environment.procthor.resources import REMAP_DIR_ENV_VAR, get_procthor_10k_dir
from railroad.environment.procthor.scenegraph import SceneGraph

NUM_DATUM = 500  # datums to generate for each scene seed
DATA_GENERATION_SEED = 37
REMOVE_DUPLICATES = True
H_MULTIPLIER = 1
WRITE_OUT_INDIVIDUAL_TASK_COSTS = True
ONE_ROOM_FILTER = {
    "num_scenes": 22, "num_pickupable_objects": 11, "num_valid_locations": 6,
    "room_types": {"Kitchen"},
}
TWO_ROOM_FILTER = {"num_scenes": 10, "num_pickupable_objects": 20, "num_valid_locations": 10}


# constants used to filter scenes
NUM_ROOMS_FILTER = {1}

# Concurrent AI2-THOR Controller instances this machine sustains without
# throughput degrading (see benchmark_thor_concurrency.py: 8 is the
# peak-throughput point, 16+ regresses, 24 produced an outright hang).
MAX_WORKERS = 2

# Folded into the per-(scene, worker, count) random.seed() used for task
# sampling, so different scenes/workers don't draw identical task
# sequences at the same count. No longer used for object_seed search
# ranges - kept only as get_randomized_procthor_data's own worker_id
# disambiguation, in case that path is reinstated later.
SEED_STRIDE = 100_000

# How many times a scene visit may re-deal a scene's objects (see
# _plan_next_action) before giving up on that visit.
MAX_RESHUFFLES_PER_VISIT = 5


def _stable_hash(obj) -> int:
    """
    40-bit int from a canonical-JSON sha256. Unlike hash(), which is
    randomized per process for strings, this is identical across processes
    and machines - which is what lets a datum key double as a restart-safety
    key.
    """
    payload = json.dumps(obj, sort_keys=True).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:10], 16)


@dataclass(frozen=True)
class SceneSelection:
    matching_seeds: list[int]
    task_objects: set[str]
    task_locations: set[str]
    remapped_scenes: dict[int, dict]  # empty when the plain scene is used as-is


def main():
    """
    Data-generation entrypoint: select scenes -> build the task distribution ->
    key each scene -> generate across workers -> merge the index CSVs.
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
    relevant_objects = extract_relevant_objects(task_distribution[0]) if FILTER_OBJECTS else None

    # NUM_DATUM is per scene: each worker takes a share of it for every scene
    num_workers = min(MAX_WORKERS, NUM_DATUM)
    targets_per_scene = _split_evenly(NUM_DATUM, num_workers)

    with _remaps_enabled(selection.remapped_scenes, scene_keys):
        total_written = _run_workers(
            scene_keys, task_distribution, relevant_objects, targets_per_scene
        )

    _merge_all_csvs(scene_keys, num_workers)

    print(f"Data Generation took: {time.perf_counter() - start: .4f} seconds")
    print(
        f"Wrote {total_written} data points ({NUM_DATUM} for each of {len(scene_keys)} "
        f"scene(s)) across {num_workers} worker(s)"
    )


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
            PROCTHOR_SEED, remove_duplicates=REMOVE_DUPLICATES
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
    env = construct_procthor_kitchen_environment(exemplar_seed, remove_duplicates=REMOVE_DUPLICATES)
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
        "remove_duplicates": REMOVE_DUPLICATES,
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
    }
    # a plain scene is fully determined by its seed (the dataset is static),
    # so it contributes None; a remapped one contributes its actual content
    return {
        seed: _stable_hash(
            {"scene": selection.remapped_scenes.get(seed), "settings": generation_settings}
        )
        for seed in selection.matching_seeds
    }


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
    remap_dir = (
        Path(get_procthor_10k_dir()) / "remapped_scenes"
        / str(_stable_hash(sorted(scene_keys.items())))
    )
    for seed, remapped_scene in remapped_scenes.items():
        _write_remapped_scene_cache(remap_dir, seed, remapped_scene)
    os.environ[REMAP_DIR_ENV_VAR] = str(remap_dir)
    print(f"Remapped scenes for this run: {remap_dir}")
    try:
        yield
    finally:
        os.environ.pop(REMAP_DIR_ENV_VAR, None)


def _split_evenly(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if i < remainder else 0) for i in range(parts)]


def _run_workers(
    scene_keys: dict[int, int],
    task_distribution: tuple[Sequence[Goal], list[float]],
    relevant_objects: list[str] | None,
    targets_per_scene: list[int],
) -> int:
    """
    Every worker gets the full scene list (not a disjoint slice) so a small
    pool, including the single-scene case, never starves workers, and
    generates its share (targets_per_scene[worker_id]) of datums for every
    scene; the shares sum to NUM_DATUM per scene. Workers share scenes freely,
    each with its own in-memory environment; filename collisions are avoided
    by _generate_worker_share's worker-disjoint counter offset, not by scene
    assignment. Returns the total data points written.
    """
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(targets_per_scene), mp_context=ctx) as executor:
        futures = [
            executor.submit(
                _generate_worker_share,
                scene_keys, task_distribution, relevant_objects, worker_id, target,
            )
            for worker_id, target in enumerate(targets_per_scene)
        ]
        return sum(future.result() for future in futures if future.result() != -1)


def _merge_all_csvs(scene_keys: dict[int, int], num_workers: int) -> None:
    for seed, key in scene_keys.items():
        _merge_csv_shards(seed, key, num_workers)
        if WRITE_OUT_INDIVIDUAL_TASK_COSTS:
            _merge_csv_shards(seed, key, num_workers, True)
        print(f"Index CSV for scene {seed}: {_csv_path('procthor_data', seed, key)}")


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


@dataclass
class _SceneRun:
    """A worker's persistent environment for one scene, plus its RNG bookkeeping."""
    config: ExperimentConfig
    data: ExperimentData
    attempt: int = 0       # goal samples drawn so far; never reused, so no sample repeats
    num_shuffles: int = 0  # reshuffles so far; seeds the next one
    # whether the datum for the environment's current state was already written
    # by the previous visit's walk (which passed through it), so it isn't
    # written twice
    start_state_recorded: bool = False


def _open_scene_run(
    scene_seed: int,
    task_distribution: tuple[Sequence[Goal], list[float]],
    relevant_objects: list[str] | None,
) -> _SceneRun:
    config = initialize_experiment_config(
        get_example_procthor_goal(), task_distribution, scene_seed, None
    )
    data = initialize_experiment_data(
        config, PlannerMode.MYOPIC, relevant_objects, REMOVE_DUPLICATES,
        h_multiplier=H_MULTIPLIER,
    )
    return _SceneRun(config, data)


def _sample_plan(run: _SceneRun, scene_seed: int, worker_id: int) -> list[Action] | None:
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
        random.seed(
            DATA_GENERATION_SEED + scene_seed * SEED_STRIDE + worker_id * SEED_STRIDE
            + run.attempt
        )
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


def _plan_next_task(run: _SceneRun, scene_seed: int, worker_id: int) -> list[Action] | None:
    """
    A plan for a task sampled from the environment's current state. When every
    task is already satisfied, or none can be planned, re-deals the objects
    over the scene's original container occupancy (held objects included) and
    tries again, up to MAX_RESHUFFLES_PER_VISIT times. The shuffle is seeded by
    (scene, worker, shuffle count), so replaying a worker from a fresh
    environment reproduces the same trajectory.
    """
    for reshuffle in range(MAX_RESHUFFLES_PER_VISIT + 1):
        if reshuffle:
            rng = random.Random(
                _stable_hash([DATA_GENERATION_SEED, scene_seed, worker_id, run.num_shuffles])
            )
            shuffled = run.data.env.fork()
            shuffled.randomize_object_locations(rng)
            run.data.env = shuffled
            run.start_state_recorded = False  # a state nothing has passed through yet
            run.num_shuffles += 1
        plan = _sample_plan(run, scene_seed, worker_id)
        if plan is not None:
            return plan
    return None


def _record_datum(
    env: KitchenProcTHOREnvironment,
    data: ExperimentData,
    task_distribution: tuple[Sequence[Goal], list[float]],
    scene_seed: int,
    scene_key: int,
    worker_id: int,
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
    expected_value, task_costs = compute_interruption_value(
        convert_state_to_positive_preconditions(env.state, data.neg_to_pos_mapping),
        data.search_problem.actions,
        data.search_problem.interrupting_task_dist,
        data.planner_parameters.heuristic_fn,
    )
    if expected_value == -1:
        return False
    write_datum_to_file(
        scene_seed, scene_key, (env.scene.scene_graph, expected_value), file_counter,
        csv_suffix=worker_id,
    )
    if WRITE_OUT_INDIVIDUAL_TASK_COSTS:
        assert task_costs is not None
        write_out_individual_task_costs(
            scene_seed, scene_key, env.scene.scene_graph,
            (task_distribution[0], task_costs), file_counter, csv_suffix=worker_id,
        )
    return True


def _generate_worker_share(
    scene_keys: dict[int, int],
    task_distribution: tuple[Sequence[Goal], list[float]],
    relevant_objects: list[str] | None,
    worker_id: int,
    target_per_scene: int,
) -> int:
    """
    Generates `target_per_scene` data points for every one of scene_keys' scenes
    in this worker process, cycling through the scenes still short of their
    target, one plan per visit. Each scene's environment is built once, on
    first visit, persists until that scene reaches its target, and is then
    released. A visit plans a task sampled from the scene's current state (see
    _plan_next_task), then walks a copy of the environment through the whole
    plan, writing a datum for every state on the way (expected value over the
    task distribution, then the action). The scene itself advances by only the
    plan's first action: a fork taken after it becomes the scene's environment
    for the next visit, and the walked one is discarded. The next visit then
    starts at a state the walk already passed through, so that datum isn't
    written again. Workers share scenes freely, each with its own
    environments; file_counter (this scene's count, offset per worker) keeps
    pickle filenames distinct across workers. Returns the number of data
    points written, over all scenes.
    """
    scene_seeds = list(scene_keys)
    runs: dict[int, _SceneRun] = {}
    counts = {scene_seed: 0 for scene_seed in scene_seeds}
    scene_idx = worker_id % len(scene_seeds)
    stalled_visits = 0

    while pending := [s for s in scene_seeds if counts[s] < target_per_scene]:
        scene_seed = pending[scene_idx % len(pending)]
        scene_key = scene_keys[scene_seed]
        if scene_seed not in runs:
            runs[scene_seed] = _open_scene_run(scene_seed, task_distribution, relevant_objects)
        run = runs[scene_seed]
        data = run.data

        plan = _plan_next_task(run, scene_seed, worker_id)
        if plan is None:
            # nothing plannable here even after reshuffling - move on, but
            # fail loudly if every remaining scene is stuck rather than spin forever
            stalled_visits += 1
            if stalled_visits >= 2 * len(pending):
                raise RuntimeError(
                    f"worker {worker_id}: no plannable task in scene(s) {pending}"
                )
            scene_idx += 1
            continue
        stalled_visits = 0

        walk_env = data.env
        next_env = None
        second_state_recorded = False
        for step, converted_action in enumerate(plan):
            if counts[scene_seed] >= target_per_scene:
                break

            if step == 0 and run.start_state_recorded:
                recorded = True  # the previous visit's walk already wrote this state
            else:
                recorded = _record_datum(
                    walk_env, data, task_distribution, scene_seed, scene_key, worker_id,
                    file_counter=worker_id * SEED_STRIDE + counts[scene_seed],
                )
                if recorded:
                    counts[scene_seed] += 1
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
        if counts[scene_seed] >= target_per_scene:
            del runs[scene_seed]
        scene_idx += 1

    return sum(counts.values())


def write_out_individual_task_costs(
    scene_seed: int,
    object_randomization_seed: int,
    scene_graph: SceneGraph,
    tasks_with_costs: tuple[Sequence[Goal], list[float]],
    counter: int,
    csv_suffix: int | None = None
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
            "procthor_individual_task_data", scene_seed, object_randomization_seed, csv_suffix
        )
        with open(csv_filepath, 'a', encoding="utf-8") as f:
            f.write(f'{data_filepath}\n')

def _csv_path(prefix: str, scene_seed: int, key: int, shard: int | None = None) -> Path:
    """
    Index-CSV location: {prefix}_{scene_seed}_{key}[_{shard}].csv. `key` is the
    same content-addressed key that names the datum pickles (see `main`), so
    a CSV only ever lists pickles generated from identical inputs; a run with
    different inputs writes a different CSV instead of appending to this one.
    """
    name = f"{prefix}_{scene_seed}_{key}" + ("" if shard is None else f"_{shard}")
    return Path(get_procthor_10k_dir()) / f"{name}.csv"

def _task_datum_pickle_path(
    scene_seed: int, object_randomization_seed: int, counter: int, task_idx: int
) -> Path:
    return (
        Path(get_procthor_10k_dir()) / "pickles" / "task_costs"
        / f"dat_{scene_seed}_{object_randomization_seed}_{counter}_{task_idx}.pgz"
    )

def datum_pickle_path(
    scene_seed: int, object_randomization_seed: int, counter: int
) -> Path:
    """
    Canonical on-disk location of a single training-datum pickle. Kept as one
    function so the existence check in `_generate_worker_share` and the write
    in `write_datum_to_file` can never disagree about the filename.

    `object_randomization_seed` is a real object seed on the legacy
    get_randomized_procthor_data path; on the path `_generate_worker_share`
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
    csv_suffix: int | None = None,
) -> None:
    """
    Helper function for writing out the training data.
    Writes out the datum as a zipped pickle file and
    adds an entry to the csv file used for tracking all the
    datum files generated. csv_suffix routes concurrent workers to separate
    shard files, avoiding interleaved/corrupted writes to one shared CSV.
    """
    data_filepath = datum_pickle_path(scene_seed, object_randomization_seed, counter)
    data_filepath.parent.mkdir(parents=True, exist_ok=True)
    write_compressed_pickle(data_filepath, datum)
    csv_filepath = _csv_path("procthor_data", scene_seed, object_randomization_seed, csv_suffix)
    with open(csv_filepath, 'a', encoding="utf-8") as f:
        f.write(f'{data_filepath}\n')

def _merge_csv_shards(
    scene_seed: int, key: int, num_workers: int, individual_tasks: bool = False
) -> None:
    """
    Concatenates each worker's CSV shard into the scene's combined,
    key-named CSV (see _csv_path).
    """
    prefix = "procthor_individual_task_data" if individual_tasks else "procthor_data"
    combined_path = _csv_path(prefix, scene_seed, key)
    with open(combined_path, 'a', encoding="utf-8") as combined:
        for worker_id in range(num_workers):
            shard_path = _csv_path(prefix, scene_seed, key, worker_id)
            if shard_path.exists():
                combined.write(shard_path.read_text())
                shard_path.unlink()


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
            REMOVE_DUPLICATES,
            h_multiplier=H_MULTIPLIER
        )

        if (
            len(data.env.scene.objects) == num_objects and
            len(data.env.scene.locations) == num_locations
        ):
            return data, start_seed

        start_seed+=1


if __name__ == "__main__":
    main()

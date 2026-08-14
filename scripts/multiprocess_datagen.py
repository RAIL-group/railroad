"""
Data-generation script for expected value over interrupting task distribution for
ProcTHOR environments.
"""
from typing import Sequence
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor
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
    ExperimentMode,
    ExperimentSeeds,
    initialize_experiment_data,
)

# from railroad.environment.procthor.environment import ProcTHOREnvironment
from interruption.learning.data import write_compressed_pickle
from interruption.planner import astar_search, compute_interruption_value
from interruption.utilities import DistributionType, RandomVariableType, TaskArrivalProb
from railroad.core import (
    Goal,
    convert_state_to_positive_preconditions,
    ff_heuristic,
    get_action_by_name,
)
from railroad.environment.procthor.resources import get_procthor_10k_dir
from railroad.environment.procthor.scenegraph import SceneGraph

NUM_DATUM = 1000
DATA_GENERATION_SEED = 37
PROCTHOR_SEED = 201

# Concurrent AI2-THOR Controller instances this machine sustains without
# throughput degrading (see benchmark_thor_concurrency.py: 8 is the
# peak-throughput point, 16+ regresses, 24 produced an outright hang).
MAX_WORKERS = 4

# Disjoint object_seed range handed to each worker so their scene-randomization
# searches (get_randomized_procthor_data) never land on the same seed, and
# wide enough that no worker exhausts its range before hitting its quota.
SEED_STRIDE = 100_000


def main():
    """
    Data-generation entrypoint. Splits NUM_DATUM across up to MAX_WORKERS
    worker processes, each generating an independent share of the data from
    its own disjoint object_seed range.
    """
    start = time.perf_counter()

    num_workers = min(MAX_WORKERS, NUM_DATUM)
    base, remainder = divmod(NUM_DATUM, num_workers)
    targets = [base + (1 if i < remainder else 0) for i in range(num_workers)]

    # get task distribution from alfred tasks
    env = construct_procthor_kitchen_environment(PROCTHOR_SEED)
    task_distribution = get_alfred_task_distribution(env.scene.objects, set(env.scene.locations))
    num_objects = len(env.scene.objects)
    num_locations = len(env.scene.locations)

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(
                _generate_worker_share,
                num_objects,
                num_locations,
                task_distribution,
                worker_id,
                target,
                worker_id * SEED_STRIDE
            )
            for worker_id, target in enumerate(targets)
        ]
        total_written = sum(future.result() for future in futures if future.result() != -1)

    _merge_csv_shards(PROCTHOR_SEED, num_workers)

    print(f"Data Generation took: {time.perf_counter() - start: .4f} seconds")
    print(f"Wrote {total_written} data points across {num_workers} worker(s)")


def _generate_worker_share(
    num_objects: int,
    num_locations: int,
    task_distribution: tuple[Sequence[Goal], list[float]],
    worker_id: int,
    target_count: int,
    seed_start: int
) -> int:
    """
    Generates `target_count` data points in this worker process, using
    object_seeds starting from `seed_start` (a range disjoint from every
    other worker's). Returns the number of data points actually written.
    """
    count = 0
    objects_seed = seed_start

    while count < target_count:
        # find an object randomization seed that includes all objects and locations
        data, objects_seed = get_randomized_procthor_data(
            get_example_procthor_goal(),
            task_distribution,
            PROCTHOR_SEED, objects_seed,
            num_objects,
            num_locations
        )

        # this check isn't really needed since get_randomized_procthor_data requires
        # a valid task_distribution to be passed in, but it removes type checking errors
        if data.search_problem.interrupting_task_dist is None:
            return -1

        random.seed(DATA_GENERATION_SEED + worker_id * SEED_STRIDE + count)
        while True:
            sampled_task_idx = random.randint(0, len(task_distribution[0]))
            # sample with replacement
            # TODO - verify this works how I think it does
            if sampled_task_idx > 0:
                temp = data.search_problem.interrupting_task_dist[0][sampled_task_idx-1]
                data.search_problem.interrupting_task_dist[0][sampled_task_idx-1] = (
                    data.search_problem.goal
                )
                data.search_problem.goal = temp

            initial_state = convert_state_to_positive_preconditions(
                data.env.state, data.neg_to_pos_mapping
            )

            plan, _, success, _ = astar_search(
                (initial_state, None),
                data.search_problem,
                data.planner_parameters
            )
            if success:
                break

        # get expected value over the task distribution for subsequent states
        for converted_action in plan:
            if count >= target_count:
                break

            # compute expected value of state over the interrupting task distribution
            expected_value = compute_interruption_value(
                convert_state_to_positive_preconditions(data.env.state, data.neg_to_pos_mapping),
                data.search_problem.actions,
                data.search_problem.interrupting_task_dist,
                data.planner_parameters.heuristic_fn
            )

            if expected_value != -1:
                # write out the training datum
                write_datum_to_file(
                    PROCTHOR_SEED, objects_seed,
                    (data.env.scene.scene_graph, expected_value), count,
                    csv_suffix=worker_id
                )
                count += 1

            # progress the environment to the next state
            action = get_action_by_name(data.env.get_actions(), converted_action.name)
            data.env.act(action)
            if isinstance(data.env, KitchenProcTHOREnvironment):
                data.env.update_scene_graph(action)

    return count


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
    save_dir = Path(get_procthor_10k_dir()) / "pickles"
    save_dir.mkdir(parents=True, exist_ok=True)
    data_filepath = (
        save_dir / f"dat_{scene_seed}_{object_randomization_seed}_{counter}.pgz"
    )
    write_compressed_pickle(data_filepath, datum)
    csv_name = (
        f"procthor_data_{scene_seed}.csv" if csv_suffix is None
        else f"procthor_data_{scene_seed}_{csv_suffix}.csv"
    )
    csv_filepath = Path(get_procthor_10k_dir()) / csv_name
    with open(csv_filepath, 'a') as f:
        f.write(f'{data_filepath}\n')


def _merge_csv_shards(scene_seed: int, num_workers: int) -> None:
    """
    Concatenates each worker's CSV shard into the combined output file that
    downstream consumers of procthor_data_{scene_seed}.csv already expect.
    """
    data_dir = Path(get_procthor_10k_dir())
    combined_path = data_dir / f"procthor_data_{scene_seed}.csv"
    with open(combined_path, 'a') as combined:
        for worker_id in range(num_workers):
            shard_path = data_dir / f"procthor_data_{scene_seed}_{worker_id}.csv"
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
    task_arrival_model = TaskArrivalProb(
        0, RandomVariableType.CONTINUOUS,
        DistributionType.EXPONENTIAL
    )
    return ExperimentConfig(
        ExperimentSeeds(procthor_seed, object_placement_seed=objects_seed),
        goal,
        task_distribution,
        task_arrival_model,
        ff_heuristic
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
            ExperimentMode.MYOPIC
        )
        start_seed+=1
        if (
            len(data.env.scene.objects) == num_objects and
            len(data.env.scene.locations) == num_locations
        ):
            return data, start_seed


if __name__ == "__main__":
    main()

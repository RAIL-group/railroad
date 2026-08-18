"""
Data-generation script for expected value over interrupting task distribution for
ProcTHOR environments.
"""
from typing import Sequence
import random
import time
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
    ExperimentMode,
    ExperimentData,
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

NUM_DATUM = 1
DATA_GENERATION_SEED = 37

def main():
    """
    Data-generation function.
    """
    start = time.perf_counter()

    # data generation settings
    procthor_seed = 201

    count = 0
    objects_seed = 0

    # get task distribution from alfred tasks
    env = construct_procthor_kitchen_environment(procthor_seed)
    task_distribution = get_alfred_task_distribution(
        env.scene.objects,
        set(env.scene.locations),
        one_object_per_taskdist=True
    )

    # generic_task_distribution = get_example_procthor_task_distribution(1)
    num_objects = len(env.scene.objects)
    num_locations = len(env.scene.locations)

    while count < NUM_DATUM:
        # find an object randomization seed that includes all objects and locations
        data, objects_seed = get_randomized_procthor_data(
            get_example_procthor_goal(),
            task_distribution,
            procthor_seed, objects_seed,
            num_objects,
            num_locations
        )

        assert data.search_problem.interrupting_task_dist is not None

        random.seed(DATA_GENERATION_SEED+count)
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
                    procthor_seed, objects_seed,
                    (data.env.scene.scene_graph, expected_value), count
                )
                count+=1

            # progress the environment to the next state
            action = get_action_by_name(data.env.get_actions(), converted_action.name)
            data.env.act(action)
            if isinstance(data.env, KitchenProcTHOREnvironment):
                data.env.update_scene_graph(action)

    print(f"Data Generation took: {time.perf_counter() - start: .4f} seconds")


def write_datum_to_file(
    scene_seed: int,
    object_randomization_seed: int,
    datum: tuple[SceneGraph, float],
    counter: int
) -> None:
    """
    Helper function for writing out the training data.
    Writes out the datum as a zipped pickle file and 
    adds an entry to the csv file used for tracking all the
    datum files generated.
    """
    save_dir = Path(get_procthor_10k_dir()) / "pickles"
    save_dir.mkdir(parents=True, exist_ok=True)
    data_filepath = (
        save_dir / f"dat_{scene_seed}_{object_randomization_seed}_{counter}.pgz"
    )
    write_compressed_pickle(data_filepath, datum)
    csv_filepath = (
        Path(get_procthor_10k_dir()) / f"procthor_data_{scene_seed}.csv"
    )
    with open(csv_filepath, 'a') as f:
        f.write(f'{data_filepath}\n')


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

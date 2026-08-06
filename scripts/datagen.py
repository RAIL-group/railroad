"""
Data-generation script for expected value over interrupting task distribution for
ProcTHOR environments.
"""
from typing import Sequence
import random
import time
from functools import partial
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
    task_distribution = get_alfred_task_distribution(env.scene.objects, set(env.scene.locations))

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

        random.seed(DATA_GENERATION_SEED+count)
        while True:
            sampled_task_idx = random.randint(0, len(task_distribution[0]))
            task = (
                data.converted_goal if sampled_task_idx == 0
                else data.converted_interrupting_task_dist[0][sampled_task_idx-1]
            )
            plan, _, success = astar_search(
                convert_state_to_positive_preconditions(data.env.state, data.neg_to_pos_mapping),
                task,
                data.converted_actions,
                None,
                ff_heuristic,
                0,
                None,
                0,
                print_trace=False
            )
            if success:
                break


        # data.env.act(get_action_by_name(data.env.get_actions(), "move robot1 start_loc fridge_4"))

        # expected_value = compute_interruption_value(
        #     convert_state_to_positive_preconditions(data.env.state, data.neg_to_pos_mapping),
        #     data.converted_actions,
        #     data.converted_interrupting_task_dist,
        #     partial(
        #         ff_heuristic, lambda_add=0.5, lambda_max=0, lambda_ff=0.5, at_implies_found=True
        #     ),
        #     0
        # )
        # count+=1


        # get expected value over the task distribution for subsequent states
        for converted_action in plan:
            # compute expected value of state over the interrupting task distribution
            expected_value = compute_interruption_value(
                convert_state_to_positive_preconditions(data.env.state, data.neg_to_pos_mapping),
                data.converted_actions,
                data.converted_interrupting_task_dist,
                partial(
                    ff_heuristic, lambda_add=0.5, lambda_max=0, lambda_ff=0.5, at_implies_found=True
                ),
                0
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
        goal,
        task_distribution,
        True,
        task_arrival_model,
        ExperimentSeeds(procthor_seed, object_placement_seed=objects_seed)
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
            )
        )
        start_seed+=1
        if (
            len(data.env.scene.objects) == num_objects and
            len(data.env.scene.locations) == num_locations
        ):
            return data, start_seed


if __name__ == "__main__":
    main()

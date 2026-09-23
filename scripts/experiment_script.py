from functools import partial
from interruption.constants import (
    MODEL_NAME, EXPECTED_TIME_NEXT_ARRIVAL,
    PROCTHOR_SEED, OBJ_PLACEMENT_SEED, FILTER_OBJECTS, REMAPPED_SCENES_HASH
)
from interruption.environments import (
    get_scene_task_distribution,
    # get_example_procthor_goal,
    # get_example_procthor_task_distribution,
)
from interruption.experiments import (
    ExperimentConfig,
    ExperimentSeeds,
    run_experiment,
)
from interruption.planning_framework import PlannerMode
from interruption.utilities import (
    RandomVariableType, randomize_task_distribution_order, get_task_arrival_prob,
    extract_relevant_objects, use_remapped_scenes
)
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE
from railroad.core import LiteralGoal, Fluent as F

# constants
MODEL_PATH = DEFAULT_RESOURCES_BASE / "models"
RANDOMIZE_TASK_SEQUENCE = True
RUN_IDX_SEED = 2

def main(randomize_order: bool = False, filter_objects: bool = False):
    # evaluate on the scenes and task distribution of a data-generation run.
    # Must come before any environment is built.
    remap_dir = (
        use_remapped_scenes(REMAPPED_SCENES_HASH, PROCTHOR_SEED) if REMAPPED_SCENES_HASH else None
    )
    seeds = ExperimentSeeds(
        procthor_seed=PROCTHOR_SEED,
        experiment_seed=140 + RUN_IDX_SEED, 
        # an object seed makes ThorInterface skip the remap, so a remapped
        # scene keeps the object placement it was generated with
        object_placement_seed=None if remap_dir else OBJ_PLACEMENT_SEED,
        task_sample_seed=75
    )
    task_arrival_fn = partial(
        get_task_arrival_prob, RandomVariableType.CONTINUOUS,
        -1, EXPECTED_TIME_NEXT_ARRIVAL[0]
    )

    # get task distribution from alfred dataset used during training
    task_distribution = get_scene_task_distribution(seeds.procthor_seed, remap_dir)
    if randomize_order:
        current_goal, task_distribution = randomize_task_distribution_order(
            task_distribution, seeds.task_sample_seed
        )

        # for smaller scale experiments, just reorder the task sequence
        task_sequence = (
            task_distribution[0][:4],
            task_distribution[1][:4]
        )
        _, task_sequence = randomize_task_distribution_order(task_sequence, RUN_IDX_SEED)

        task_distribution[0][:4] = task_sequence[0]
        task_distribution[1][:4] = task_sequence[1]
    else:
        # both apple and pan are located at countertop3
        current_goal = task_distribution[0][0] # apple at fridge
        task_distribution = (list(task_distribution[0]), task_distribution[1])
        tmp_goal = task_distribution[0][0]
        task_distribution[0][0] = task_distribution[0][3] # pan at fridge
        task_distribution[0][3] = tmp_goal

    config = ExperimentConfig(
        seeds,
        current_goal,
        task_distribution,
        task_arrival_fn,
        MODEL_PATH / MODEL_NAME,
        num_task_sequence=5,
        augment_task=True,
        retry_with_subgoals=True
    )

    run_experiment(
        config,
        PlannerMode.ANTICIPATORY_PLANNING,
        show_plot=False,
        remove_duplicates=True,
        benchmark_flag=False,
        relevant_objects= extract_relevant_objects(task_distribution[0]) if filter_objects else None
    )

if __name__ == "__main__":
    main(RANDOMIZE_TASK_SEQUENCE, FILTER_OBJECTS)

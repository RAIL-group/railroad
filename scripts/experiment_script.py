from functools import partial
from interruption.constants import MODEL_NAME
from interruption.environments import (
    construct_procthor_kitchen_environment,
    get_alfred_task_distribution,
    # get_example_procthor_goal,
    # get_example_procthor_task_distribution,
)
from interruption.experiments import (
    ExperimentConfig,
    ExperimentMode,
    ExperimentSeeds,
    run_experiment,
)
from interruption.utilities import (
    RandomVariableType, randomize_task_distribution_order, get_task_arrival_prob, 
    calibrate_beta_parameter
)
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE

# constants
MODEL_PATH = DEFAULT_RESOURCES_BASE / "models"
RANDOMIZE_TASK_SEQUENCE = False
NUM_TASKS = 11

def main(randomize_order: bool = False):

    seeds = ExperimentSeeds(
        procthor_seed=201, experiment_seed=20, object_placement_seed=19, task_sample_seed=75
    )
    task_arrival_fn = partial(
        get_task_arrival_prob, RandomVariableType.CONTINUOUS,
        -1, calibrate_beta_parameter(0.5, 76.998)
    )

    # get task distribution from alfred dataset used during training
    env = construct_procthor_kitchen_environment(seeds.procthor_seed, remove_duplicates=True)
    task_distribution = get_alfred_task_distribution(
        env.scene.objects,
        set(env.scene.locations),
        size=NUM_TASKS,
        one_object_per_taskdist=True
    )
    if randomize_order:
        current_goal, task_distribution = randomize_task_distribution_order(
            task_distribution, seeds.task_sample_seed
        )
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
        augment_task=True
    )

    run_experiment(
        config, ExperimentMode.ANTICIPATORY_PLANNING, show_plot=True, remove_duplicates=True
    )

if __name__ == "__main__":
    main(RANDOMIZE_TASK_SEQUENCE)

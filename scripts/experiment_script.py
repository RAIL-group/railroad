from functools import partial
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
    RandomVariableType, randomize_task_distribution_order, get_task_arrival_prob
)
from railroad.core import ff_heuristic
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE

# constants
MODEL_PATH = DEFAULT_RESOURCES_BASE / "models"
RANDOMIZE_TASK_SEQUENCE = True

def main(randomize_order: bool = False):
    model_name = "best_model_experiment10_val.pt"

    seeds = ExperimentSeeds(
        procthor_seed=201, experiment_seed=20, object_placement_seed=21, task_sample_seed=75
    )
    task_arrival_fn = partial(
        get_task_arrival_prob, RandomVariableType.CONTINUOUS, -1, 5
    )

    # get task distribution from alfred dataset used during training
    env = construct_procthor_kitchen_environment(seeds.procthor_seed, remove_duplicates=True)
    task_distribution = get_alfred_task_distribution(
        env.scene.objects,
        set(env.scene.locations),
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
        ff_heuristic,
        MODEL_PATH / model_name,
        num_task_sequence=5,
        augment_task=False
    )

    run_experiment(
        config, ExperimentMode.INTERRUPTION, show_plot=True, remove_duplicates=True
    )

if __name__ == "__main__":
    main(RANDOMIZE_TASK_SEQUENCE)

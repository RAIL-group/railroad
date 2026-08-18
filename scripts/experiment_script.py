from interruption.environments import (
    construct_procthor_kitchen_environment,
    get_alfred_task_distribution,
    get_example_procthor_goal,
    # get_example_procthor_task_distribution,
)
from interruption.experiments import (
    ExperimentConfig,
    ExperimentMode,
    ExperimentSeeds,
    TaskArrivalProb,
    run_experiment,
)
from interruption.utilities import (
    DistributionType, RandomVariableType, randomize_task_distribution_order
)
from railroad.core import ff_heuristic
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE

# constants
MODEL_PATH = DEFAULT_RESOURCES_BASE / "models"
RANDOMIZE_TASK_SEQUENCE = True

def main(randomize_order: bool = False):
    model_name = "best_model_experiment8.pt"

    seeds = ExperimentSeeds(
        procthor_seed=201, experiment_seed=20, object_placement_seed=None, task_sample_seed=91
    )
    task_arrival_model = TaskArrivalProb(
        0, RandomVariableType.CONTINUOUS,
        DistributionType.EXPONENTIAL
    )
    # get task distribution from alfred dataset used during training
    env = construct_procthor_kitchen_environment(seeds.procthor_seed)
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
        current_goal = get_example_procthor_goal()


    config = ExperimentConfig(
        seeds,
        current_goal,
        task_distribution,
        task_arrival_model,
        ff_heuristic,
        MODEL_PATH / model_name,
        num_task_sequence=2
    )

    run_experiment(
        config, ExperimentMode.ANTICIPATORY_PLANNING, show_plot=True
    )

if __name__ == "__main__":
    main(RANDOMIZE_TASK_SEQUENCE)

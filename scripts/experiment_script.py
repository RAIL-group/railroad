from interruption.environments import (
    get_example_procthor_goal,
    get_example_procthor_task_distribution,
)
from interruption.experiments import (
    ExperimentConfig,
    ExperimentSeeds,
    TaskArrivalProb,
    run_experiment,
)
from interruption.utilities import DistributionType, RandomVariableType


def main():
    seeds = ExperimentSeeds(procthor_seed=201, experiment_seed=20, object_placement_seed=0)
    task_arrival_model = TaskArrivalProb(
        0, RandomVariableType.CONTINUOUS,
        DistributionType.EXPONENTIAL
    )
    config = ExperimentConfig(
        get_example_procthor_goal(),
        get_example_procthor_task_distribution(index=1),
        False,
        task_arrival_model,
        seeds
    )
    run_experiment(config, show_plot=True, save_video="test1.mp4")

if __name__ == "__main__":
    main()

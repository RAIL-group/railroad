"""
Real-time Task Stream Planning Benchmark

Wraps `run_experiment` (experiments.py) as a railroad.bench benchmark so
interruption-planning sweeps can be run in parallel and tracked in
MLflow / viewed via `railroad benchmarks dashboard`.
"""
import random
import itertools
from typing import Any

from railroad.bench import BenchmarkCase, benchmark
from railroad.core import ff_heuristic
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE

from ..environments import (
    construct_procthor_kitchen_environment,
    get_alfred_task_distribution,
    get_example_procthor_goal,
    # get_example_procthor_task_distribution,
)
from ..experiments import ExperimentConfig, ExperimentSeeds, run_experiment, ExperimentMode
from ..utilities import (
    DistributionType, RandomVariableType, TaskArrivalProb, randomize_task_distribution_order
)

# CONSTANTS
MODEL_NAME = "best_model_experiment10_val.pt"
EXPERIMENT_REPEATS = 32


def _get_cases() -> list[dict[str, Any]]:
    """
    Helper function for computing the benchmark cases for both the baseline and 
    the interruption-based planner experiments in procthor environments.
    """
    return [
        {
            "procthor_seed": 201,
            "task_dist_idx": 0,
            "interruption_prob": interruption_prob,
            "interruption_seed": seed,
            "num_task_sequence": num_task_sequence,
            "randomize_task_sequence": True
        }
        for (interruption_prob, seed), num_task_sequence in itertools.product(
            zip(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                [140, 42, 240, 57, 1096, 4065, 720, 391, 875]
            ),
            [5]
        )
    ]


def _setup_experiment_config(
        case: BenchmarkCase,
        experiment_mode: ExperimentMode
    ) -> ExperimentConfig:
    """
    Helper function for setting up the experimental config for both the 
    baseline and interruption-based planner benchmark experiments.
    """
    seeds = ExperimentSeeds(
        case.params["procthor_seed"],
        case.params["interruption_seed"] + case.repeat_idx,
        75, # keep fixed for right now
        object_placement_seed=19
    )
    task_arrival_model = TaskArrivalProb(
        case.params["interruption_prob"], RandomVariableType.CONTINUOUS,
        DistributionType.EXPONENTIAL
    )
    # get task distribution from alfred dataset used during training
    env = construct_procthor_kitchen_environment(
        seeds.procthor_seed, remove_duplicates=True
    )
    task_distribution = get_alfred_task_distribution(
        env.scene.objects,
        set(env.scene.locations),
        one_object_per_taskdist=True
    )
    current_goal = get_example_procthor_goal()
    if case.params["randomize_task_sequence"]:
        current_goal, task_distribution = randomize_task_distribution_order(
            task_distribution, seeds.task_sample_seed
        )
        # for smaller scale experiments, just reorder the task sequence
        task_distribution = (
            list(task_distribution[0][:case.params["num_task_sequence"]-1]),
            list(task_distribution[1][:case.params["num_task_sequence"]-1])
        )
        rng = random.Random(case.repeat_idx)
        tasks, probs = task_distribution
        idxes = rng.sample(range(len(tasks)), k=len(tasks))
        task_distribution = ([tasks[i] for i in idxes], [probs[i] for i in idxes])

    model_path = (
        DEFAULT_RESOURCES_BASE / f"models/{MODEL_NAME}"
        if experiment_mode in [ExperimentMode.INTERRUPTION, ExperimentMode.ANTICIPATORY_PLANNING]
        else ""
    )

    config = ExperimentConfig(
        seeds,
        current_goal,
        task_distribution,
        task_arrival_model,
        ff_heuristic,
        model_path,
        case.params["num_task_sequence"]
    )
    return config


@benchmark(
    name="procthor_interruption",
    description=(
        "Evaluates the interruption-based planner across "
        "task-arrival probabilities in specified procthor environments."
    ),
    tags=["interruption", "procthor"],
    timeout=600.0,
    repeat=EXPERIMENT_REPEATS,
)
def bench_interruption_kitchen(case: BenchmarkCase):
    """
    Wrapper function to evaluate the interruption-based planner on procthor kitchen
    environments. 
    """
    config = _setup_experiment_config(case, ExperimentMode.INTERRUPTION)
    return run_experiment(config, ExperimentMode.INTERRUPTION, True, True)

bench_interruption_kitchen.add_cases(_get_cases())


@benchmark(
    name="myopic_procthor_interruption",
    description=(
        "Evaluates the myopic planner across "
        "task-arrival probabilities in specified procthor environments."
    ),
    tags=["interruption", "procthor", "myopic"],
    timeout=600.0,
    repeat=EXPERIMENT_REPEATS,
)
def bench_myopic_interruption_kitchen(case: BenchmarkCase):
    """
    Wrapper function to evaluate the interruption-based planner on procthor kitchen
    environments. 
    """
    config = _setup_experiment_config(case, ExperimentMode.MYOPIC)
    return run_experiment(config, ExperimentMode.MYOPIC, True, True)

bench_myopic_interruption_kitchen.add_cases(_get_cases())


@benchmark(
    name="ap_procthor_interruption",
    description=(
        "Evaluates the anticipatory planning planner across "
        "task-arrival probabilities in specified procthor environments."
    ),
    tags=["interruption", "procthor", "ap"],
    timeout=600.0,
    repeat=EXPERIMENT_REPEATS,
)
def bench_ap_interruption_kitchen(case: BenchmarkCase):
    """
    Wrapper function to evaluate the interruption-based planner on procthor kitchen
    environments. 
    """
    config = _setup_experiment_config(case, ExperimentMode.ANTICIPATORY_PLANNING)
    return run_experiment(config, ExperimentMode.ANTICIPATORY_PLANNING, True, True)

bench_ap_interruption_kitchen.add_cases(_get_cases())

"""
Real-time Task Stream Planning Benchmark

Wraps `run_experiment` (experiments.py) as a railroad.bench benchmark so
interruption-planning sweeps can be run in parallel and tracked in
MLflow / viewed via `railroad benchmarks dashboard`.
"""

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
from ..utilities import DistributionType, RandomVariableType, TaskArrivalProb


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
            "num_task_sequence": num_task_sequence
        }
        for (interruption_prob, seed), num_task_sequence in itertools.product(
            zip(
                [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
                [140, 42, 240, 57, 630, 175]
            ),
            [2]
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
        object_placement_seed=None
    )
    task_arrival_model = TaskArrivalProb(
        case.params["interruption_prob"], RandomVariableType.CONTINUOUS,
        DistributionType.EXPONENTIAL
    )
    # get task distribution from alfred dataset used during training
    env = construct_procthor_kitchen_environment(seeds.procthor_seed)
    task_distribution = get_alfred_task_distribution(env.scene.objects, set(env.scene.locations))

    model_path = (
        DEFAULT_RESOURCES_BASE / "models/best_model_experiment5.pt"
        if experiment_mode == ExperimentMode.INTERRUPTION
        else ""
    )

    config = ExperimentConfig(
        seeds,
        get_example_procthor_goal(),
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
    repeat=32,
)
def bench_interruption_kitchen(case: BenchmarkCase):
    """
    Wrapper function to evaluate the interruption-based planner on procthor kitchen
    environments. 
    """
    config = _setup_experiment_config(case, ExperimentMode.INTERRUPTION)
    return run_experiment(config, ExperimentMode.INTERRUPTION, True)

bench_interruption_kitchen.add_cases(_get_cases())


@benchmark(
    name="myopic_procthor_interruption",
    description=(
        "Evaluates the myopic planner across "
        "task-arrival probabilities in specified procthor environments."
    ),
    tags=["interruption", "procthor", "myopic"],
    timeout=600.0,
    repeat=100,
)
def bench_myopic_interruption_kitchen(case: BenchmarkCase):
    """
    Wrapper function to evaluate the interruption-based planner on procthor kitchen
    environments. 
    """
    config = _setup_experiment_config(case, ExperimentMode.MYOPIC)
    return run_experiment(config, ExperimentMode.MYOPIC, True)

bench_myopic_interruption_kitchen.add_cases(_get_cases())


# TODO - add bench mark for anticipatory planner

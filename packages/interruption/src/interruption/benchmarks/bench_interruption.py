"""
Real-time Task Stream Planning Benchmark

Wraps `run_experiment` (experiments.py) as a railroad.bench benchmark so
interruption-planning sweeps can be run in parallel and tracked in
MLflow / viewed via `railroad benchmarks dashboard`.
"""

import itertools
from typing import Any

from railroad.bench import BenchmarkCase, benchmark

from ..environments import (
    get_example_procthor_goal,
    get_example_procthor_task_distribution,
)
from ..experiments import ExperimentConfig, ExperimentSeeds, run_experiment
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
                [0.0, 0.1, 0.2],
                [140, 42, 240]
            ),
            [2]
        )
    ]


def _setup_experiment_config(case: BenchmarkCase, baseline_flag: bool) -> ExperimentConfig:
    """
    Helper function for setting up the experimental config for both the 
    baseline and interruption-based planner benchmark experiments.
    """
    seeds = ExperimentSeeds(
        case.params["procthor_seed"],
        case.params["interruption_seed"] + case.repeat_idx
    )
    task_arrival_model = TaskArrivalProb(
        case.params["interruption_prob"], RandomVariableType.CONTINUOUS,
        DistributionType.EXPONENTIAL
    )
    config = ExperimentConfig(
        get_example_procthor_goal(),
        get_example_procthor_task_distribution(case.params["task_dist_idx"]),
        baseline_flag,
        task_arrival_model,
        seeds,
        case.params["num_task_sequence"],
        True
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
    config = _setup_experiment_config(case, False)
    return run_experiment(config)

bench_interruption_kitchen.add_cases(_get_cases())


@benchmark(
    name="myopic_procthor_interruption",
    description=(
        "Evaluates the myopic planner across "
        "task-arrival probabilities in specified procthor environments."
    ),
    tags=["interruption", "procthor", "myopic"],
    timeout=600.0,
    repeat=32,
)
def bench_myopic_interruption_kitchen(case: BenchmarkCase):
    """
    Wrapper function to evaluate the interruption-based planner on procthor kitchen
    environments. 
    """
    config = _setup_experiment_config(case, True)
    return run_experiment(config)

bench_myopic_interruption_kitchen.add_cases(_get_cases())

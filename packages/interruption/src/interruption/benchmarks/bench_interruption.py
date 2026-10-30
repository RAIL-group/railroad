"""
Real-time Task Stream Planning Benchmark

Wraps `run_experiment` (experiments.py) as a railroad.bench benchmark so
interruption-planning sweeps can be run in parallel and tracked in
MLflow / viewed via `railroad benchmarks dashboard`.
"""
import math
from collections.abc import Callable
from typing import Any

from railroad.bench import BenchmarkCase, benchmark
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE

from ..constants import (
    MODEL_NAME, EXPERIMENT_REPEATS, AUGMENT_TASK, ARRIVAL_EXCEEDANCE_FRACTIONS,
    INTERRUPTION_SEEDS, PROCTHOR_SEED, TIMEOUT,
)
from ..cost_formulations import NO_INTERRUPTION
from ..experiments import run_experiment
from ..planners import (
    AnticipatoryPlanner, InterruptionAPPlanner, MyopicPlanner, load_ev_model,
)
from ..planning_framework import Planner
from ..scenarios import Scenario

# number of tasks in each sampled task sequence
NUM_TASK_SEQUENCE = 5

# the learned expected-value model of the planners that use one
MODEL_PATH = DEFAULT_RESOURCES_BASE / "models" / MODEL_NAME


def _get_cases() -> list[dict[str, Any]]:
    """
    Helper function for computing the benchmark cases for both the baseline and 
    the interruption-based planner experiments in procthor environments.
    """
    return [
        {
            "procthor_seed": PROCTHOR_SEED,
            "arrival_exceedance": arrival_exceedance,
            "interruption_seed": seed,
            "num_task_sequence": NUM_TASK_SEQUENCE,
            "randomize_task_sequence": True,
            "augment_task": AUGMENT_TASK
        }
        # each arrival rate is paired with its own seed
        for arrival_exceedance, seed in zip(
            ARRIVAL_EXCEEDANCE_FRACTIONS, INTERRUPTION_SEEDS, strict=True
        )
    ]


def _interruption_ap(case: BenchmarkCase) -> Planner:
    # augmented tasks are solved undiscounted; otherwise each step is discounted
    # by the chance no task arrives during it
    discount = 1.0 if case.params["augment_task"] else NO_INTERRUPTION
    return InterruptionAPPlanner(ev_model=load_ev_model(MODEL_PATH), discount=discount)


def _anticipatory(case: BenchmarkCase) -> Planner:
    return AnticipatoryPlanner(ev_model=load_ev_model(MODEL_PATH))


def _myopic(case: BenchmarkCase) -> Planner:
    return MyopicPlanner()


def _scenario(case: BenchmarkCase) -> Scenario:
    """
    The scenario of one repeat of a case. Built in the worker process, so the
    remapped scene is set up there before any environment is built, and
    registering the benchmarks doesn't need the run's task-cost quantiles on disk.
    """
    return Scenario.procthor(
        case.params["procthor_seed"],
        arrival_exceedance=case.params["arrival_exceedance"],
        experiment_seed=case.params["interruption_seed"] + case.repeat_idx,
        task_sample_seed=case.repeat_idx if case.params["randomize_task_sequence"] else None,
        num_task_sequence=case.params["num_task_sequence"],
        augment_task=case.params["augment_task"],
    )


def _run(case: BenchmarkCase, make_planner: Callable[[BenchmarkCase], Planner]) -> dict:
    """
    Runs a single benchmark case with the planner make_planner builds for it.
    """
    scenario = _scenario(case)
    result = run_experiment(scenario, make_planner(case), record=True)
    # logged as a metric; inf (no interruptions) is left out
    assert scenario.time_between_arrivals is not None
    if math.isfinite(scenario.time_between_arrivals):
        result["time_between_arrivals"] = scenario.time_between_arrivals
    return result


def _register(
    make_planner: Callable[[BenchmarkCase], Planner], name: str, label: str, tag: str
):
    """
    Registers a benchmark that evaluates the given planner on the shared
    benchmark cases.
    """
    @benchmark(
        name=name,
        description=(
            f"Evaluates the {label} planner across "
            "task-arrival probabilities in specified procthor environments."
        ),
        tags=["interruption", "procthor", tag, "interruption_experiments"],
        timeout=TIMEOUT,
        repeat=EXPERIMENT_REPEATS,
    )
    def bench(case: BenchmarkCase):
        return _run(case, make_planner)

    bench.add_cases(_get_cases())
    return bench


bench_interruption_ap = _register(
    _interruption_ap, "procthor_interruption_ap", "interruption-ap", "interruption_ap"
)
bench_myopic = _register(_myopic, "procthor_myopic", "myopic", "myopic")
bench_ap = _register(_anticipatory, "procthor_ap", "anticipatory planning", "ap")

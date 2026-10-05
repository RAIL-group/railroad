"""
Real-time Task Stream Planning Benchmark

Wraps `run_experiment` (experiments.py) as a railroad.bench benchmark so
interruption-planning sweeps can be run in parallel and tracked in
MLflow / viewed via `railroad benchmarks dashboard`.
"""
from functools import partial
from pathlib import Path
from typing import Any

from railroad.bench import BenchmarkCase, benchmark
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE

from ..constants import (
    MODEL_NAME, EXPERIMENT_REPEATS, AUGMENT_TASK, EXPECTED_TIME_NEXT_ARRIVAL,
    INTERRUPTION_SEEDS, PROCTHOR_SEED, OBJ_PLACEMENT_SEED, FILTER_OBJECTS, TIMEOUT,
    RETRY_WITH_SUBGOALS, REMAPPED_SCENES_HASH
)
from ..environments import (
    get_example_procthor_goal,
    get_scene_task_distribution,
)
from ..experiments import ExperimentConfig, ExperimentSeeds, run_experiment
from ..planning_framework import PlannerMode
from ..utilities import (
    RandomVariableType, randomize_task_distribution_order, get_task_arrival_prob,
    extract_relevant_objects, use_remapped_scenes
)

# number of tasks in each sampled task sequence
NUM_TASK_SEQUENCE = 5

# planner modes that need the learned expected-value model
_LEARNED_MODES = {
    PlannerMode.INTERRUPTION,
    PlannerMode.ANTICIPATORY_PLANNING,
    PlannerMode.INTERRUPTION_AP,
}


def _get_cases() -> list[dict[str, Any]]:
    """
    Helper function for computing the benchmark cases for both the baseline and 
    the interruption-based planner experiments in procthor environments.
    """
    return [
        {
            "procthor_seed": PROCTHOR_SEED,
            "time_between_arrivals": time_between_arrivals,
            "interruption_seed": seed,
            "num_task_sequence": NUM_TASK_SEQUENCE,
            "randomize_task_sequence": True,
            "augment_task": AUGMENT_TASK
        }
        # each arrival rate is paired with its own seed
        for time_between_arrivals, seed in zip(
            EXPECTED_TIME_NEXT_ARRIVAL, INTERRUPTION_SEEDS, strict=True
        )
    ]


def _build_seeds(case: BenchmarkCase, remap_dir: Path | None) -> ExperimentSeeds:
    """
    Helper function for deriving the per-repeat experiment seeds of a case.
    """
    return ExperimentSeeds(
        procthor_seed=case.params["procthor_seed"],
        experiment_seed=case.params["interruption_seed"] + case.repeat_idx,
        task_sample_seed=case.repeat_idx,
        # an object seed makes ThorInterface skip the remap, so a remapped
        # scene keeps the object placement it was generated with
        object_placement_seed=None if remap_dir else OBJ_PLACEMENT_SEED
    )


def _model_path(experiment_mode: PlannerMode) -> Path | str:
    """
    Helper function that returns the path of the learned expected-value model
    for the planner modes that use it, and an empty string otherwise.
    """
    if experiment_mode in _LEARNED_MODES:
        return DEFAULT_RESOURCES_BASE / f"models/{MODEL_NAME}"
    return ""


def _setup_experiment_config(
        case: BenchmarkCase,
        experiment_mode: PlannerMode
    ) -> ExperimentConfig:
    """
    Helper function for setting up the experimental config for both the 
    baseline and interruption-based planner benchmark experiments.
    """
    # evaluate on the scenes and task distribution of a data-generation run.
    # Set per case, in the worker process, before any environment is built.
    remap_dir = (
        use_remapped_scenes(REMAPPED_SCENES_HASH, case.params["procthor_seed"])
        if REMAPPED_SCENES_HASH else None
    )
    seeds = _build_seeds(case, remap_dir)
    task_arrival_fn = partial(
        get_task_arrival_prob,
        RandomVariableType.CONTINUOUS,
        -1,  # arrival_prob: only used by the discrete random variable
        case.params["time_between_arrivals"]
    )

    # get task distribution from alfred dataset used during training
    task_distribution = get_scene_task_distribution(seeds.procthor_seed, remap_dir)
    if case.params["randomize_task_sequence"]:
        current_goal, task_distribution = randomize_task_distribution_order(
            task_distribution, seeds.task_sample_seed
        )
    else:
        current_goal = get_example_procthor_goal()

    return ExperimentConfig(
        seeds=seeds,
        goal=current_goal,
        interrupting_task_dist=task_distribution,
        task_arrival_fn=task_arrival_fn,
        ev_model_path=_model_path(experiment_mode),
        num_task_sequence=case.params["num_task_sequence"],
        augment_task=case.params["augment_task"],
        retry_with_subgoals=RETRY_WITH_SUBGOALS
    )


def _run(case: BenchmarkCase, experiment_mode: PlannerMode) -> dict:
    """
    Runs a single benchmark case with the given planner.
    """
    config = _setup_experiment_config(case, experiment_mode)
    return run_experiment(
        config,
        experiment_mode,
        remove_duplicates=True,
        benchmark_flag=True,
        relevant_objects=(
            extract_relevant_objects(config.interrupting_task_dist[0])
            if FILTER_OBJECTS else None
        )
    )


def _register(experiment_mode: PlannerMode, name: str, label: str, tag: str):
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
        return _run(case, experiment_mode)

    bench.add_cases(_get_cases())
    return bench


bench_interruption_ap = _register(
    PlannerMode.INTERRUPTION_AP, "procthor_interruption_ap", "interruption-ap", "interruption_ap"
)
bench_myopic = _register(
    PlannerMode.MYOPIC, "procthor_myopic", "myopic", "myopic"
)
bench_ap = _register(
    PlannerMode.ANTICIPATORY_PLANNING, "procthor_ap", "anticipatory planning", "ap"
)

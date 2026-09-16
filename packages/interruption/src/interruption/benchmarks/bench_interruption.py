"""
Real-time Task Stream Planning Benchmark

Wraps `run_experiment` (experiments.py) as a railroad.bench benchmark so
interruption-planning sweeps can be run in parallel and tracked in
MLflow / viewed via `railroad benchmarks dashboard`.
"""
from functools import partial
import itertools
from typing import Any

from railroad.bench import BenchmarkCase, benchmark
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE

from ..constants import (
    MODEL_NAME, EXPERIMENT_REPEATS, AUGMENT_TASK, EXPECTED_TIME_NEXT_ARRIVAL, NUM_TASKS,
    PROCTHOR_SEED, OBJ_PLACEMENT_SEED, FILTER_OBJECTS, TIMEOUT, RETRY_WITH_SUBGOALS
)
from ..environments import (
    construct_procthor_kitchen_environment,
    get_alfred_task_distribution,
    get_example_procthor_goal,
    # get_example_procthor_task_distribution,
)
from ..experiments import ExperimentConfig, ExperimentSeeds, run_experiment
from ..planning_framework import PlannerMode
from ..utilities import (
    RandomVariableType, randomize_task_distribution_order, get_task_arrival_prob,
    extract_relevant_objects
)


def _get_cases() -> list[dict[str, Any]]:
    """
    Helper function for computing the benchmark cases for both the baseline and 
    the interruption-based planner experiments in procthor environments.
    """
    return [
        {
            "procthor_seed": PROCTHOR_SEED,
            "task_dist_idx": 0,
            "time_between_arrivals": time_between_arrivals,
            "interruption_seed": seed,
            "num_task_sequence": num_task_sequence,
            "randomize_task_sequence": True,
            "augment_task": AUGMENT_TASK
        }
        for (time_between_arrivals, seed), num_task_sequence in itertools.product(
            zip(EXPECTED_TIME_NEXT_ARRIVAL, [140, 42, 240, 57, 1096, 4065, 720]),
            [5]
        )
    ]


def _setup_experiment_config(
        case: BenchmarkCase,
        experiment_mode: PlannerMode
    ) -> ExperimentConfig:
    """
    Helper function for setting up the experimental config for both the 
    baseline and interruption-based planner benchmark experiments.
    """
    seeds = ExperimentSeeds(
        case.params["procthor_seed"],
        case.params["interruption_seed"] + case.repeat_idx,
        75, # keep fixed for right now
        # case.repeat_idx,
        object_placement_seed=OBJ_PLACEMENT_SEED
    )
    task_arrival_fn = partial(
        get_task_arrival_prob,
        RandomVariableType.CONTINUOUS,
        -1,
        case.params["time_between_arrivals"]
    )

    # get task distribution from alfred dataset used during training
    env = construct_procthor_kitchen_environment(
        seeds.procthor_seed, remove_duplicates=True
    )
    task_distribution = get_alfred_task_distribution(
        env.scene.objects,
        set(env.scene.locations),
        size=NUM_TASKS,
        one_object_per_taskdist=True
    )
    current_goal = get_example_procthor_goal()
    if case.params["randomize_task_sequence"]:
        current_goal, task_distribution = randomize_task_distribution_order(
            task_distribution, seeds.task_sample_seed
        )

        # for smaller scale experiments, just reorder the task sequence
        task_sequence = (
            task_distribution[0][:case.params["num_task_sequence"]-1],
            task_distribution[1][:case.params["num_task_sequence"]-1]
        )
        _, task_sequence = randomize_task_distribution_order(task_sequence, case.repeat_idx)

        task_distribution[0][:case.params["num_task_sequence"]-1] = task_sequence[0]
        task_distribution[1][:case.params["num_task_sequence"]-1] = task_sequence[1]

    model_path = (
        DEFAULT_RESOURCES_BASE / f"models/{MODEL_NAME}"
        if experiment_mode in [
            PlannerMode.INTERRUPTION,
            PlannerMode.ANTICIPATORY_PLANNING,
            PlannerMode.INTERRUPTION_AP
        ]
        else ""
    )

    config = ExperimentConfig(
        seeds,
        current_goal,
        task_distribution,
        task_arrival_fn,
        model_path,
        case.params["num_task_sequence"],
        case.params["augment_task"],
        retry_with_subgoals=RETRY_WITH_SUBGOALS
    )
    return config


@benchmark(
    name="procthor_interruption",
    description=(
        "Evaluates the interruption planner across "
        "task-arrival probabilities in specified procthor environments."
    ),
    tags=["interruption", "procthor"],
    timeout=TIMEOUT,
    repeat=EXPERIMENT_REPEATS,
)
def bench_interruption_kitchen(case: BenchmarkCase):
    """
    Wrapper function to evaluate the interruption-based planner on procthor kitchen
    environments. 
    """
    config = _setup_experiment_config(case, PlannerMode.INTERRUPTION)
    return run_experiment(config, PlannerMode.INTERRUPTION, True, True)

bench_interruption_kitchen.add_cases(_get_cases())


@benchmark(
    name="procthor_interruption_ap",
    description=(
        "Evaluates the interruption-ap planner across "
        "task-arrival probabilities in specified procthor environments."
    ),
    tags=["interruption", "procthor", "ap", "interruption_experiments"],
    timeout=TIMEOUT,
    repeat=EXPERIMENT_REPEATS,
)
def bench_interruption_ap_kitchen(case: BenchmarkCase):
    """
    Wrapper function to evaluate the interruption-based planner on procthor kitchen
    environments. 
    """
    config = _setup_experiment_config(case, PlannerMode.INTERRUPTION_AP)
    return run_experiment(
        config,
        PlannerMode.INTERRUPTION_AP,
        True,
        True,
        extract_relevant_objects(config.interrupting_task_dist[0]) if FILTER_OBJECTS else None
    )

bench_interruption_ap_kitchen.add_cases(_get_cases())


@benchmark(
    name="procthor_myopic",
    description=(
        "Evaluates the myopic planner across "
        "task-arrival probabilities in specified procthor environments."
    ),
    tags=["interruption", "procthor", "myopic", "interruption_experiments"],
    timeout=TIMEOUT,
    repeat=EXPERIMENT_REPEATS,
)
def bench_myopic_interruption_kitchen(case: BenchmarkCase):
    """
    Wrapper function to evaluate the interruption-based planner on procthor kitchen
    environments. 
    """
    config = _setup_experiment_config(case, PlannerMode.MYOPIC)
    return run_experiment(
        config,
        PlannerMode.MYOPIC,
        True,
        True,
        extract_relevant_objects(config.interrupting_task_dist[0]) if FILTER_OBJECTS else None
    )

bench_myopic_interruption_kitchen.add_cases(_get_cases())


# @benchmark(
#     name="procthor_ap",
#     description=(
#         "Evaluates the anticipatory planning planner across "
#         "task-arrival probabilities in specified procthor environments."
#     ),
#     tags=["interruption", "procthor", "ap", "interruption_experiments"],
#     timeout=TIMEOUT,
#     repeat=EXPERIMENT_REPEATS,
# )
# def bench_ap_kitchen(case: BenchmarkCase):
#     """
#     Wrapper function to evaluate the interruption-based planner on procthor kitchen
#     environments
#     """
#     config = _setup_experiment_config(case, PlannerMode.ANTICIPATORY_PLANNING)
#     return run_experiment(
#         config,
#         PlannerMode.ANTICIPATORY_PLANNING,
#         True,
#         True,
#         extract_relevant_objects(config.interrupting_task_dist[0]) if FILTER_OBJECTS else None
#     )

# bench_ap_kitchen.add_cases(_get_cases())

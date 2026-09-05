"""
Real-time Task Stream Planning Benchmark

Wraps `run_experiment` (experiments.py) as a railroad.bench benchmark so
interruption-planning sweeps can be run in parallel and tracked in
MLflow / viewed via `railroad benchmarks dashboard`.
"""
from functools import partial
import random
import itertools
from typing import Any

from railroad.bench import BenchmarkCase, benchmark
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE

from ..constants import MODEL_NAME, EXPERIMENT_REPEATS, AUGMENT_TASK, EXPECTED_TIME_NEXT_ARRIVAL
from ..environments import (
    construct_procthor_kitchen_environment,
    get_alfred_task_distribution,
    get_example_procthor_goal,
    # get_example_procthor_task_distribution,
)
from ..experiments import ExperimentConfig, ExperimentSeeds, run_experiment, ExperimentMode
from ..utilities import (
    RandomVariableType, randomize_task_distribution_order, get_task_arrival_prob
)


def _get_cases() -> list[dict[str, Any]]:
    """
    Helper function for computing the benchmark cases for both the baseline and 
    the interruption-based planner experiments in procthor environments.
    """
    return [
        {
            "procthor_seed": 201,
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
        # case.repeat_idx,
        object_placement_seed=19
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
        one_object_per_taskdist=True
    )
    current_goal = get_example_procthor_goal()
    if case.params["randomize_task_sequence"]:
        # current_goal, task_distribution = randomize_task_distribution_order(
        #     task_distribution, seeds.task_sample_seed
        # )
        # task_distribution = (
        #     list(task_distribution[0][:case.params["num_task_sequence"]-1]),
        #     list(task_distribution[1][:case.params["num_task_sequence"]-1])
        # )

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
        if experiment_mode in [
            ExperimentMode.INTERRUPTION,
            ExperimentMode.ANTICIPATORY_PLANNING,
            ExperimentMode.INTERRUPTION_AP
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
        case.params["augment_task"]
    )
    return config


@benchmark(
    name="procthor_interruption",
    description=(
        "Evaluates the interruption planner across "
        "task-arrival probabilities in specified procthor environments."
    ),
    tags=["interruption", "procthor"],
    timeout=900.0,
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
    name="procthor_interruption_ap",
    description=(
        "Evaluates the interruption-ap planner across "
        "task-arrival probabilities in specified procthor environments."
    ),
    tags=["interruption", "procthor", "ap"],
    timeout=900.0,
    repeat=EXPERIMENT_REPEATS,
)
def bench_interruption_ap_kitchen(case: BenchmarkCase):
    """
    Wrapper function to evaluate the interruption-based planner on procthor kitchen
    environments. 
    """
    config = _setup_experiment_config(case, ExperimentMode.INTERRUPTION_AP)
    return run_experiment(config, ExperimentMode.INTERRUPTION_AP, True, True)

bench_interruption_ap_kitchen.add_cases(_get_cases())


@benchmark(
    name="procthor_myopic",
    description=(
        "Evaluates the myopic planner across "
        "task-arrival probabilities in specified procthor environments."
    ),
    tags=["interruption", "procthor", "myopic"],
    timeout=900.0,
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
    name="procthor_ap",
    description=(
        "Evaluates the anticipatory planning planner across "
        "task-arrival probabilities in specified procthor environments."
    ),
    tags=["interruption", "procthor", "ap"],
    timeout=900.0,
    repeat=EXPERIMENT_REPEATS,
)
def bench_ap_kitchen(case: BenchmarkCase):
    """
    Wrapper function to evaluate the interruption-based planner on procthor kitchen
    environments. 
    """
    config = _setup_experiment_config(case, ExperimentMode.ANTICIPATORY_PLANNING)
    return run_experiment(config, ExperimentMode.ANTICIPATORY_PLANNING, True, True)

bench_ap_kitchen.add_cases(_get_cases())

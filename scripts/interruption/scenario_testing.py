"""
Scenario testing: compares the total task-sequence cost of the plans executed by
  - the interruption-aware anticipatory planner (InterruptionAPPlanner),
  - the sampling-based vanilla anticipatory planner (AnticipatoryPlanner), and
  - the vanilla anticipatory objective searched without sampling
    (AnticipatorySearchPlanner),
on the same scenarios (scene, starting task, task sequence, arrival seeds).
"""
from collections.abc import Callable
from pathlib import Path

from interruption.constants import (
    MODEL_NAME, ARRIVAL_EXCEEDANCE_FRACTIONS, PROCTHOR_SEED, INTERRUPTION_SEEDS
)
from interruption.cost_formulations import NO_INTERRUPTION
from interruption.experiments import RunResult, execute
from interruption.planners import (
    AnticipatoryPlanner, AnticipatorySearchPlanner, InterruptionAPPlanner, load_ev_model,
)
from interruption.planning_framework import Planner
from interruption.scenarios import Scenario
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE
from railroad.environment.procthor.scenegraph import SceneGraph

# constants
MODEL_PATH = DEFAULT_RESOURCES_BASE / "models"
OUTPUT_PATH = Path("scenario_testing_results.csv")


def make_scenarios() -> dict[str, Scenario]:
    """The scenarios to compare the planners on, by name."""
    # TODO: fill in the scenarios to test
    return {
        "example": Scenario.procthor(
            PROCTHOR_SEED,
            arrival_exceedance=ARRIVAL_EXCEEDANCE_FRACTIONS[-1],
            experiment_seed=INTERRUPTION_SEEDS[-1] + 20,
            task_sample_seed=20,
            num_task_sequence=2,
            augment_task=True,
            retry_with_subgoals=True,
        ),
    }


def make_planners(ev_model: Callable[[SceneGraph], float]) -> dict[str, Planner]:
    """The planners to compare, by name."""
    return {
        # TODO: confirm this is the intended baseline discounting for the
        # interruption-AP planner: by the chance no task arrives during each step
        "interruption_ap": InterruptionAPPlanner(ev_model=ev_model, discount=NO_INTERRUPTION),
        "anticipatory_planning": AnticipatoryPlanner(ev_model=ev_model),
        "ap_search": AnticipatorySearchPlanner(ev_model=ev_model),
    }


def summarize(all_results: dict[str, dict[str, RunResult]]) -> None:
    """Prints, and writes to OUTPUT_PATH, the total sequence cost of each planner per scenario."""
    names = list(next(iter(all_results.values())))
    rows = ["scenario," + ",".join(f"{n}_cost,{n}_success" for n in names)]
    for scenario_name, results in all_results.items():
        rows.append(scenario_name + "," + ",".join(
            f"{results[n].cost:.2f},{results[n].success}" for n in names
        ))
    OUTPUT_PATH.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    # TODO: aggregate statistics (mean/std of cost, pairwise cost differences, win rates)


def main():
    planners = make_planners(load_ev_model(MODEL_PATH / MODEL_NAME))
    all_results = {
        scenario_name: {name: execute(scenario, planner) for name, planner in planners.items()}
        for scenario_name, scenario in make_scenarios().items()
    }
    summarize(all_results)


if __name__ == "__main__":
    main()

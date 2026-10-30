from dataclasses import replace

from interruption.constants import (
    ARRIVAL_EXCEEDANCE_FRACTIONS, PROCTHOR_SEED, FILTER_OBJECTS, INTERRUPTION_SEEDS
)
from interruption.experiments import run_experiment
from interruption.planners import MyopicPlanner
from interruption.scenarios import Scenario
# from railroad.core import LiteralGoal, Fluent as F

# constants
RANDOMIZE_TASK_SEQUENCE = True
RUN_IDX_SEED = 20

def main(randomize_order: bool = False, filter_objects: bool = False):
    # evaluate on the scenes and task distribution of a data-generation run
    # (constants.REMAPPED_SCENES_HASH)
    scenario = Scenario.procthor(
        PROCTHOR_SEED,
        arrival_exceedance=ARRIVAL_EXCEEDANCE_FRACTIONS[-1],
        experiment_seed=INTERRUPTION_SEEDS[-1] + RUN_IDX_SEED,
        task_sample_seed=RUN_IDX_SEED if randomize_order else None,
        filter_objects=filter_objects,
        num_task_sequence=2,
        augment_task=True,
        retry_with_subgoals=True,
    )
    if not randomize_order:
        # both apple and pan are located at countertop3
        tasks, probs = scenario.interrupting_task_dist
        tasks = list(tasks)
        tasks[0], tasks[3] = tasks[3], tasks[0]  # pan at fridge arrives first
        scenario = replace(
            scenario, goal=scenario.interrupting_task_dist[0][0],  # apple at fridge
            interrupting_task_dist=(tasks, probs),
        )

    run_experiment(scenario, MyopicPlanner(), show_plot=False)

if __name__ == "__main__":
    main(RANDOMIZE_TASK_SEQUENCE, FILTER_OBJECTS)

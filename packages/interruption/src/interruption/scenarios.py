"""
Scenarios: everything that pins down one experiment run except the planner. A
Scenario is frozen; running it never changes it, so the same one can be run with
any number of planners (see experiments.execute and experiments.run_experiment):

    scenario = Scenario.procthor(158, arrival_exceedance=0.95, experiment_seed=740,
                                 task_sample_seed=20)
    for planner in (MyopicPlanner(), InterruptionAPPlanner(ev_model=ev)):
        print(execute(scenario, planner).cost)

Goals use generic object and location names (spoon, countertop); each run maps
them to the scene's own (spoon_14, countertop_3).
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial

from railroad.core import Goal

from .constants import (
    AUGMENT_TASK, FILTER_OBJECTS, OBJ_PLACEMENT_SEED, REMAPPED_SCENES_HASH, RETRY_WITH_SUBGOALS,
)
from .environments import (
    KitchenProcTHOREnvironment,
    construct_procthor_kitchen_environment,
    get_example_procthor_goal,
    get_scene_task_distribution,
)
from .utilities import (
    RandomVariableType,
    extract_relevant_objects,
    get_expected_time_next_arrival,
    get_task_arrival_prob,
    randomize_task_distribution_order,
    use_remapped_scenes,
)


@dataclass(frozen=True)
class ExperimentSeeds:
    """
    Data struct that contains all the seeds needed to make an
    interruption experiment deterministic.
    """
    procthor_seed: int = 201
    experiment_seed: int = 240
    task_sample_seed: int = 75
    object_placement_seed: int | None = None
    duplicate_objects_containers_seed: int = 240


@dataclass(frozen=True, kw_only=True)
class Scenario:
    """
    One experiment run's scene, tasks and rules, without the planner. Build one
    with Scenario.procthor, or directly for a hand-made task sequence.
    """
    seeds: ExperimentSeeds
    # the first task, in generic names
    goal: Goal
    # the tasks that arrive, in arrival order, and their probabilities, which the
    # planners use as the distribution of tasks that may arrive; generic names
    interrupting_task_dist: tuple[Sequence[Goal], list[float]]
    # the chance a task arrives during an action, from the action's cost
    task_arrival_fn: Callable[[float], float]
    # tasks in the run, the first one included
    num_task_sequence: int = 2
    # an arriving task is added to the unfinished current one, rather than replacing it
    augment_task: bool = False
    # a failed search is retried on smaller sub-tasks (planning_framework.search_with_retry)
    retry_with_subgoals: bool = False
    # time charged for a task the planner fails to find a plan for; about the
    # worst plan cost in the scene (2-room seed 64: 375, 1-room seed 201: 500)
    failure_cost: float = 500
    # the scene's objects to keep (None: all of them), and whether to keep one
    # object per category
    relevant_objects: list[str] | None = None
    remove_duplicate_objects: bool = True
    # the multi-scene data-generation run whose remapped scene to load (see
    # utilities.use_remapped_scenes); None loads the plain scene
    remap_hash: str | None = None
    # the average time between task arrivals task_arrival_fn was built from,
    # when known; recorded with results, not used to run the scenario
    time_between_arrivals: float | None = field(default=None, compare=False)

    @classmethod
    def procthor(
        cls,
        procthor_seed: int,
        *,
        arrival_exceedance: float,
        experiment_seed: int,
        task_sample_seed: int | None = None,
        remap_hash: str | None = REMAPPED_SCENES_HASH,
        object_placement_seed: int | None = OBJ_PLACEMENT_SEED,
        filter_objects: bool = FILTER_OBJECTS,
        num_task_sequence: int = 2,
        augment_task: bool = AUGMENT_TASK,
        retry_with_subgoals: bool = RETRY_WITH_SUBGOALS,
        failure_cost: float = 500,
    ) -> "Scenario":
        """
        A ProcTHOR scenario on the scene's task distribution: the one the
        remap_hash data-generation run used, or the plain scene's own.

        arrival_exceedance sets how often tasks arrive, as the fraction of the
        training tasks that take longer than the median time to the next arrival
        (0: never; see utilities.get_expected_time_next_arrival). experiment_seed
        decides when they arrive. task_sample_seed shuffles the task order and
        picks the first task from it; None keeps the stored order and starts
        with the example goal.
        """
        # must come before the task distribution is loaded or a scene is built
        remap_dir = use_remapped_scenes(remap_hash, procthor_seed) if remap_hash else None
        time_between_arrivals = get_expected_time_next_arrival(arrival_exceedance, remap_hash)
        task_distribution = get_scene_task_distribution(procthor_seed, remap_dir)
        if task_sample_seed is None:
            goal = get_example_procthor_goal()
        else:
            goal, task_distribution = randomize_task_distribution_order(
                task_distribution, task_sample_seed
            )
        seeds = ExperimentSeeds(
            procthor_seed=procthor_seed,
            experiment_seed=experiment_seed,
            # an object seed makes ThorInterface skip the remap, so a remapped
            # scene keeps the object placement it was generated with
            object_placement_seed=None if remap_dir else object_placement_seed,
        )
        if task_sample_seed is not None:
            seeds = replace(seeds, task_sample_seed=task_sample_seed)
        return cls(
            seeds=seeds,
            goal=goal,
            interrupting_task_dist=task_distribution,
            task_arrival_fn=partial(
                get_task_arrival_prob,
                RandomVariableType.CONTINUOUS,
                -1,  # arrival_prob: only used by the discrete random variable
                time_between_arrivals,
            ),
            num_task_sequence=num_task_sequence,
            augment_task=augment_task,
            retry_with_subgoals=retry_with_subgoals,
            failure_cost=failure_cost,
            relevant_objects=(
                extract_relevant_objects(task_distribution[0]) if filter_objects else None
            ),
            remap_hash=remap_hash,
            time_between_arrivals=time_between_arrivals,
        )

    def build_environment(self) -> KitchenProcTHOREnvironment:
        """A fresh environment in the scenario's starting state."""
        if self.remap_hash:
            use_remapped_scenes(self.remap_hash, self.seeds.procthor_seed)
        return construct_procthor_kitchen_environment(
            self.seeds.procthor_seed,
            object_seed=self.seeds.object_placement_seed,
            relevant_objects=self.relevant_objects,
            remove_duplicate_objects=self.remove_duplicate_objects,
        )

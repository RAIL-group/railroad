"""
The planners an interruption experiment can run. Each is a frozen description of
one planner variant, so a variant is just a constructor argument:

    ev_model = load_ev_model(MODEL_PATH / MODEL_NAME)
    planners = {
        "myopic": MyopicPlanner(),
        "ap": AnticipatoryPlanner(ev_model=ev_model, num_samples=40),
        "int_ap": InterruptionAPPlanner(ev_model=ev_model),
        "int_ap_d1": InterruptionAPPlanner(ev_model=ev_model, discount=1.0),
    }

Pass one to run_experiment, or to search_with_retry / search_without_retry; they
call its plan(). Each planner's config is the PlannerConfig its A* searches use.
"""
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from railroad.core import Action, Fluent as F
from railroad.environment.procthor.scenegraph import SceneGraph

from .cost_formulations import NO_INTERRUPTION, DefaultFormulation, Discount
from .environments import KitchenProcTHOREnvironment
from .learning.models.gcn import AnticipateGCN
from .learning.utils import get_torch_device
from .planner import (
    FFHeuristic, InterruptionSearchProblem, PlannerConfig, SearchStats, astar_search,
)
from .planning_framework import _halo_problem, anticipatory_planner, planning_state


def load_ev_model(model_path: Path | str) -> Callable[[SceneGraph], float]:
    """The learned expected-value model (an AnticipateGCN) saved at model_path."""
    return AnticipateGCN.get_net_eval_fn(model_path, get_torch_device())


def _batch_fn(ev_model: Callable[[SceneGraph], float] | None):
    # AnticipateGCN.get_net_eval_fn attaches a batched sibling to the eval fn it
    # returns; a stub without one just disables batching
    return getattr(ev_model, "batch", None)


@dataclass(frozen=True, kw_only=True)
class MyopicPlanner:
    """A* for the current task alone."""
    h_multiplier: float = 1
    # search first over only the objects and locations of the goal's relaxed
    # plan, falling back to the full problem (planner.relaxed_plan_problem)
    relaxed_plan_filtering: bool = True
    config: PlannerConfig = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "config", PlannerConfig(
            heuristic_fn=FFHeuristic(multiplier=self.h_multiplier),
            cost_formulation=DefaultFormulation(discount=1.0),
            relaxed_plan_filtering=self.relaxed_plan_filtering,
        ))

    def plan(
        self,
        environment: KitchenProcTHOREnvironment,
        search_problem: InterruptionSearchProblem,
        neg_to_pos_mapping: dict[F, F],
        stats: SearchStats | None = None,
    ) -> tuple[list[Action], bool]:
        initial_state = planning_state(environment.state, neg_to_pos_mapping)
        plan, _, success, _ = astar_search(
            (initial_state, None), search_problem, self.config, stats=stats
        )
        return plan, success

    def myopic(self) -> "MyopicPlanner":
        return self


@dataclass(frozen=True, kw_only=True)
class AnticipatoryPlanner:
    """
    The anticipatory planning framework of Talukder et al. (see
    planning_framework.anticipatory_planner): myopic plans for the current task
    and for augmented versions of it, chosen by cost plus ev_model's value of
    the state each one ends in.
    """
    ev_model: Callable[[SceneGraph], float]
    h_multiplier: float = 1
    # narrows each search to its goal's relaxed plan, which keeps the extra
    # object and location an augmented goal names
    relaxed_plan_filtering: bool = True
    # augmented tasks evaluated per plan, and the seed they're sampled with
    num_samples: int = 20
    seed: int = 12
    config: PlannerConfig = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "config", PlannerConfig(
            heuristic_fn=FFHeuristic(multiplier=self.h_multiplier),
            cost_formulation=DefaultFormulation(discount=1.0),
            interruption_value_fn=self.ev_model,
            interruption_value_batch_fn=_batch_fn(self.ev_model),
            relaxed_plan_filtering=self.relaxed_plan_filtering,
        ))

    def plan(
        self,
        environment: KitchenProcTHOREnvironment,
        search_problem: InterruptionSearchProblem,
        neg_to_pos_mapping: dict[F, F],
        stats: SearchStats | None = None,
    ) -> tuple[list[Action], bool]:
        initial_state = planning_state(environment.state, neg_to_pos_mapping)
        plan, _, success = anticipatory_planner(
            (initial_state, environment.scene.scene_graph),
            search_problem,
            self.config,
            environment.scene,
            neg_to_pos_mapping,
            num_samples=self.num_samples,
            seed=self.seed,
        )
        return plan, success

    def myopic(self) -> MyopicPlanner:
        return MyopicPlanner(h_multiplier=self.h_multiplier)


@dataclass(frozen=True, kw_only=True)
class InterruptionPlanner:
    """
    Interruption-aware A*: each step's cost includes the chance an interrupting
    task arrives during it times ev_model's value of the state it leads to.
    """
    ev_model: Callable[[SceneGraph], float]
    # the planner's task-arrival model: a probability per action, or a function
    # of the action's cost; None uses the search problem's own model
    interruption_prob_fn: float | Callable[[float], float] | None = None
    # NO_INTERRUPTION discounts each step by the chance no task arrives during
    # it; a number is a fixed discount factor (1.0: undiscounted)
    discount: Discount = NO_INTERRUPTION
    h_multiplier: float = 1
    # (FF weight, value weight) to add the value of the state to the heuristic;
    # None leaves it out
    h_weights: tuple[float, float] | None = None
    # search first over the task's relaxed plan plus a halo of nearby objects
    # future tasks need (planning_framework.interruption_halo)
    halo_filtering: bool = False
    config: PlannerConfig = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "config", PlannerConfig(
            heuristic_fn=FFHeuristic(multiplier=self.h_multiplier, weights=self.h_weights),
            cost_formulation=DefaultFormulation(discount=self.discount),
            planner_interruption_prob_fn=self.interruption_prob_fn,
            interruption_value_fn=self.ev_model,
            interruption_value_batch_fn=_batch_fn(self.ev_model),
            interruption_halo_filtering=self.halo_filtering,
        ))

    def plan(
        self,
        environment: KitchenProcTHOREnvironment,
        search_problem: InterruptionSearchProblem,
        neg_to_pos_mapping: dict[F, F],
        stats: SearchStats | None = None,
    ) -> tuple[list[Action], bool]:
        config = self.config
        if config.planner_interruption_prob_fn is None:
            config = replace(
                config, planner_interruption_prob_fn=search_problem.interruption_prob_fn
            )
        initial_state = planning_state(environment.state, neg_to_pos_mapping)
        state_scene_graph = (initial_state, environment.scene.scene_graph)

        plan: list[Action] = []
        success = False
        if config.interruption_halo_filtering:
            narrowed = _halo_problem(environment, search_problem, config, initial_state)
            if narrowed is not None:
                plan, _, success, _ = astar_search(
                    state_scene_graph, narrowed, config, stats=stats
                )
        if not success:  # no halo, or the narrowed problem failed: solve it all
            plan, _, success, _ = astar_search(
                state_scene_graph, search_problem, config, stats=stats
            )
        return plan, success

    def myopic(self) -> MyopicPlanner:
        return MyopicPlanner(h_multiplier=self.h_multiplier)


@dataclass(frozen=True, kw_only=True)
class InterruptionAPPlanner(InterruptionPlanner):
    """InterruptionPlanner with the state's value added to the FF heuristic."""
    h_weights: tuple[float, float] | None = (1, 1)

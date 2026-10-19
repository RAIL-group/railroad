import heapq
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

from railroad.core import Action, Fluent, Goal, State, get_next_actions
from railroad.environment.procthor.scenegraph import SceneGraph

from .constants import SEARCH_DEBUG
from .cost_formulations import CostFormulation, DefaultFormulation, StepTerms
from .utilities import (
    get_action_cost,
    get_next_state,
    get_updated_scene_graph
)

# data structures for astar search
@dataclass
class InterruptionSearchProblem:
    """
    Data structure used store inputs required to define a 
    search problem with continual task arrival.
    """
    goal: Goal
    actions: list[Action]
    interrupting_task_dist: tuple[list[Goal], list[float]] | None = None
    # environment side interruption probability function
    interruption_prob_fn: float | Callable[[float], float] = 0


@dataclass
class PlannerConfig:
    """
    Data stucture used to store user-specified hyperparameters for
    A* Search.
    """
    discount_by_no_int_prob: bool
    heuristic_fn: Callable[[State, Goal, list[Action], float], float] | float
    discount: Optional[float] = None
    planner_interruption_prob_fn: float | Callable[[float], float] | None = None
    interruption_value_fn: Callable[[SceneGraph], float] | None = None
    current_task_reward: float = 0
    # maximum number of goals in a task that can be solved within the planner's budget
    max_task_complexity: int = -1
    # optional batched form of interruption_value_fn: takes every one of an expanded
    # node's candidate children's scene graphs and returns one value per graph, in
    # order, via a single model call instead of one call per child. astar_search uses
    # this (when provided) to pre-warm ev_cache before create_child runs; falls back
    # to calling interruption_value_fn once per child when left as None.
    interruption_value_batch_fn: Callable[[list[SceneGraph]], list[float]] | None = None
    # how a search step's costs are computed (see cost_formulations.py); None ->
    # DefaultFormulation, built from discount_by_no_int_prob, discount and
    # current_task_reward above
    cost_formulation: CostFormulation | None = None

    def get_cost_formulation(self) -> CostFormulation:
        """The cost formulation the search uses."""
        if self.cost_formulation is not None:
            return self.cost_formulation
        return DefaultFormulation(
            self.discount_by_no_int_prob, self.discount, self.current_task_reward
        )


@dataclass
class SearchStats:
    """How much of its expansion budget one astar_search call used."""
    goal: Goal | None = None
    expanded: int = 0
    budget: int = 0
    success: bool = False
    # a retry over only the goal's own objects, after an unfiltered search failed
    filtered: bool = False

    @property
    def fraction(self) -> float:
        return self.expanded / self.budget if self.budget else 0.0


@dataclass
class InterruptionTrajectory:
    """
    Data structure used to represent search tree trajectories (paths).
    """
    state: State
    # action that was executed from the parent state that resulted in the current state
    action: Optional[Action]
    # running discount of the path so far (gamma_t), applied to the next step's costs;
    # see cost_formulations.py for how each step updates it
    discount_factor: float
    scene_graph: SceneGraph | None
    cost: float = 0.0
    value: float = 0.0
    discounted_h_value: float = 0.0
    h_value: float = 0.0
    parent: Optional["InterruptionTrajectory"] = None
    # the state's fluents as a hashable key, built once and shared by every cache
    # and set in astar_search, rather than rebuilt (and stored again) by each
    key: frozenset[Fluent] | None = None

    def create_child(
        self,
        search_problem: InterruptionSearchProblem,
        planner_params: PlannerConfig,
        action: Action,
        interruption_prob: float,
        ev_cache: Optional[dict[frozenset[Fluent], float]] = None,
        heuristic_cache: Optional[dict[frozenset[Fluent], float]] = None,
        next_state: State | None = None,
        next_state_key: frozenset[Fluent] | None = None,
    ) -> 'InterruptionTrajectory':
        """
        Helper function for creation of trajectories to add to the frontier.

        ev_cache/heuristic_cache, when provided, are keyed by the resulting state's
        fluents -- the same notion of "same state" astar_search's own `expanded` set
        already uses. interruption_value_fn is a pure function of those fluents (the
        scene graph it's evaluated on only ever reflects active fluents, never time or
        pending effects), so ev_cache is exact. heuristic_fn is NOT exact under this
        key: ff_heuristic also reads state.time and pending timed effects, so two
        fluent-identical states reached with different timing can get a heuristic
        value computed for the other one's timing. This can change search order (and
        so which valid plan is found first) but never corrupts accumulated_cost, since
        the heuristic value only ever feeds estimated_future_cost/h_value, not cost.

        The costs themselves come from planner_params' cost formulation (see
        cost_formulations.py).

        next_state/next_state_key, when given, are the resulting state and its key,
        which astar_search has already computed; they are computed here otherwise.
        """
        if next_state is None:
            next_state, _ = get_next_state(self.state, action)
        if next_state_key is None:
            next_state_key = frozenset(next_state.fluents)

        # compute accumulated cost (g(traj))
        interrupting_task_ev = 0
        scene_graph = None
        if self.scene_graph is not None:
            scene_graph = self.scene_graph.shallow_copy()
            get_updated_scene_graph(scene_graph, next_state, action)
        if planner_params.interruption_value_fn is not None:
            assert scene_graph is not None
            if ev_cache is not None and check_value_cache(next_state_key, ev_cache):
                interrupting_task_ev = ev_cache[next_state_key]
            else:
                interrupting_task_ev = planner_params.interruption_value_fn(scene_graph)
                if ev_cache is not None:
                    ev_cache[next_state_key] = interrupting_task_ev

        formulation = planner_params.get_cost_formulation()
        terms = self.step_terms(action, interruption_prob, interrupting_task_ev)
        accumulated_cost = formulation.accumulated_cost(terms)

        # compute estimated future cost (h(s'))
        if isinstance(planner_params.heuristic_fn, (int, float)):
            undiscounted_future_cost = planner_params.heuristic_fn
        else:
            v_ap = 0 if planner_params.interruption_value_fn is None else interrupting_task_ev
            if heuristic_cache is not None and check_value_cache(next_state_key, heuristic_cache):
                undiscounted_future_cost = heuristic_cache[next_state_key]
            else:
                undiscounted_future_cost = planner_params.heuristic_fn(
                    next_state, search_problem.goal, search_problem.actions, v_ap
                )
                if heuristic_cache is not None:
                    heuristic_cache[next_state_key] = undiscounted_future_cost

        discount_factor = formulation.child_discount(terms)
        estimated_future_cost = formulation.estimated_future_cost(
            terms, discount_factor, undiscounted_future_cost
        )

        return InterruptionTrajectory(
            cost=accumulated_cost,
            value=accumulated_cost+estimated_future_cost,
            state=next_state,
            action=action,
            discount_factor=discount_factor,
            discounted_h_value=estimated_future_cost,
            h_value=undiscounted_future_cost,
            scene_graph=scene_graph,
            parent=self,
            key=next_state_key,
        )

    def step_terms(
        self, action: Action, interruption_prob: float, interrupting_task_ev: float
    ) -> StepTerms:
        """
        The inputs a cost formulation reads for the step from this trajectory
        through action.
        """
        return StepTerms(
            parent_cost=self.cost,
            parent_discount=self.discount_factor,
            action_cost=get_action_cost(action),
            interruption_prob=interruption_prob,
            interrupting_task_ev=interrupting_task_ev,
        )

    def get_plan(self: "InterruptionTrajectory") -> list[Action]:
        """
        Recovers the plan that resulted in the trajectory.
        """
        plan = []
        current_node = self
        while current_node.parent is not None:
            assert current_node.action is not None
            plan.append(current_node.action)
            current_node = current_node.parent
        plan.reverse()
        return plan

    def get_plan_cost(self):
        """
        Returns actual cost of a trajectory, without factoring the interruption probabilities.
        """
        plan_cost = 0
        for act in self.get_plan():
            plan_cost+=get_action_cost(act)
        return plan_cost

    def __eq__(self, other):
        if not isinstance(other, InterruptionTrajectory):
            raise NotImplementedError
        return self.value == other.value

    def __lt__(self, other):
        if not isinstance(other, InterruptionTrajectory):
            raise NotImplementedError
        return self.value < other.value


    def __le__(self, other):
        if not isinstance(other, InterruptionTrajectory):
            raise NotImplementedError
        return self.value <= other.value

    def __gt__(self, other):
        if not isinstance(other, InterruptionTrajectory):
            raise NotImplementedError
        return self.value > other.value

    def __ge__(self, other):
        if not isinstance(other, InterruptionTrajectory):
            raise NotImplementedError
        return self.value >= other.value


def astar_search(
    initial_state: tuple[State, SceneGraph | None],
    interruption_problem: InterruptionSearchProblem,
    search_params: PlannerConfig,
    num_steps: int = 30000,
    show_progress: bool = False,
    stats: SearchStats | None = None,
) -> tuple[list[Action], float, bool, SceneGraph | None]:
    """
    Astar algorithm implementation. show_progress prints the goal and a progress
    bar over num_steps; off by default, since callers like data generation run
    many searches across parallel workers. stats, when given, is filled in with
    the nodes expanded against the num_steps budget.
    """
    # caches keyed by resulting-state fluents; see create_child's docstring for
    # why ev_cache is exact and heuristic_cache is an approximation
    ev_cache: dict[frozenset[Fluent], float] = {}
    heuristic_cache: dict[frozenset[Fluent], float] = {}
    # best accumulated_cost seen so far per resulting-state fluents, used below to
    # skip create_child (and so the ff_heuristic/GCN work inside it) for a candidate
    # that's provably no better than one already pushed for the same state this
    # search. This is exact, not an approximation like heuristic_cache: it never
    # discards a candidate whose true accumulated_cost we haven't actually computed.
    best_cost: dict[frozenset[Fluent], float] = {}
    cost_formulation = search_params.get_cost_formulation()
    # one key object per distinct state, however many paths reach it: every
    # cache, set and trajectory above shares it rather than holding its own copy
    state_keys: dict[frozenset[Fluent], frozenset[Fluent]] = {}

    def shared_key(state: State) -> frozenset[Fluent]:
        key = frozenset(state.fluents)
        return state_keys.setdefault(key, key)

    # the heap stores tuple(traj, heuristic_value, insertion_order)
    frontier = []
    expanded = set()
    insertion_order = 1
    # keeps track of number of astar search iterations
    num_expanded_nodes = 0

    # push initial state onto heap
    heapq.heappush(
        frontier,
        (
            InterruptionTrajectory(
                state=initial_state[0],
                action=None,
                discount_factor=1,
                # interruption_probs=[],
                scene_graph=initial_state[1],
                key=shared_key(initial_state[0]),
            ), -1, 0
        )
    )

    # search loop
    if show_progress:
        print(f"Current Goal: {interruption_problem.goal}")
    with tqdm(total=num_steps, disable=not show_progress) as pbar:
        while num_expanded_nodes < num_steps and frontier:
            # some logging functionality for debugging
            if SEARCH_DEBUG:
                print_frontier_trace(num_expanded_nodes, frontier)

            # find expansion node
            expand, _, _ = heapq.heappop(frontier)

            # check for goal condition being met
            if interruption_problem.goal.evaluate(expand.state.fluents):
                _fill_stats(stats, interruption_problem.goal, num_expanded_nodes, num_steps, True)
                return expand.get_plan(), expand.cost, True, expand.scene_graph#, curr_state

            # check if we've already expanded this state
            assert expand.key is not None
            if expand.key in expanded:
                continue
            # otherwise add it
            expanded.add(expand.key)
            # expand search tree
            num_expanded_nodes+=1

            # resolve each candidate action's next_state/interruption_prob once,
            # filtering out any that would re-expand an already-closed state
            candidates = []
            for action in get_next_actions(expand.state, interruption_problem.actions):
                # probability of interruption after taking action from current state
                next_state, interruption_prob = get_next_state(
                    expand.state,
                    action,
                    (
                        search_params.planner_interruption_prob_fn
                        if search_params.planner_interruption_prob_fn is not None
                        else 0
                    )
                )

                next_state_key = shared_key(next_state)
                # check if this state has already been expanded
                if next_state_key in expanded:
                    continue
                candidates.append((action, next_state, next_state_key, interruption_prob))

            # batch-evaluate interruption_value_fn once across all of this node's
            # candidate children instead of once per child inside create_child -- the
            # GCN forward pass dominates create_child's cost, and torch_geometric
            # batches naturally. create_child still does its own (uncached) call when
            # this isn't provided, so this is purely an optimization, not a behavior
            # change to what gets computed.
            if (
                search_params.interruption_value_batch_fn is not None
                and expand.scene_graph is not None
            ):
                batch_keys: list[frozenset[Fluent]] = []
                batch_graphs: list[SceneGraph] = []
                seen_keys: set[frozenset[Fluent]] = set()
                for action, next_state, next_state_key, _ in candidates:
                    if next_state_key in ev_cache or next_state_key in seen_keys:
                        continue
                    seen_keys.add(next_state_key)
                    scene_graph = expand.scene_graph.shallow_copy()
                    get_updated_scene_graph(scene_graph, next_state, action)
                    batch_keys.append(next_state_key)
                    batch_graphs.append(scene_graph)
                if batch_graphs:
                    for key, ev in zip(
                        batch_keys, search_params.interruption_value_batch_fn(batch_graphs)
                    ):
                        ev_cache[key] = ev

            # frontier dedup: with ev_cache now warm for every candidate above (when
            # a batch fn was available), accumulated_cost can be computed exactly
            # with no GCN/heuristic call at all -- it's the same cost formulation
            # create_child uses internally, just evaluated early. Skip create_child
            # entirely for a candidate that isn't strictly better than the best path
            # already pushed to the same resulting state this search. When ev_cache
            # can't be trusted to already hold every candidate's value (no batch fn,
            # or no scene graph), skip this filtering and fall back to today's
            # behavior of costing every candidate via create_child.
            can_precompute_ev = (
                search_params.interruption_value_fn is None
                or (
                    search_params.interruption_value_batch_fn is not None
                    and expand.scene_graph is not None
                )
            )
            for action, next_state, next_state_key, interruption_prob in candidates:
                if can_precompute_ev:
                    interrupting_task_ev = (
                        ev_cache.get(next_state_key, 0)
                        if search_params.interruption_value_fn is not None
                        else 0
                    )
                    tentative_cost = cost_formulation.accumulated_cost(
                        expand.step_terms(action, interruption_prob, interrupting_task_ev)
                    )
                    if tentative_cost >= best_cost.get(next_state_key, float("inf")):
                        continue
                    best_cost[next_state_key] = tentative_cost

                # construct new trajectory
                child_traj = expand.create_child(
                    interruption_problem,
                    search_params,
                    action,
                    interruption_prob,
                    ev_cache,
                    heuristic_cache,
                    next_state=next_state,
                    next_state_key=next_state_key,
                )
                heapq.heappush(frontier, (child_traj, child_traj.h_value, insertion_order))
                insertion_order+=1

            pbar.update(1)

    # goal not reached, get best trajectory found
    _fill_stats(stats, interruption_problem.goal, num_expanded_nodes, num_steps, False)
    if not frontier:
        # every reachable state was expanded without reaching the goal: it is
        # unreachable with these actions, which a filtered action set makes common
        return [], float("inf"), False, initial_state[1]
    best_found, _, _ = heapq.heappop(frontier)
    return best_found.get_plan(), best_found.cost, False, best_found.scene_graph


def _fill_stats(
    stats: SearchStats | None, goal: Goal, expanded: int, budget: int, success: bool
) -> None:
    if stats is not None:
        stats.goal, stats.expanded, stats.budget, stats.success = goal, expanded, budget, success


def goal_objects(goal: Goal, all_objects: set[str]) -> set[str]:
    """The objects among all_objects that any literal of goal names."""
    return {
        arg for literal in goal.get_all_literals() for arg in literal.args if arg in all_objects
    }


def held_objects(state: State, all_objects: set[str]) -> set[str]:
    """
    The objects among all_objects that a robot is holding in state
    (holding-in-left / holding-in-right / holding fluents). A filtered search must
    keep them: with its hands full, a robot that can't put them down can't pick up
    anything else.
    """
    return {
        fluent.args[-1] for fluent in state.fluents
        if fluent.name.startswith("holding") and fluent.args and fluent.args[-1] in all_objects
    }


def filter_actions_to_objects(
    actions: list[Action], keep: set[str], all_objects: set[str]
) -> list[Action]:
    """
    The actions that involve no object of all_objects outside keep; actions
    involving no object at all (moves) are always kept. An action's arguments are
    the words of its name after the operator name.
    """
    excluded = all_objects - keep
    return [action for action in actions if excluded.isdisjoint(action.name.split(" ")[1:])]


def compute_interruption_value(
    state: State,
    actions: list[Action],
    interrupting_task_dist: tuple[list[Goal], list[float]],
    heuristic_fn: float | Callable[[State, Goal, list[Action], float], float] = 0,
    search_stats: Optional[list[SearchStats]] = None,
) -> tuple[float, Optional[list[float]]]:
    """
    Computes the expected value of a state for a task distribution.
    Returns -1 if a successful plan for one of the tasks in the task distribution
    was unable to be found. Otherwise returns the expected cost. search_stats,
    when given, gets one SearchStats appended per search run.
    """
    task_costs = []
    expected_cost = 0.0
    # setup for myopic planning approach
    search_params = PlannerConfig(
        False,
        heuristic_fn,
        1
    )

    for task, prob in zip(*interrupting_task_dist):
        # setup data structure for interruption problem where there is
        # no chance another interrupting task will arrive
        search_problem = InterruptionSearchProblem(
            task,
            actions
        )
        stats = SearchStats()
        _, cost, success, _ = astar_search(
            (state, None), search_problem, search_params, stats=stats
        )
        if search_stats is not None:
            search_stats.append(stats)

        if not success:
            return -1, None
        task_costs.append(cost)
        expected_cost += (prob * cost)
    return expected_cost, task_costs


def check_value_cache(
    state: frozenset[Fluent], value_cache: dict[frozenset[Fluent], float]
) -> bool:
    """
    Checks if the value of a state is already cached.
    """
    return state in value_cache


# debug helper functions
def print_frontier_trace(step: int, frontier: list[tuple[InterruptionTrajectory, float]]) -> None:
    """
    Prints out a trace of the trajectories currently stored in the frontier.
    """
    print(f"Planning Step: {step}")
    print(f"Frontier: # of trajectories in frontier = {len(frontier)}\n")
    for _, traj_tuple in enumerate(frontier[:5]):
        traj = traj_tuple[0]
        print(f"Value: {traj.value}")
        print(f"Discounted Cost: {traj.cost}; Plan Cost: {traj.get_plan_cost()}")
        print(f"Discounted h-value: {traj.discounted_h_value}; h-value: {traj.h_value}")

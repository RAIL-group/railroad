import heapq
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import NamedTuple, Optional
from tqdm import tqdm

from railroad._bindings import astar_search as _cpp_astar_search
# the C++ search's own successor generator, so both searches expand a state into
# the same successors (railroad.core.get_next_actions is an older Python version
# without its rule that a robot waits, no_op, only when it can do nothing else)
from railroad._bindings import get_next_actions
from railroad.core import (
    Action, Goal, State, ff_heuristic, project_action, project_state,
    relevant_predicates,
)
from railroad.environment.procthor.scenegraph import SceneGraph

from .constants import LAMBDA_ADD, LAMBDA_FF, LAMBDA_MAX, SEARCH_DEBUG
from .cost_formulations import CostFormulation, DefaultFormulation, NO_INTERRUPTION, StepTerms
from .utilities import get_action_cost, get_next_state
from .narrowing import relaxed_plan_problem
from .scene_graph import get_updated_scene_graph

def state_key(state: State) -> State:
    """
    What makes two search states the same: their fluents and their upcoming
    effects at their times relative to now, so states differing only in the
    clock are one state, but not states waiting different times for an effect.
    railroad's State hashes and compares on time, fluents and upcoming effects,
    so its zero-time copy (as MCTSPlanner roots its search) is that key.
    """
    return state.copy_and_zero_out_time()


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


@dataclass(frozen=True, kw_only=True)
class PlannerConfig:
    """
    The settings of one A* search. Frozen: to vary a setting, build a new config
    (dataclasses.replace). The planners in planners.py each build theirs.
    """
    heuristic_fn: Callable[[State, Goal, list[Action], float], float] | float
    planner_interruption_prob_fn: float | Callable[[float], float] | None = None
    interruption_value_fn: Callable[[SceneGraph], float] | None = None
    # optional batched form of interruption_value_fn: takes the scene graphs of an
    # expanded node's new successor states and returns one value per graph, in
    # order, in a single model call (see _PythonSearch.learned_values); when None,
    # interruption_value_fn is called once per state.
    interruption_value_batch_fn: Callable[[list[SceneGraph]], list[float]] | None = None
    # how a search step's costs are computed, including its discount (see
    # cost_formulations.py); the default is undiscounted
    cost_formulation: CostFormulation = field(default_factory=DefaultFormulation)

    # run eligible searches with the C++ A* (see astar_search); False forces the
    # Python implementation, e.g. to compare the two
    use_cpp_search: bool = True
    # search first over only the objects and locations the relaxed plan for the
    # goal uses, falling back to the full problem if that fails (see
    # relaxed_plan_problem)
    relaxed_plan_filtering: bool = False
    # interruption-aware planners: search first over the current task's relaxed
    # plan plus a halo of nearby objects future tasks need and where they need
    # them, falling back to the full problem (see planning_framework.interruption_halo)
    interruption_halo_filtering: bool = False


@dataclass(frozen=True)
class FFHeuristic:
    """
    h = multiplier * ff_heuristic(LAMBDA_ADD, LAMBDA_MAX, LAMBDA_FF) + v_ap, the
    learned value of the state, which astar_search passes only for a search with
    an interruption_value_fn (0 otherwise). Being a plain object rather than a
    closure lets astar_search hand it to the C++ search.
    """
    multiplier: float = 1

    def __call__(self, state: State, goal: Goal, actions: list[Action], v_ap: float = 0) -> float:
        h_val = ff_heuristic(state, goal, actions, LAMBDA_ADD, LAMBDA_MAX, LAMBDA_FF)
        return self.multiplier * h_val + v_ap


@dataclass
class SearchStats:
    """How much of its expansion budget one astar_search call used."""
    goal: Goal | None = None
    expanded: int = 0
    budget: int = 0
    success: bool = False
    # a retry over only the goal's own objects, after an unfiltered search failed
    filtered: bool = False
    # actions in the relaxed-plan-filtered problem this search ran on; 0 when it
    # ran on the full problem (see relaxed_plan_problem)
    relaxed_plan_actions: int = 0

    @property
    def fraction(self) -> float:
        return self.expanded / self.budget if self.budget else 0.0


@dataclass
class SearchNode:
    """
    A node of the Python A* search: a state, the action and parent node that
    reached it, and its costs under the search's cost formulation.
    """
    state: State
    # the action that reached state from the parent's state; None at the root
    action: Optional[Action]
    # running discount of the path so far (gamma_t), applied to the next step's costs;
    # see cost_formulations.py for how each step updates it
    discount_factor: float
    scene_graph: SceneGraph | None
    # accumulated cost g; value is the priority, g + discounted_h_value
    cost: float = 0.0
    value: float = 0.0
    discounted_h_value: float = 0.0
    h_value: float = 0.0
    parent: Optional["SearchNode"] = None
    # the state's state_key, shared with the search's caches (see _PythonSearch.key)
    key: State | None = None

    def step_terms(
        self, action: Action, interruption_prob: float, interrupting_task_ev: float
    ) -> StepTerms:
        """
        The inputs a cost formulation reads for the step from this node
        through action.
        """
        return StepTerms(
            parent_cost=self.cost,
            parent_discount=self.discount_factor,
            action_cost=get_action_cost(action),
            interruption_prob=interruption_prob,
            interrupting_task_ev=interrupting_task_ev,
        )

    def get_plan(self) -> list[Action]:
        """
        Recovers the plan that resulted in the node.
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
        Returns actual cost of a node's plan, without factoring the interruption probabilities.
        """
        plan_cost = 0
        for act in self.get_plan():
            plan_cost+=get_action_cost(act)
        return plan_cost


class SearchResult(NamedTuple):
    """What astar_search returns; unpacks as (plan, cost, success, scene_graph)."""
    plan: list[Action]
    # the plan's accumulated cost under the search's cost formulation
    cost: float
    # whether the plan reaches the goal; if not, it's the best partial plan found
    success: bool
    # the scene graph after the plan
    scene_graph: SceneGraph | None


def astar_search(
    initial_state: tuple[State, SceneGraph | None],
    search_problem: InterruptionSearchProblem,
    config: PlannerConfig,
    num_steps: int = 30000,
    show_progress: bool = False,
    stats: SearchStats | None = None,
) -> SearchResult:
    """
    Astar algorithm implementation. show_progress prints the goal and a progress
    bar over num_steps; off by default, since callers like data generation run
    many searches across parallel workers. stats, when given, is filled in with
    the nodes expanded against the num_steps budget.

    Searches with no learned value function, the default cost formulation and a
    constant or FFHeuristic heuristic run in C++ (railroad's astar.hpp), with
    identical results; others, and config.use_cpp_search=False, run here.
    """
    if config.relaxed_plan_filtering:
        narrowed = relaxed_plan_problem(initial_state[0], search_problem)
        if narrowed is not None:
            result = astar_search(
                initial_state, narrowed, replace(config, relaxed_plan_filtering=False),
                num_steps, show_progress, stats,
            )
            if stats is not None:
                stats.relaxed_plan_actions = len(narrowed.actions)
            if result.success:
                return result
        # otherwise: nothing to narrow, or the narrowed problem failed - solve it all
        if stats is not None:
            stats.relaxed_plan_actions = 0

    if config.use_cpp_search and _cpp_search_applies(config):
        return _astar_search_cpp(
            initial_state, search_problem, config, num_steps, show_progress, stats
        )
    return _PythonSearch(search_problem, config).run(initial_state, num_steps, show_progress, stats)


class _Candidate(NamedTuple):
    """A successor of an expanded node, before it's costed."""
    action: Action
    state: State
    key: State
    # the chance a task arrives during action
    interruption_prob: float


class _PythonSearch:
    """
    astar_search in Python, for the searches the C++ one can't run: those with a
    learned value function, or another cost formulation or heuristic. It holds
    what every node of one search shares: the problem, the settings, and caches
    keyed by state_key -- each state's learned value and heuristic value, and the
    cheapest accumulated cost a path to it has had. The caches are exact: the
    learned value reads only the fluents (through the scene graph), and
    ff_heuristic the fluents and the upcoming effects relative to the state's
    time, all of which the key holds.
    """

    def __init__(self, search_problem: InterruptionSearchProblem, config: PlannerConfig):
        self.search_problem = search_problem
        self.config = config
        self.values: dict[State, float] = {}
        self.heuristic_values: dict[State, float] = {}
        self.best_costs: dict[State, float] = {}
        # one key object per distinct state, however many paths reach it, shared by
        # every cache and node rather than each holding its own copy
        self._keys: dict[State, State] = {}

    def key(self, state: State) -> State:
        key = state_key(state)
        return self._keys.setdefault(key, key)

    def run(
        self,
        initial_state: tuple[State, SceneGraph | None],
        num_steps: int,
        show_progress: bool,
        stats: SearchStats | None,
    ) -> SearchResult:
        state, scene_graph = initial_state
        goal = self.search_problem.goal
        expanded: set[State] = set()
        root = SearchNode(
            state=state, action=None, discount_factor=1, scene_graph=scene_graph,
            key=self.key(state),
        )
        # entries are (value, h, insertion order, node): the lowest value first, ties
        # to the lower heuristic, then to the earlier push (as astar.hpp's QueueEntry)
        frontier: list[tuple[float, float, int, SearchNode]] = [(root.value, -1, 0, root)]
        insertion_order = 1
        num_expanded_nodes = 0

        if show_progress:
            print(f"Current Goal: {goal}")
        with tqdm(total=num_steps, disable=not show_progress) as pbar:
            while num_expanded_nodes < num_steps and frontier:
                if SEARCH_DEBUG:
                    print_frontier_trace(num_expanded_nodes, frontier)

                node = heapq.heappop(frontier)[-1]
                if goal.evaluate(node.state.fluents):
                    _fill_stats(stats, goal, num_expanded_nodes, num_steps, True)
                    return SearchResult(node.get_plan(), node.cost, True, node.scene_graph)
                assert node.key is not None
                if node.key in expanded:
                    continue
                expanded.add(node.key)
                num_expanded_nodes += 1

                candidates = self.successors(node, expanded)
                for candidate, value in zip(candidates, self.learned_values(node, candidates)):
                    # a path to the state no cheaper than one already pushed would only
                    # be discarded when popped: skip building it
                    cost = self.config.cost_formulation.accumulated_cost(
                        node.step_terms(candidate.action, candidate.interruption_prob, value)
                    )
                    if cost >= self.best_costs.get(candidate.key, float("inf")):
                        continue
                    self.best_costs[candidate.key] = cost

                    child = self.child(node, candidate, value)
                    heapq.heappush(frontier, (child.value, child.h_value, insertion_order, child))
                    insertion_order += 1

                pbar.update(1)

        # goal not reached, get best node found
        _fill_stats(stats, goal, num_expanded_nodes, num_steps, False)
        if not frontier:
            # every reachable state was expanded without reaching the goal: it is
            # unreachable with these actions, which a filtered action set makes common
            return SearchResult([], float("inf"), False, scene_graph)
        best_found = heapq.heappop(frontier)[-1]
        return SearchResult(best_found.get_plan(), best_found.cost, False, best_found.scene_graph)

    def successors(self, node: SearchNode, expanded: set[State]) -> list[_Candidate]:
        """The successors of node whose states aren't expanded yet."""
        prob_fn = _arrival_prob_fn(self.config)
        candidates = []
        for action in get_next_actions(node.state, self.search_problem.actions):
            next_state, interruption_prob = get_next_state(node.state, action, prob_fn)
            key = self.key(next_state)
            if key not in expanded:
                candidates.append(_Candidate(action, next_state, key, interruption_prob))
        return candidates

    def learned_values(self, node: SearchNode, candidates: list[_Candidate]) -> list[float]:
        """
        The learned value of each candidate's state (0 without a learned value
        function), computed once per state: for all of a node's new states in one
        batched model call when the config has a batched form, one call each otherwise.
        """
        value_fn = self.config.interruption_value_fn
        if value_fn is None:
            return [0.0] * len(candidates)
        pending: dict[State, SceneGraph] = {}
        for candidate in candidates:
            if candidate.key not in self.values and candidate.key not in pending:
                pending[candidate.key] = _scene_graph_after(node, candidate)
        if pending:
            graphs = list(pending.values())
            batch_fn = self.config.interruption_value_batch_fn
            values = batch_fn(graphs) if batch_fn is not None else [value_fn(g) for g in graphs]
            self.values.update(zip(pending, values))
        return [self.values[candidate.key] for candidate in candidates]

    def child(self, node: SearchNode, candidate: _Candidate, value: float) -> SearchNode:
        """
        The node candidate leads to from node, value being the learned value of its
        state. Its costs come from the config's cost formulation (see
        cost_formulations.py), its heuristic from the config's heuristic_fn.
        """
        formulation = self.config.cost_formulation
        terms = node.step_terms(candidate.action, candidate.interruption_prob, value)
        cost = formulation.accumulated_cost(terms)

        heuristic_fn = self.config.heuristic_fn
        if isinstance(heuristic_fn, (int, float)):
            h_value = heuristic_fn
        elif candidate.key in self.heuristic_values:
            h_value = self.heuristic_values[candidate.key]
        else:
            h_value = heuristic_fn(
                candidate.state, self.search_problem.goal, self.search_problem.actions, value
            )
            self.heuristic_values[candidate.key] = h_value

        discount_factor = formulation.child_discount(terms)
        discounted_h_value = formulation.estimated_future_cost(terms, discount_factor, h_value)
        return SearchNode(
            state=candidate.state,
            action=candidate.action,
            discount_factor=discount_factor,
            scene_graph=(
                None if node.scene_graph is None else _scene_graph_after(node, candidate)
            ),
            cost=cost,
            value=cost + discounted_h_value,
            discounted_h_value=discounted_h_value,
            h_value=h_value,
            parent=node,
            key=candidate.key,
        )


def _scene_graph_after(node: SearchNode, candidate: _Candidate) -> SceneGraph:
    """node's scene graph after candidate's action, as a copy; node's own is left as is."""
    assert node.scene_graph is not None
    scene_graph = node.scene_graph.shallow_copy()
    get_updated_scene_graph(scene_graph, candidate.state, candidate.action)
    return scene_graph


def _arrival_prob_fn(config: PlannerConfig) -> float | Callable[[float], float]:
    """The planner's chance a task arrives during an action (0 without an arrival model)."""
    prob_fn = config.planner_interruption_prob_fn
    return 0 if prob_fn is None else prob_fn


def _cpp_search_applies(config: PlannerConfig) -> bool:
    """Whether the C++ A* computes exactly what the Python one would."""
    return (
        config.interruption_value_fn is None
        and type(config.cost_formulation) is DefaultFormulation
        and (
            isinstance(config.heuristic_fn, (int, float))
            or isinstance(config.heuristic_fn, FFHeuristic)
        )
    )


# (id of an action list, relevant predicates) -> (that list, projected actions, costs);
# searches reuse one action list many times, so its projection is built once
_CachedProjection = tuple[list[Action], list[Action], list[float]]
_PROJECTION_CACHE: dict[tuple[int, frozenset[str]], _CachedProjection] = {}
_PROJECTION_CACHE_SIZE = 8


def _projected_actions(
    actions: list[Action], relevant: frozenset[str]
) -> tuple[list[Action], list[float]]:
    """
    The actions without effects on predicates nothing reads (railroad's relevance
    projection, as MCTSPlanner uses it), and each action's cost, cached per list.
    """
    key = (id(actions), relevant)
    cached = _PROJECTION_CACHE.get(key)
    if cached is not None and cached[0] is actions:
        return cached[1], cached[2]
    projected = [project_action(action, relevant) for action in actions]
    costs = [get_action_cost(action) for action in actions]
    if len(_PROJECTION_CACHE) >= _PROJECTION_CACHE_SIZE:
        _PROJECTION_CACHE.pop(next(iter(_PROJECTION_CACHE)))
    _PROJECTION_CACHE[key] = (actions, projected, costs)
    return projected, costs


def _astar_search_cpp(
    initial_state: tuple[State, SceneGraph | None],
    search_problem: InterruptionSearchProblem,
    config: PlannerConfig,
    num_steps: int,
    show_progress: bool,
    stats: SearchStats | None,
) -> SearchResult:
    """
    astar_search in C++. States and actions are first projected onto the
    predicates the search can read (bisimulation-preserving, so the plan is
    unchanged), and the scene graph, which the C++ search doesn't track, is
    rebuilt afterwards by replaying the plan.
    """
    state, scene_graph = initial_state
    goal = search_problem.goal
    actions = search_problem.actions
    if show_progress:
        print(f"Current Goal: {goal}")

    relevant = frozenset(relevant_predicates(actions, goal, state.upcoming_effects))
    projected, costs = _projected_actions(actions, relevant)
    prob_fn = _arrival_prob_fn(config)
    probs = (
        [float(prob_fn)] * len(actions) if isinstance(prob_fn, (int, float))
        else [prob_fn(cost) for cost in costs]
    )
    heuristic = config.heuristic_fn
    use_ff = isinstance(heuristic, FFHeuristic)
    multiplier = heuristic.multiplier if isinstance(heuristic, FFHeuristic) else 1.0
    h_constant = float(heuristic) if isinstance(heuristic, (int, float)) else 0.0
    formulation = config.cost_formulation
    assert isinstance(formulation, DefaultFormulation)
    by_interruption = formulation.discount == NO_INTERRUPTION

    result = _cpp_astar_search(
        project_state(state, relevant), goal, projected, costs, probs,
        num_steps=num_steps, use_ff=use_ff, h_constant=h_constant,
        heuristic_multiplier=multiplier,
        lambda_add=LAMBDA_ADD, lambda_max=LAMBDA_MAX, lambda_ff=LAMBDA_FF,
        discount_by_interruption=by_interruption,
        fixed_discount=1.0 if by_interruption else float(formulation.discount),
        task_reward=formulation.current_task_reward,
    )
    plan = [actions[i] for i in result["plan"]]
    _fill_stats(stats, goal, result["expanded"], num_steps, result["success"])
    return SearchResult(
        plan, result["cost"], result["success"], _replay_scene_graph(state, scene_graph, plan)
    )


def _replay_scene_graph(
    state: State, scene_graph: SceneGraph | None, plan: list[Action]
) -> SceneGraph | None:
    """The scene graph after plan, updated step by step as the Python search does."""
    if scene_graph is None or not plan:
        return scene_graph
    scene_graph = scene_graph.shallow_copy()
    for action in plan:
        state, _ = get_next_state(state, action)
        get_updated_scene_graph(scene_graph, state, action)
    return scene_graph


def _fill_stats(
    stats: SearchStats | None, goal: Goal, expanded: int, budget: int, success: bool
) -> None:
    if stats is not None:
        stats.goal, stats.expanded, stats.budget, stats.success = goal, expanded, budget, success


# debug helper functions
def print_frontier_trace(
    step: int, frontier: list[tuple[float, float, int, SearchNode]]
) -> None:
    """
    Prints out a trace of the trajectories currently stored in the frontier.
    """
    print(f"Planning Step: {step}")
    print(f"Frontier: # of trajectories in frontier = {len(frontier)}\n")
    for *_, traj in frontier[:5]:
        print(f"Value: {traj.value}")
        print(f"Discounted Cost: {traj.cost}; Plan Cost: {traj.get_plan_cost()}")
        print(f"Discounted h-value: {traj.discounted_h_value}; h-value: {traj.h_value}")

import heapq
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

from railroad.core import Action, Fluent, Goal, State, get_next_actions
from railroad.environment.procthor.scenegraph import SceneGraph

from .constants import SEARCH_DEBUG
from .utilities import (
    get_action_cost,
    get_next_state,
    get_reward,
    get_discounted_value,
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


@dataclass
class InterruptionTrajectory:
    """
    Data structure used to represent search tree trajectories (paths).
    """
    state: State
    # action that was executed from the parent state that resulted in the current state
    action: Optional[Action]
    no_interruption_prob: float
    scene_graph: SceneGraph | None
    cost: float = 0.0
    value: float = 0.0
    discounted_h_value: float = 0.0
    h_value: float = 0.0
    parent: Optional["InterruptionTrajectory"] = None

    def create_child(
        self,
        search_problem: InterruptionSearchProblem,
        planner_params: PlannerConfig,
        action: Action,
        interruption_prob: float,
        ev_cache: Optional[dict[frozenset[Fluent], float]] = None,
        heuristic_cache: Optional[dict[frozenset[Fluent], float]] = None,
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
        """
        next_state, _ = get_next_state(self.state, action)
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

        accumulated_cost = self.cost + get_reward(
            action,
            self.no_interruption_prob,
            interrupting_task_ev * interruption_prob
        )

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

        if planner_params.discount_by_no_int_prob:
            discount_factor = self.no_interruption_prob * (1-interruption_prob)
        else:
            assert planner_params.discount is not None
            discount_factor = self.no_interruption_prob * planner_params.discount

        estimated_future_cost = get_discounted_value(
            undiscounted_future_cost,
            discount_factor,
            planner_params.current_task_reward
        )

        return InterruptionTrajectory(
            cost=accumulated_cost,
            value=accumulated_cost+estimated_future_cost,
            state=next_state,
            action=action,
            no_interruption_prob=discount_factor,
            discounted_h_value=estimated_future_cost,
            h_value=undiscounted_future_cost,
            scene_graph=scene_graph,
            parent=self,
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
    num_steps: int = 10000,
) -> tuple[list[Action], float, bool, SceneGraph | None]:
    """
    Astar algorithm implementation.
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
                no_interruption_prob=1,
                # interruption_probs=[],
                scene_graph=initial_state[1]
            ), -1, 0
        )
    )

    # search loop
    print(f"Current Goal: {interruption_problem.goal}")
    with tqdm(total=num_steps) as pbar:
        while num_expanded_nodes < num_steps:
            # some logging functionality for debugging
            if SEARCH_DEBUG:
                print_frontier_trace(num_expanded_nodes, frontier)

            # find expansion node
            expand, _, _ = heapq.heappop(frontier)

            # check for goal condition being met
            if interruption_problem.goal.evaluate(expand.state.fluents):
                return expand.get_plan(), expand.cost, True, expand.scene_graph#, curr_state

            # check if we've already expanded this state
            if expand.state.fluents in expanded:
                continue
            # otherwise add it
            expanded.add(frozenset(expand.state.fluents))
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

                next_state_key = frozenset(next_state.fluents)
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
            # with no GCN/heuristic call at all -- it's the same get_reward call
            # create_child makes internally, just evaluated early. Skip create_child
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
                    tentative_cost = expand.cost + get_reward(
                        action,
                        expand.no_interruption_prob,
                        interrupting_task_ev * interruption_prob
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
                    heuristic_cache
                )
                heapq.heappush(frontier, (child_traj, child_traj.h_value, insertion_order))
                insertion_order+=1

            pbar.update(1)

    # goal not reached, get best trajectory found
    best_found, _, _ = heapq.heappop(frontier)
    return best_found.get_plan(), best_found.cost, False, best_found.scene_graph


def compute_interruption_value(
    state: State,
    actions: list[Action],
    interrupting_task_dist: tuple[list[Goal], list[float]],
    heuristic_fn: float | Callable[[State, Goal, list[Action], float], float] = 0,
) -> tuple[float, Optional[list[float]]]:
    """
    Computes the expected value of a state for a task distribution.
    Returns -1 if a successful plan for one of the tasks in the task distribution
    was unable to be found. Otherwise returns the expected cost.
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
        _, cost, success, _ = astar_search(
            (state, None),
            search_problem,
            search_params
        )

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

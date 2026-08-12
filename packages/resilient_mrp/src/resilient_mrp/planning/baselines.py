# The two baselines, and the map math the failure-aware leaf shares. build_route_table is Dijkstra
# for the cheapest route to a goal; best_assignment picks who takes which goal, priced by those routes.

import heapq
import math
from itertools import count
from typing import Callable, NamedTuple

from .core import (ResilientGraph, RobotProfile, compute_p_success,
                    parse_available_paths)

_EPS = 1e-6
UNREACHABLE = 1e12


# One robot's trip to one goal. so we never mix a cost from one route with a risk from another.
class RouteToGoal(NamedTuple):
    weight: float         # what djikstra is minimizing
    travel_cost: float    # actual travel time of the route
    survival: float       # probability of surviving that route


NO_ROUTE = RouteToGoal(UNREACHABLE, UNREACHABLE, 0.0)


# The two ways of pricing, each used at both levels: on one edge's cost and survival, and on a
# whole plan's makespan and team survival.

# optimistic: time only, failures ignored
def optimistic_weight(travel_cost: float, survival: float) -> float:
    return travel_cost


# cautious: -log of the chance of getting through, so the safest option wins and time never enters
def cautious_weight(travel_cost: float, survival: float) -> float:
    return -math.log(max(survival, _EPS))


# use Djikstra to precomputes the cheapest route from a goal under weigh, with the 
# real travel cost and survival carried along that route.
# open_edges restricts it to the edges a state still has, so a route never runs through one that a
# failure has closed. None means the whole graph, which is what planning from the start state sees.
def build_route_table(graph: ResilientGraph, profiles: dict[str, RobotProfile], robot: str,
                      goal: str, weigh, open_edges: set | None = None
                      ) -> dict[str, RouteToGoal]:
    profile = profiles[robot]
    incoming: dict[str, list] = {}

    # get
    for (from_node, to_node), edge in graph.edges.items():
        if open_edges is not None and (from_node, to_node) not in open_edges:
            continue
        survival = compute_p_success(profile, edge["terrain_type"], edge["hazard_severity"])
        incoming.setdefault(to_node, []).append(
            (from_node, weigh(edge["cost"], survival), edge["cost"], survival))

    routes = {goal: RouteToGoal(0.0, 0.0, 1.0)}
    frontier = [(0.0, goal)]
    while frontier:
        weight_so_far, node = heapq.heappop(frontier)
        if weight_so_far > routes[node].weight:
            continue
        onward = routes[node]
        for from_node, hop_weight, hop_cost, hop_survival in incoming.get(node, ()):
            weight = weight_so_far + hop_weight
            if weight < routes.get(from_node, NO_ROUTE).weight:
                routes[from_node] = RouteToGoal(weight, onward.travel_cost + hop_cost,
                                                onward.survival * hop_survival)
                heapq.heappush(frontier, (weight, from_node))
    return routes


# Uniform-cost search over (where each robot stands, which goals are done), priced by the same weight
# that routed its legs. Piling every goal on one robot loses by itself: that robot's legs stack up.
def best_assignment(goal_sites, robot_positions: dict, visited_goals,
                    route_to: Callable[[str, str, str], RouteToGoal],
                    weigh: Callable[[float, float], float]) -> list | None:
    outstanding = frozenset(g for g in goal_sites if g not in visited_goals)
    if not outstanding:
        return []
    if not robot_positions:
        return None

    robots = list(robot_positions)
    tiebreak = count()  # keeps heap entries orderable when two assignments weigh the same
    # loads is each robot's own running travel, so a second goal costs it both legs
    start = (tuple(sorted(robot_positions.items())), frozenset(),
             tuple((r, 0.0) for r in sorted(robots)), 1.0)
    nothing_yet = weigh(0.0, 1.0)
    frontier = [(nothing_yet, 0.0, next(tiebreak), start, [])]
    cheapest = {start: nothing_yet}

    while frontier:
        weight_so_far, _, _, state, assignment = heapq.heappop(frontier)
        positions, covered, loads, survival = state
        if weight_so_far > cheapest.get(state, UNREACHABLE):
            continue
        if covered == outstanding:
            return assignment   # uniform cost, so the first full cover popped is the cheapest

        standing_at = dict(positions)
        load_of = dict(loads)
        for robot in robots:
            from_node = standing_at[robot]
            for goal in outstanding - covered:
                route = route_to(robot, from_node, goal)
                if route.weight >= UNREACHABLE:
                    continue
                moved = dict(standing_at, **{robot: goal})
                carried = dict(load_of, **{robot: load_of[robot] + route.travel_cost})
                travel, lived = max(carried.values()), survival * route.survival
                next_state = (tuple(sorted(moved.items())), covered | {goal},
                              tuple(sorted(carried.items())), lived)
                weight = weigh(travel, lived)
                if weight < cheapest.get(next_state, UNREACHABLE):
                    cheapest[next_state] = weight
                    # travel breaks ties, so cautious still splits the goals
                    heapq.heappush(frontier, (weight, travel, next(tiebreak), next_state,
                                              assignment + [(robot, from_node, goal)]))
    return None


# baseline runs on the same concurrent railroad environment as the failure-aware but it
# commits an assignment deterministically and only replans when a robot fails, whereas the failure-
# aware planner searches the stochastic model with a lookahead over failure outcomes.
class _RoutePolicy:
    def __init__(self, graph: ResilientGraph, goal_sites, profiles: dict[str, RobotProfile],
                 weigh):
        self.graph = graph
        self.goal_sites = list(goal_sites)
        self.profiles = profiles
        self._weigh = weigh
        self._open: set | None = None
        self._routes: dict = {}
        self._assignment_cache: dict = {}
        self._rebuild(None)

    def _rebuild(self, open_edges: set | None) -> None:
        self._open = open_edges
        self._routes = {(robot, goal): build_route_table(self.graph, self.profiles, robot, goal,
                                                         self._weigh, open_edges)
                        for robot in self.profiles for goal in self.goal_sites}
        self._assignment_cache.clear()

    # Re-read which edges the state still has. A failure closes one, which can move every route and
    # can put a goal out of reach, so the tables and the committed assignment are both rebuilt.
    def observe(self, state) -> None:
        open_edges = parse_available_paths(state)
        if self._open is None or open_edges != self._open:
            self._rebuild(open_edges)

    def route_to(self, robot: str, from_node: str, goal: str) -> RouteToGoal:
        return self._routes[(robot, goal)].get(from_node, NO_ROUTE)

    # cached per (robots alive, goals done) so a robot isn't pulled off its goal mid-route
    def assign(self, robot_positions: dict, visited_goals) -> dict:
        cache_key = (frozenset(robot_positions), frozenset(visited_goals))
        if cache_key in self._assignment_cache:
            return self._assignment_cache[cache_key]
        assignment = best_assignment(self.goal_sites, robot_positions, visited_goals,
                                     self.route_to, self._weigh) or []
        queues: dict = {robot: [] for robot in robot_positions}
        for robot, _from_node, goal in assignment:
            queues[robot].append(goal)
        self._assignment_cache[cache_key] = queues
        return queues

    def _weigh_edge(self, robot: str, edge: dict) -> float:
        survival = compute_p_success(self.profiles[robot], edge["terrain_type"],
                                     edge["hazard_severity"])
        return self._weigh(edge["cost"], survival)

    # next hop toward the assigned goal; candidates limits it to the moves allowed right now
    def step_toward(self, from_node: str | None, robot: str, goal: str | None,
                    candidates=None) -> str | None:
        if from_node is None or goal is None:
            return None
        routes = self._routes[(robot, goal)]
        best_hop, best_weight = None, UNREACHABLE
        for (edge_from, edge_to), edge in self.graph.edges.items():
            if edge_from != from_node or (candidates is not None and edge_to not in candidates):
                continue
            if self._open is not None and (edge_from, edge_to) not in self._open:
                continue
            weight = self._weigh_edge(robot, edge) + routes.get(edge_to, NO_ROUTE).weight
            if weight < best_weight:
                best_weight, best_hop = weight, edge_to
        return f"risk_move {robot} {from_node} {best_hop}" if best_hop is not None else None


# optimistic: assign and route by time, assume nothing ever fails
class OptimisticPolicy(_RoutePolicy):
    def __init__(self, graph, goal_sites, profiles):
        super().__init__(graph, goal_sites, profiles, weigh=optimistic_weight)


# cautious: assign and route by safety, take the routes most likely to survive
class CautiousPolicy(_RoutePolicy):
    def __init__(self, graph, goal_sites, profiles):
        super().__init__(graph, goal_sites, profiles, weigh=cautious_weight)

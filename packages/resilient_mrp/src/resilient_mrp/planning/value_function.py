# The leaf estimate for the failure-aware planner: what finishing the mission costs from here
# It hands the outstanding goals out once, then reads that one assignment twice: how fast
# it goes off the optimistic table, how likely it is to survive off the cautious one.

from collections import OrderedDict

from .core import (ResilientGraph, RobotProfile, parse_available_paths,
                   parse_state)
from .baselines import (NO_ROUTE, UNREACHABLE, build_route_table,
                        cautious_weight, optimistic_weight)

# How many distinct edge sets to keep route tables for. The leaf runs at every node of the search
# and sibling branches close different edges, so a single current-map slot would rebuild Dijkstra
# thousands of times per call to the planner. A handful of sets covers the branching that one
# search actually explores; past that the least recently used one goes.
_MAX_CACHED_MAPS = 32


class RiskAwareCostToGo:
    def __init__(self, graph: ResilientGraph, goal_sites, profiles: dict[str, RobotProfile],
                 failure_cost: float):
        self.graph = graph
        self.profiles = profiles
        # sorted so the hand-out below cannot depend on the order the caller happened to list them
        self.goal_sites = sorted(goal_sites)
        self.failure_cost = failure_cost
        self._tables: OrderedDict = OrderedDict()

    # The fastest route to each goal, and separately the one most likely to survive, for one set of
    # open edges. A failure shuts the edge it happened on, so a table built from the whole graph
    # goes stale and can quote a route through an edge that is no longer there.
    def _tables_for(self, open_edges: frozenset):
        cached = self._tables.get(open_edges)
        if cached is not None:
            self._tables.move_to_end(open_edges)
            return cached

        usable = set(open_edges)
        built = (
            {(robot, goal): build_route_table(self.graph, self.profiles, robot, goal,
                                              optimistic_weight, usable)
             for robot in self.profiles for goal in self.goal_sites},
            {(robot, goal): build_route_table(self.graph, self.profiles, robot, goal,
                                              cautious_weight, usable)
             for robot in self.profiles for goal in self.goal_sites},
        )
        self._tables[open_edges] = built
        if len(self._tables) > _MAX_CACHED_MAPS:
            self._tables.popitem(last=False)
        return built

    # our leaf evaluator here
    def estimate(self, state) -> float:
        positions, visited, pending = parse_state(state)
        outstanding = [g for g in self.goal_sites if g not in visited]
        if not outstanding:
            return 0.0
        # return the failure cost if there are no robots left to do the work
        if not positions:
            return self.failure_cost

        fast, safe = self._tables_for(frozenset(parse_available_paths(state)))

        # One assignment, then both numbers come off it. A goal goes to whoever would finish it
        # soonest counting what they already carry, so two free robots split rather than stack up.
        # A robot still crossing an edge starts owing the rest of that crossing, and the mission
        # already owes the chance it does not survive it.
        where, load = dict(positions), {r: 0.0 for r in positions}
        survival = 1.0
        for robot, leg in pending.items():
            load[robot] = leg.remaining_time
            survival *= leg.survival

        for goal in outstanding:
            taker, finish_at = None, UNREACHABLE
            for robot, node in where.items():
                finish = load[robot] + fast[(robot, goal)].get(node, NO_ROUTE).travel_cost
                if finish < finish_at:
                    taker, finish_at = robot, finish
            if taker is None:
                return self.failure_cost   # this goal is out of everyone's reach now
            # the robot that took it, priced on its safest route rather than its fastest one
            survival *= safe[(taker, goal)].get(where[taker], NO_ROUTE).survival
            load[taker], where[taker] = finish_at, goal

        # relaxation here
        # load is the optimistic travel, survival the cautious and they combine into one cost.
        return max(load.values()) + (1.0 - survival) * self.failure_cost

    def __call__(self, state) -> float:
        return self.estimate(state)

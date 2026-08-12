# The leaf estimate for the failure-aware planner: what finishing the mission costs from here
# It hands the outstanding goals out once, then reads that one assignment twice: how fast
# it goes off the optimistic table, how likely it is to survive off the cautious one.

from .core import ResilientGraph, RobotProfile, parse_state
from .baselines import (NO_ROUTE, UNREACHABLE, build_route_table,
                        cautious_weight, optimistic_weight)


class RiskAwareCostToGo:
    def __init__(self, graph: ResilientGraph, goal_sites, profiles: dict[str, RobotProfile],
                 failure_cost: float):
        self.goal_sites = list(goal_sites)
        self.failure_cost = failure_cost
        # the fastest route to each goal, and separately the one most likely to survive
        self._fast = {(robot, goal): build_route_table(graph, profiles, robot, goal,
                                                       optimistic_weight)
                      for robot in profiles for goal in self.goal_sites}
        self._safe = {(robot, goal): build_route_table(graph, profiles, robot, goal,
                                                       cautious_weight)
                      for robot in profiles for goal in self.goal_sites}

    # our leaf evaluator here
    def estimate(self, state) -> float:
        positions, visited = parse_state(state)
        outstanding = [g for g in self.goal_sites if g not in visited]
        if not outstanding:
            return 0.0
        # return the failure cost if there are no robots left to do the work
        if not positions:
            return self.failure_cost

        # One assignment, then both numbers come off it. A goal goes to whoever would finish it
        # soonest counting what they already carry, so two free robots split rather than stack up.
        where, load = dict(positions), {r: 0.0 for r in positions}
        survival = 1.0
        for goal in outstanding:
            taker, finish_at = None, UNREACHABLE
            for robot, node in where.items():
                finish = load[robot] + self._fast[(robot, goal)].get(node, NO_ROUTE).travel_cost
                if finish < finish_at:
                    taker, finish_at = robot, finish
            if taker is None:
                return self.failure_cost   # this goal is out of everyone's reach now
            # the robot that took it, priced on its safest route rather than its fastest one
            survival *= self._safe[(taker, goal)].get(where[taker], NO_ROUTE).survival
            load[taker], where[taker] = finish_at, goal

        # relaxation here
        # load is the optimistic travel, survival the cautious and they combine into one cost.
        return max(load.values()) + (1.0 - survival) * self.failure_cost

    def __call__(self, state) -> float:
        return self.estimate(state)

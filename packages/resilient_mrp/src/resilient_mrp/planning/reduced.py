# The optimistic relaxation as a planning problem rather than an assignment search.
#
# best_assignment enumerated (positions, goals covered, per-robot loads, survival) directly. The
# loads and survival are floats, so two orderings of the same goal set almost never compare equal
# and the search degenerates into enumerating orderings: ~5x per goal added, 9.6s at ten goals, and
# re-solved once per goal completion.
#
# Instead: throw away everything between the key nodes. A robot's trip from one key node to another
# is one action whose duration is the precomputed shortest-path cost, failure is dropped, and what
# is left is a small deterministic planning problem that railroad's own model can express and A*
# can solve.

from railroad._bindings import astar
from railroad.core import Effect, Fluent as F, Operator, State
from railroad.environment import SymbolicEnvironment

from .baselines import build_route_table, optimistic_weight
from .core import ResilientGraph, RobotProfile

# Stands in for a pair of key nodes with no route between them. The `route` precondition already
# keeps those actions from ever applying; this only has to be finite so grounding can price them.
_NO_ROUTE_TIME = 1e6


class ReducedProblem:
    """The determinized optimistic problem over key nodes: where the robots are, and the goals."""

    def __init__(self, graph: ResilientGraph, goal_sites, profiles: dict[str, RobotProfile],
                 open_edges: set | None = None):
        self.graph = graph
        self.goal_sites = sorted(goal_sites)
        self.profiles = profiles
        self._legs: dict = {}
        for robot in profiles:
            for goal in self.goal_sites:
                table = build_route_table(graph, profiles, robot, goal,
                                          optimistic_weight, open_edges)
                for node, leg in table.items():
                    self._legs[(robot, node, goal)] = leg.travel_cost

    # One trip between key nodes, priced at the shortest-path cost between them.
    #
    # No goal is ever targeted twice. That is not an optimisation: without it a robot can shuttle
    # between two goals forever, and since every hop advances the clock no two of those states
    # compare equal, so the search never terminates.
    #
    # The guard is on `claimed`, asserted the moment a robot sets off, rather than on
    # `safely_visited`, which only becomes true on arrival. Guarding on arrival leaves a window the
    # width of the leg in which the goal is neither claimed nor visited, and the robots move
    # concurrently: both would set off for the same goal and one trip would be wasted. Measured on
    # a 3-goal ring, that is exactly what happened -- the plan visited g0 twice and g2 not at all.
    def _visit_operator(self) -> Operator:
        legs = self._legs

        def duration(robot: str, from_node: str, to_node: str) -> float:
            return legs.get((robot, from_node, to_node), _NO_ROUTE_TIME)

        return Operator(
            name="visit",
            parameters=[("?robot", "robot"), ("?from", "site"), ("?to", "site")],
            preconditions=[
                F("at ?robot ?from"),
                F("free ?robot"),
                F("route ?from ?to"),
                ~F("claimed ?to"),
            ],
            effects=[
                Effect(time=0, resulting_fluents={~F("at ?robot ?from"), ~F("free ?robot"),
                                                  F("claimed ?to")}),
                Effect(time=(duration, ["?robot", "?from", "?to"]),
                       resulting_fluents={F("at ?robot ?to"), F("free ?robot"),
                                          F("safely_visited ?to")}),
            ])

    # The problem seen from where the robots are standing now.
    def _instance(self, positions: dict, visited_goals):
        key_nodes = sorted(set(positions.values()) | set(self.goal_sites))
        fluents = set()
        for robot, node in positions.items():
            fluents |= {F(f"at {robot} {node}"), F(f"free {robot}")}
        for goal in visited_goals:
            fluents |= {F(f"safely_visited {goal}"), F(f"claimed {goal}")}
        for node in key_nodes:
            for goal in self.goal_sites:
                if node != goal and all((r, node, goal) in self._legs for r in positions):
                    fluents.add(F(f"route {node} {goal}"))

        env = SymbolicEnvironment(
            state=State(0.0, fluents, []),
            objects_by_type={"robot": set(positions), "site": set(key_nodes)},
            operators=[self._visit_operator()],
            true_object_locations={})
        return env.state, env.get_actions()

    # Every goal actually visited, so the clock at the goal is the last arrival and astar is
    # minimising the makespan. `claimed` still guards against two robots setting off for the same
    # goal, but it no longer has to double as the goal itself.
    def _goal(self, outstanding):
        goal = F(f"safely_visited {outstanding[0]}")
        for site in outstanding[1:]:
            goal = goal & F(f"safely_visited {site}")
        return goal

    # The plan, as (robot, from_node, goal) legs in the order A* committed them. None when the
    # goals cannot all be reached from here.
    def plan(self, positions: dict, visited_goals) -> list | None:
        outstanding = [g for g in self.goal_sites if g not in visited_goals]
        if not outstanding:
            return []
        if not positions:
            return None

        state, actions = self._instance(positions, visited_goals)
        goal = self._goal(outstanding)

        # Uniform cost. g is the state clock and the goal is stated on arrival, so this is
        # makespan-optimal; see the note at the bottom of this file for why no heuristic is used
        # and why this is still not what OptimisticPolicy runs.
        found = astar(state, actions, goal, lambda _state: 0.0)
        if found is None:
            return None
        return [tuple(action.name.split()[1:]) for action in found]

    # Makespan of that plan, which is what the optimistic relaxation is estimating.
    def cost(self, positions: dict, visited_goals) -> float | None:
        legs = self.plan(positions, visited_goals)
        if legs is None:
            return None
        load, where = {r: 0.0 for r in positions}, dict(positions)
        for robot, _from_node, goal in legs:
            load[robot] += self._legs[(robot, where[robot], goal)]
            where[robot] = goal
        return max(load.values()) if load else 0.0


# ---------------------------------------------------------------------------------------------
# Why this is still not wired into OptimisticPolicy.
#
# It is correct now. The plans are makespan-optimal, matching best_assignment -- which is exact for
# this weight -- on rings of 2 to 10 goals. That took two fixes: astar gained a visited set, and it
# gained a step that lets the clock run to the next scheduled effect when nothing else applies. The
# second one is what made the objective right. Before it, a goal stated on arrival was unreachable
# (once the last robot is dispatched, no action applies and the search dead-ends), so the goal had
# to be stated on dispatch instead -- and astar minimises the state clock, which then read the last
# dispatch rather than the last arrival. A 5-goal ring came back at 50.2 against an optimum of 37.4.
#
# What is left is cost. Speed was the whole reason to replace best_assignment, and it does not:
#
#     goals       reduced      best_assignment
#         8         2.25s                0.23s
#         9        18.48s                2.67s
#        10        49.10s                4.99s
#
# The visited set took ten goals from 163s to 49s, so this is much closer than it was, but it is
# still an order of magnitude the wrong way and both are exponential. Measured as a leaf evaluator
# at the size the experiments actually run -- 2 goals, 2 robots -- ReducedProblem.cost is 418us a
# call against best_assignment's 27us, and the leaf is called once per search node.
#
# So best_assignment stays. This module is the reference implementation of the relaxation stated as
# a planning problem: correct, checkable against the oracle, and the thing to reach for if the
# assignment search ever becomes the bottleneck rather than the cheap option.

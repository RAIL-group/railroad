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

    # Every goal claimed, not every goal visited. Nothing in this relaxation can stop a robot that
    # has set off, so a claim is an arrival that has not happened yet -- and searching for arrivals
    # would dead-end: once the last robot is dispatched no action is applicable, and raw astar has
    # no way to let the clock run the way SymbolicEnvironment.act does. Makespan is recovered from
    # the legs afterwards in cost(), so nothing is lost by stopping at the last dispatch.
    def _goal(self, outstanding):
        goal = F(f"claimed {outstanding[0]}")
        for site in outstanding[1:]:
            goal = goal & F(f"claimed {site}")
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

        # Uniform cost. See the note at the bottom of this file: the plans are valid but not yet
        # makespan-optimal, so this is not wired into OptimisticPolicy.
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
# Why this is not wired into OptimisticPolicy yet.
#
# The modelling works: the reduced problem grounds, plans, visits every goal exactly once, and on
# rings of 2 to 4 goals it returns the same makespan best_assignment does. Three things stand
# between here and replacing best_assignment, and all three are about the objective rather than
# the model.
#
# 1. The goal has to be "every goal claimed" rather than "every goal visited", because astar has no
#    way to advance the clock. Once the last robot is dispatched no action is applicable, and there
#    is no equivalent of SymbolicEnvironment.act's wait-for-the-next-event. Searching for arrivals
#    dead-ends there and astar reports no plan.
#
# 2. But astar minimises state.time, and with the goal on the claim that clock reads the last
#    dispatch, not the last arrival. So what it optimises is when the team finishes handing out
#    work, which is not the makespan. That is why rings of 5 and 6 goals come back valid but
#    suboptimal. Fixing this needs either a cost that counts committed-but-unresolved legs, or a
#    wait action fine-grained enough not to quantise the clock.
#
# 3. Neither heuristic tried is admissible for makespan, so both cost optimality rather than just
#    speed:
#
#      ff_heuristic / len(robots)   ff sums its relaxed plan, so it measures total work rather than
#                                   a schedule -- 60 for six legs of ten however many robots walk
#                                   them -- and dividing by the team size does not turn that into a
#                                   lower bound on the clock. Wrong from 5 goals up: 40.8, 50.0,
#                                   46.9, 53.0 against optima of 33.5, 30.0, 36.0, 33.0.
#
#      earliest arrival per goal    max over outstanding goals of the soonest any robot could get
#                                   there. Reads like a lower bound, and was exact to 10 goals, but
#                                   overestimates at 11: 57.5 against an optimum of 38.2. Bounds on
#                                   a concurrent model need checking well past the size they were
#                                   derived at.
#
# Speed was the reason for the migration, and uniform cost does not deliver it either: 163s at ten
# goals against 5.1s for best_assignment. So best_assignment stays the production path until there
# is an admissible heuristic and a cost that reads arrivals rather than dispatches.

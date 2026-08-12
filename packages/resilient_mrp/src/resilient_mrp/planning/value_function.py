# The leaf estimate for the failure-aware planner: what finishing the mission costs from here
# It hands the outstanding goals out once, then reads that one assignment twice: how fast
# it goes off the optimistic table, how likely it is to survive off the cautious one.

import math
from collections import OrderedDict

from .core import ResilientGraph, RobotProfile, parse_state, parse_state_and_paths
from .baselines import (NO_ROUTE, UNREACHABLE, best_assignment, build_route_table,
                        cautious_weight, optimistic_weight)

# How many distinct edge sets to keep route tables for. The leaf runs at every node of the search
# and sibling branches close different edges, so a single current-map slot would rebuild Dijkstra
# thousands of times per call to the planner. A handful of sets covers the branching that one
# search actually explores; past that the least recently used one goes.
_MAX_CACHED_MAPS = 32

# Above this team size the exact 2^robots enumeration of who survives stops being free. Nothing in
# these experiments comes close; the fallback exists so a larger team degrades rather than hangs.
_MAX_EXACT_TEAM = 12

# Same idea for the assignment. Searching the orderings is exact but exponential in outstanding
# goals, and this runs once per search node. Measured per leaf call on a 10-node instance with two
# robots: 472us at 2 goals, 536us at 3, 854us at 4, 2173us at 5. The experiments run 2 or 3, so the
# exact search is what they get; past this the one-pass hand-out takes over, which is worse but
# bounded. Raise it if the instances grow and the leaf is not the bottleneck.
_MAX_EXACT_GOALS = 4

# How many evaluated states to remember. The estimate is a pure function of the state, and the
# search asks about the same state over and over: on one 10000-iteration trial the leaf was called
# 141739 times for 7713 distinct states, so ~95% of the work was being redone. State has value
# equality, so it can key the memo directly rather than by hash. Bounded because a long mission
# keeps finding new states and nothing else would ever drop the old ones.
_MAX_CACHED_ESTIMATES = 20000


# The bounded fallback: hand each goal to whoever would finish it soonest counting what they
# already carry, and never reconsider. Cheap and order-dependent -- which is exactly why it is not
# the default. Returns legs in the same shape best_assignment does.
def _greedy_assignment(outstanding, positions: dict, load: dict, route_to) -> list | None:
    where, carried = dict(positions), dict(load)
    legs = []
    for goal in outstanding:
        taker, finish_at = None, UNREACHABLE
        for robot, node in where.items():
            finish = carried[robot] + route_to(robot, node, goal).travel_cost
            if finish < finish_at:
                taker, finish_at = robot, finish
        if taker is None:
            return None
        legs.append((taker, where[taker], goal))
        carried[taker], where[taker] = finish_at, goal
    return legs


class RiskAwareCostToGo:
    # edges_can_close says whether a failure shuts the edge it happened on. It is the difference
    # between the map being state and the map being structure, and the leaf should not have to
    # rediscover which it is by reading fluents.
    #
    # When nothing ever closes an edge the open set is the whole graph in every state the search
    # can reach, so it is known here and never read again. That is not only cheaper -- it is what
    # makes the leaf correct under relevance projection. The planner drops fluents no effect ever
    # changes from the states it searches, which is exactly what `path_available` is when nothing
    # retracts it, and the leaf is handed those projected states. Reading the open edges out of
    # one gave the empty set, so every goal looked unreachable and the leaf returned a flat
    # failure_cost on all 26909 calls of a trial -- a constant leaf, so a search with no signal.
    # It did not crash; it just always failed.
    #
    # When edges do close, `path_available` is retracted by the failure branch, which makes it
    # dynamic, which is precisely why projection keeps it -- so reading it from the state is safe
    # in the one case where reading it is necessary.
    def __init__(self, graph: ResilientGraph, goal_sites, profiles: dict[str, RobotProfile],
                 failure_cost: float, edges_can_close: bool = True):
        self.graph = graph
        self.profiles = profiles
        # sorted so the hand-out below cannot depend on the order the caller happened to list them
        self.goal_sites = sorted(goal_sites)
        self.failure_cost = failure_cost
        self._edges_can_close = edges_can_close
        self._tables: OrderedDict = OrderedDict()
        self._estimates: OrderedDict = OrderedDict()

        # Warm start. One frozenset, built once: reused as the cache key it also stops every call
        # allocating a fresh set of edges and rehashing it, since a frozenset caches its own hash.
        self._all_edges = frozenset(graph.edges.keys())
        self._tables_for(self._all_edges)

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

    # our leaf evaluator here. The answer depends on nothing but the state, and the search revisits
    # states constantly, so the whole thing is memoised -- including the fluent scan, which is most
    # of what one evaluation costs.
    def estimate(self, state) -> float:
        remembered = self._estimates.get(state)
        if remembered is not None:
            self._estimates.move_to_end(state)
            return remembered

        value = self._evaluate(state)
        self._estimates[state] = value
        if len(self._estimates) > _MAX_CACHED_ESTIMATES:
            self._estimates.popitem(last=False)
        return value

    def _evaluate(self, state) -> float:
        # one pass for all of it: at this call rate the fluent scan dominates everything else.
        # The map is only worth reading when it can change; otherwise it is the one from __init__.
        if self._edges_can_close:
            positions, visited, pending, open_edges = parse_state_and_paths(state)
            open_edges = frozenset(open_edges)
        else:
            positions, visited, pending = parse_state(state)
            open_edges = self._all_edges
        outstanding = [g for g in self.goal_sites if g not in visited]
        if not outstanding:
            return 0.0
        # No robots left to do the work: the mission is over, not merely expensive.
        #
        # This reports infinity rather than failure_cost so the search charges it as a dead end --
        # a flat cost, with the clock and the extra cost this branch already spent left out. Those
        # belong to a branch that can still finish; here nothing can, and folding them in would
        # rank a slow loss below a fast one and pull the search toward dying quickly. The planner
        # turns this back into exactly failure_cost via dead_end_penalty, which is the same number
        # the trial metric charges, so the search and the score agree on what losing costs.
        if not positions:
            return math.inf

        fast, safe = self._tables_for(open_edges)

        # One assignment, then both numbers come off it. A robot still crossing an edge starts
        # owing the rest of that crossing, and its own odds already carry the chance it does not
        # survive it, so the assignment is seeded with both.
        where, load = dict(positions), {r: 0.0 for r in positions}
        alive = {r: 1.0 for r in positions}
        for robot, leg in pending.items():
            load[robot] = leg.remaining_time
            alive[robot] *= leg.survival

        # The exact search over orderings rather than one greedy pass. A pass that hands each goal
        # to whoever would finish it soonest cannot revisit that choice, so its answer depends on
        # the order it walked the goals -- 31 against an optimum of 12 on two robots at separate
        # depots, and overshooting is the wrong direction for a term meant to be optimistic.
        route_to = lambda robot, node, goal: fast[(robot, goal)].get(node, NO_ROUTE)  # noqa: E731
        if len(outstanding) <= _MAX_EXACT_GOALS:
            assignment = best_assignment(self.goal_sites, positions, visited, route_to,
                                         optimistic_weight, initial_loads=load)
        else:
            assignment = _greedy_assignment(outstanding, positions, load, route_to)
        if assignment is None:
            # something outstanding is out of everyone's reach now, so this is lost too --
            # flat, for the same reason as the no-robots case above
            return math.inf

        for robot, _from_node, goal in assignment:
            # time off the fastest route, odds off the safest one
            load[robot] += fast[(robot, goal)].get(where[robot], NO_ROUTE).travel_cost
            alive[robot] *= safe[(robot, goal)].get(where[robot], NO_ROUTE).survival
            where[robot] = goal

        # load is the optimistic travel; the odds are the cautious side, and they combine into one
        # cost. The mission is not lost when a robot is: it is lost when the robots still standing
        # cannot between them reach everything left, which is what _mission_survival works out.
        return max(load.values()) + (1.0 - self._mission_survival(alive, positions,
                                                                  outstanding, safe)) \
            * self.failure_cost

    # The chance the mission finishes, over every way the team could come apart.
    #
    # Multiplying the robots' odds together would answer a different question -- the chance nobody
    # is lost -- and that is not what failing means here. Losing one robot only loses the mission if
    # nothing else can reach what it was carrying, and when it dies the survivors get reassigned.
    # So: enumerate which robots come through, weight each outcome by its probability, and count the
    # ones where the survivors can still cover every outstanding goal between them.
    #
    # That is 2^robots outcomes, which is nothing for the team sizes here; past _MAX_EXACT_TEAM it
    # falls back to the product, which understates the odds rather than overstating them.
    def _mission_survival(self, alive: dict, positions: dict, outstanding, safe) -> float:
        robots = sorted(alive)
        if len(robots) > _MAX_EXACT_TEAM:
            return math.prod(alive.values())

        # who could reach what, from where they stand now -- a robot redirected onto someone else's
        # goal starts from its own position, not from wherever the assignment left it
        reaches = {robot: {goal for goal in outstanding
                           if safe[(robot, goal)].get(positions[robot]) is not None}
                   for robot in robots}

        survived = 0.0
        for outcome in range(1 << len(robots)):
            standing = [r for i, r in enumerate(robots) if outcome >> i & 1]
            chance = 1.0
            for i, robot in enumerate(robots):
                chance *= alive[robot] if outcome >> i & 1 else (1.0 - alive[robot])
            if chance == 0.0:
                continue
            if all(any(goal in reaches[r] for r in standing) for goal in outstanding):
                survived += chance
        return survived

    def __call__(self, state) -> float:
        return self.estimate(state)

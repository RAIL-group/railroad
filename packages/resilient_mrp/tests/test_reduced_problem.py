# The optimistic relaxation stated as a planning problem instead of an assignment search.
#
# best_assignment is exact for the optimistic weight, so it is the oracle here: the reduced problem
# has to agree with it before it can replace it. It does, on every size these tests cover. What it
# does not yet do is beat it on cost -- see the note in reduced.py about the two heuristics that
# turned out to be inadmissible -- so it is validated but not wired into the baselines.

import math

import numpy as np
import pytest

from resilient_mrp.planning.baselines import OptimisticPolicy
from resilient_mrp.planning.core import ResilientGraph
from resilient_mrp.planning.reduced import ReducedProblem

TWO_ROBOTS = {"r1": {"t": 0.9}, "r2": {"t": 0.9}}
AT_START = {"r1": "start", "r2": "start"}


# Goals evenly spaced around a depot, every pair connected. Nothing pathological: this is the shape
# a "go and look at all of these places" mission has.
def _ring(n_goals: int, hazard: float = 0.15) -> ResilientGraph:
    graph = ResilientGraph()
    coords = {"start": np.array([0.0, 0.0])}
    for i in range(n_goals):
        angle = 2 * math.pi * i / n_goals
        coords[f"g{i}"] = np.array([10 * math.cos(angle), 10 * math.sin(angle)])
    names = list(coords)
    for i, u in enumerate(names):
        for v in names[i + 1:]:
            graph.add_edge(u, v, cost=float(np.linalg.norm(coords[u] - coords[v])),
                           terrain_type="t", hazard_severity=hazard)
    return graph


# What best_assignment's answer costs the team, walking each robot's queue in order.
def _oracle_makespan(policy: OptimisticPolicy, positions: dict, queues: dict) -> float:
    worst = 0.0
    for robot, goals in queues.items():
        node, total = positions[robot], 0.0
        for goal in goals:
            total += policy._routes[(robot, goal)][node].travel_cost
            node = goal
        worst = max(worst, total)
    return worst


def _plan(graph, goals):
    legs = ReducedProblem(graph, goals, TWO_ROBOTS).plan(AT_START, set())
    assert legs is not None, "every goal is reachable from start here"
    return legs


# Makespan-optimal, not merely valid. That holds because astar can now let the clock run when a
# robot is still crossing, so the goal is stated on arrival and the state clock at the goal is the
# makespan. Before that the goal had to be stated on dispatch, and a 5-goal ring came back at 50.2
# against an optimum of 37.4.
#
# Sizes kept small deliberately: uniform-cost search over the reduced problem is slow, and past
# about eight goals it takes seconds rather than milliseconds.
@pytest.mark.parametrize("n_goals", [2, 3, 4, 5, 6])
def test_reduced_plan_matches_the_exact_assignment(n_goals):
    goals = [f"g{i}" for i in range(n_goals)]
    graph = _ring(n_goals)

    reduced = ReducedProblem(graph, goals, TWO_ROBOTS).cost(AT_START, set())
    policy = OptimisticPolicy(graph, goals, TWO_ROBOTS)
    oracle = _oracle_makespan(policy, AT_START, policy.assign(AT_START, set()))

    assert reduced == pytest.approx(oracle), (
        f"{n_goals} goals: reduced plan finishes at {reduced:.3f}, best_assignment at {oracle:.3f}")


# Every goal gets targeted once and only once. Revisiting is never useful, and without the guard
# that rules it out the state space is unbounded -- a robot can shuttle between two goals forever,
# and since every hop advances the clock no two of those states ever compare equal. The guard has
# to bite at dispatch: on arrival it leaves a window the width of a leg in which two robots can
# both set off for the same goal.
@pytest.mark.parametrize("n_goals", [3, 5])
def test_reduced_plan_visits_every_goal_exactly_once(n_goals):
    goals = [f"g{i}" for i in range(n_goals)]
    visited = [goal for _robot, _from_node, goal in _plan(_ring(n_goals), goals)]
    assert sorted(visited) == sorted(goals), f"expected each goal once, got {visited}"


def test_reduced_problem_uses_both_robots():
    legs = _plan(_ring(2), ["g0", "g1"])
    assert {robot for robot, _from_node, _goal in legs} == {"r1", "r2"}


def test_reduced_problem_reports_goals_already_done():
    goals = ["g0", "g1"]
    problem = ReducedProblem(_ring(2), goals, TWO_ROBOTS)
    assert problem.plan(AT_START, set(goals)) == []
    assert problem.cost(AT_START, set(goals)) == 0.0


# No robots left is not the same as nothing to do.
def test_reduced_problem_reports_no_plan_without_robots():
    problem = ReducedProblem(_ring(2), ["g0", "g1"], TWO_ROBOTS)
    assert problem.plan({}, set()) is None
    assert problem.cost({}, set()) is None

# astar had no coverage anywhere in the repo, which is how it came to always return nullopt: the
# search found the goal, printed that it had, and then discarded the path because the call to
# reconstruct_path was commented out. It also overwrote whatever heuristic the caller passed in.
#
# The instance below is small enough to check by hand. Two robots start together, two goals sit ten
# units away on either side, and the hop between the goals costs nine. Splitting them finishes at
# 10; one robot walking both finishes at 19.

import pytest

from railroad._bindings import LiteralGoal, astar, ff_heuristic
from railroad.core import Effect, Fluent as F, Operator, State, transition

_LEGS = {("start", "gA"): 10.0, ("start", "gB"): 10.0,
         ("gA", "gB"): 9.0, ("gB", "gA"): 9.0}
_FAR = 1e6


def _visit_operator():
    def duration(robot, from_, to_):
        return _LEGS.get((from_, to_), _FAR)

    # ~visited guards against revisiting, which is never useful here and is what keeps the state
    # space finite: without it a robot can shuttle between the two goals forever, and since every
    # hop advances the clock no two of those states ever compare equal.
    return Operator(
        name="visit",
        parameters=[("?robot", "robot"), ("?from", "site"), ("?to", "site")],
        preconditions=[F("at ?robot ?from"), F("free ?robot"), F("route ?from ?to"),
                       ~F("visited ?to")],
        effects=[
            Effect(time=0, resulting_fluents={~F("at ?robot ?from"), ~F("free ?robot")}),
            Effect(time=(duration, ["?robot", "?from", "?to"]),
                   resulting_fluents={F("at ?robot ?to"), F("free ?robot"),
                                      F("visited ?to")}),
        ])


@pytest.fixture
def problem():
    actions = _visit_operator().instantiate(
        {"robot": ["r1", "r2"], "site": ["start", "gA", "gB"]})
    fluents = {F("at r1 start"), F("free r1"), F("at r2 start"), F("free r2")}
    fluents |= {F(f"route {u} {v}") for u, v in _LEGS}
    goal = F("visited gA") & F("visited gB")
    return State(time=0, fluents=fluents), actions, goal


# Replay a plan to get the state it actually lands in.
def _replay(state, plan):
    for action in plan:
        successors = [s for s, prob in transition(state, action) if prob > 0.0]
        assert len(successors) == 1, "this problem is deterministic"
        state = successors[0]
    return state


def test_astar_returns_a_plan(problem):
    start, actions, goal = problem
    plan = astar(start, actions, goal)
    assert plan is not None, "astar found the goal but threw the path away"
    assert len(plan) == 2, f"two goals, one visit each: {[a.name for a in plan]}"


def test_astar_plan_reaches_the_goal(problem):
    start, actions, goal = problem
    reached = _replay(start, astar(start, actions, goal))
    assert goal.evaluate(reached.fluents)


# g is the state's clock, so what astar minimises is makespan. Sending one robot to each goal
# finishes at 10; the plan that walks r1 through both finishes at 19.
def test_astar_finds_the_makespan_optimal_plan(problem):
    start, actions, goal = problem
    plan = astar(start, actions, goal)
    assert plan is not None, "the goals are reachable, so there is a plan"
    assert _replay(start, plan).time == pytest.approx(10.0), (
        f"expected the split, got {[a.name for a in plan]}")
    assert {a.name.split()[1] for a in plan} == {"r1", "r2"}, "both robots should move"


# The caller's heuristic used to be overwritten with ff_heuristic before the loop ran.
def test_astar_uses_the_heuristic_it_is_given(problem):
    start, actions, goal = problem
    seen = []

    def heuristic(state):
        seen.append(state.time)
        return 0.0

    plan = astar(start, actions, goal, heuristic)
    assert seen, "the supplied heuristic was never called"
    assert plan is not None
    assert _replay(start, plan).time == pytest.approx(10.0), (
        "zero heuristic makes this uniform-cost search, which is still optimal")


# Falling back to ff_heuristic when none is given has to keep working.
def test_astar_falls_back_to_ff_heuristic(problem):
    start, actions, goal = problem
    assert ff_heuristic(start, goal, actions) > 0.0
    assert astar(start, actions, goal) is not None


# astar takes a Goal, not a Fluent -- it has no equivalent of MCTSPlanner's _normalize_goal. It also
# has no expansion bound, so it only reports an unreachable goal once it has exhausted the reachable
# states; the ~visited guard above is what makes that a finite set.
def test_astar_returns_none_when_the_goal_is_unreachable(problem):
    start, actions, _goal = problem
    assert astar(start, actions, LiteralGoal(F("visited nowhere"))) is None

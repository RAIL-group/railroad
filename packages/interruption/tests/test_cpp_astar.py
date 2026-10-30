"""
The C++ A* (railroad's astar.hpp, used by astar_search for searches without a
learned value function) must return exactly what the Python search returns.
"""
from functools import partial
from typing import Any

import pytest

from interruption.cost_formulations import NO_INTERRUPTION, DefaultFormulation
from interruption.planner import (
    FFHeuristic, InterruptionSearchProblem, PlannerConfig, SearchStats, astar_search,
)
from interruption.task_arrival import RandomVariableType, get_task_arrival_prob
from railroad.core import Fluent as F, GroundedEffect, LiteralGoal, State
from railroad.operators.core import (
    construct_move_operator,
    construct_move_operator_blocking,
    construct_no_op_operator,
    construct_pick_operator,
    construct_place_operator,
)

LOCATIONS = {"kitchen": (0, 0), "living_room": (5, 0), "bedroom": (5, 8), "office": (0, 8)}
ARRIVAL_FN = partial(get_task_arrival_prob, RandomVariableType.CONTINUOUS, -1, 30)


def _move_time(robot, start, end):
    del robot
    (x0, y0), (x1, y1) = LOCATIONS[start], LOCATIONS[end]
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5


def _problem(blocking_moves=False):
    objects = {
        "robot": {"robot1"}, "location": set(LOCATIONS), "object": {"water_bottle", "book"},
    }
    move = (construct_move_operator_blocking if blocking_moves else construct_move_operator)(_move_time)
    actions = [
        a for op in (move, construct_pick_operator(3), construct_place_operator(2))
        for a in op.instantiate(objects)
    ]
    state = State(0, {
        F("at robot1 kitchen"), F("free robot1"),
        F("at water_bottle bedroom"), F("at book office"), ~F("hand-full robot1"),
    })
    goal = F("at water_bottle living_room") & F("at book living_room") & F("free robot1")
    return state, goal, actions


CONFIGS = {
    "constant_h": dict(heuristic_fn=0),
    "ff": dict(heuristic_fn=FFHeuristic()),
    "ff_x2": dict(heuristic_fn=FFHeuristic(multiplier=2)),
    "interruption_discount": dict(
        cost_formulation=DefaultFormulation(NO_INTERRUPTION), heuristic_fn=FFHeuristic(),
        planner_interruption_prob_fn=0.1,
    ),
    "arrival_fn_discount": dict(
        cost_formulation=DefaultFormulation(NO_INTERRUPTION), heuristic_fn=FFHeuristic(),
        planner_interruption_prob_fn=ARRIVAL_FN,
    ),
    "fixed_discount_with_reward": dict(
        heuristic_fn=FFHeuristic(),
        cost_formulation=DefaultFormulation(0.95, current_task_reward=5),
        planner_interruption_prob_fn=ARRIVAL_FN,
    ),
}


def _both(config_kwargs, goal=None, num_steps=30000, blocking_moves=False):
    state, default_goal, actions = _problem(blocking_moves)
    results = []
    for use_cpp in (False, True):
        stats = SearchStats()
        plan, cost, success, _ = astar_search(
            (state, None), InterruptionSearchProblem(goal or default_goal, actions),
            PlannerConfig(**config_kwargs, use_cpp_search=use_cpp), num_steps=num_steps,
            stats=stats,
        )
        results.append(([a.name for a in plan], cost, success, stats.expanded))
    return results


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("blocking_moves", [False, True])
def test_cpp_search_matches_python_search(name, blocking_moves):
    python, cpp = _both(CONFIGS[name], blocking_moves=blocking_moves)
    assert python[2] and cpp == python  # same plan, same cost bit for bit, same expansions


@pytest.mark.parametrize("heuristic_fn", [0, FFHeuristic()])
@pytest.mark.parametrize("arrival", [20.0, 30.0, 60.0])
def test_cpp_search_matches_python_search_while_waiting_for_a_queued_effect(
    heuristic_fn, arrival
):
    """
    The goal arrives by itself, by a queued effect, while the robot can act or
    wait (no_op). Both searches must generate the same successors: a robot waits
    only when it can do nothing else, as the C++ get_next_actions has it. (The
    Python railroad.core.get_next_actions lets it wait anyway; with it, the
    Python search found a cost-20 plan here and the C++ one cost 24.) And both
    must tell states waiting different times apart (see planner.state_key):
    keyed on fluents alone, they couldn't wait more than ~25s for the effect.
    """
    objects = {"robot": {"robot1"}, "location": {"kitchen", "hall", "office"}, "object": {"book"}}
    operators = [
        construct_move_operator_blocking(5.0), construct_pick_operator(2.0),
        construct_place_operator(2.0), construct_no_op_operator(5.0),
    ]
    actions = [a for op in operators for a in op.instantiate(objects)]
    state = State(
        0, {F("at robot1 kitchen"), F("free robot1"), F("at book kitchen"), ~F("hand-full robot1")},
        [(arrival, GroundedEffect(time=arrival, resulting_fluents={F("delivered package")}))],
    )
    problem = InterruptionSearchProblem(LiteralGoal(F("delivered package")), actions)
    results = []
    for use_cpp in (False, True):
        stats = SearchStats()
        plan, cost, success, _ = astar_search(
            (state, None), problem,
            PlannerConfig(heuristic_fn=heuristic_fn, use_cpp_search=use_cpp), stats=stats,
        )
        results.append(([a.name for a in plan], cost, success, stats.expanded))
    python, cpp = results
    assert python[2] and cpp == python
    assert python[1] == arrival  # waits no longer than the effect takes


@pytest.mark.parametrize("name", ["constant_h", "ff"])
def test_cpp_search_matches_python_search_when_the_budget_runs_out(name):
    python, cpp = _both(CONFIGS[name], num_steps=3)
    assert not python[2] and cpp == python  # same best-so-far plan and cost


def test_cpp_search_matches_python_search_when_the_goal_is_unreachable():
    python, cpp = _both(CONFIGS["ff"], goal=LiteralGoal(F("at book garage")))
    assert cpp == python == ([], float("inf"), False, python[3])


def test_only_searches_the_cpp_search_computes_identically_go_to_it():
    from interruption.planner import _cpp_search_applies

    def config(**kwargs: Any) -> PlannerConfig:
        settings: dict[str, Any] = {"heuristic_fn": FFHeuristic()}
        return PlannerConfig(**{**settings, **kwargs})

    class CustomFormulation(DefaultFormulation):
        pass

    assert _cpp_search_applies(config())
    assert _cpp_search_applies(config(heuristic_fn=0))
    assert not _cpp_search_applies(config(interruption_value_fn=lambda graph: 0.0))
    assert not _cpp_search_applies(config(heuristic_fn=lambda *args: 0.0))
    assert not _cpp_search_applies(config(cost_formulation=CustomFormulation()))


def _three_object_problem(holding_book=False):
    """_problem's domain plus a pillow, with a goal that only needs the water bottle."""
    objects = {
        "robot": {"robot1"}, "location": set(LOCATIONS),
        "object": {"water_bottle", "book", "pillow"},
    }
    actions = [
        a for op in (construct_move_operator(_move_time), construct_pick_operator(3),
                     construct_place_operator(2))
        for a in op.instantiate(objects)
    ]
    fluents = {F("at robot1 kitchen"), F("free robot1"), F("at water_bottle bedroom"),
               F("at pillow office")}
    fluents |= (
        {F("holding robot1 book"), F("hand-full robot1")} if holding_book
        else {F("at book office"), ~F("hand-full robot1")}
    )
    return State(0, fluents), LiteralGoal(F("at water_bottle living_room")), actions


def test_relaxed_plan_problem_keeps_only_what_the_relaxed_plan_uses():
    from interruption.narrowing import relaxed_plan_problem

    state, goal, actions = _three_object_problem()
    narrowed = relaxed_plan_problem(state, InterruptionSearchProblem(goal, actions))

    assert narrowed is not None
    names = [a.name for a in narrowed.actions]
    assert "pick robot1 bedroom water_bottle" in names
    # FF reaches the shared `hand-full` precondition through whichever pick is
    # cheapest, so the relaxed plan can name an object the goal doesn't need:
    # one of the book and pillow, which tie (both at the office). Narrowing keeps
    # a superset of what's needed, never too little - but not both of them.
    assert not (any("book" in n for n in names) and any("pillow" in n for n in names))
    assert len(names) < len(actions)


@pytest.mark.parametrize("holding_book", [False, True])
def test_relaxed_plan_filtering_finds_the_same_plan_cost(holding_book):
    # holding the book, the robot must put it down first: the filter keeps it
    state, goal, actions = _three_object_problem(holding_book)
    results = []
    for filtering in (False, True):
        stats = SearchStats()
        plan, cost, success, _ = astar_search(
            (state, None), InterruptionSearchProblem(goal, actions),
            PlannerConfig(heuristic_fn=FFHeuristic(), relaxed_plan_filtering=filtering), stats=stats,
        )
        results.append((success, cost, stats.relaxed_plan_actions))
    (full_ok, full_cost, _), (narrow_ok, narrow_cost, kept) = results
    assert full_ok and narrow_ok and narrow_cost == full_cost
    assert 0 < kept < len(actions)


def test_relaxed_plan_filtering_falls_back_to_the_full_problem(monkeypatch):
    from dataclasses import replace

    from interruption import planner

    # a narrowed problem that can't be solved: only moves survive
    monkeypatch.setattr(
        planner, "relaxed_plan_problem",
        lambda state, problem: replace(
            problem, actions=[a for a in problem.actions if a.name.startswith("move")]
        ),
    )
    state, goal, actions = _three_object_problem()
    stats = SearchStats()
    _, cost, success, _ = astar_search(
        (state, None), InterruptionSearchProblem(goal, actions),
        PlannerConfig(heuristic_fn=FFHeuristic(), relaxed_plan_filtering=True), stats=stats,
    )
    _, full_cost, _, _ = astar_search(
        (state, None), InterruptionSearchProblem(goal, actions),
        PlannerConfig(heuristic_fn=FFHeuristic()),
    )
    assert success and cost == full_cost and stats.relaxed_plan_actions == 0

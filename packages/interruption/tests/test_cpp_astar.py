"""
The C++ A* (railroad's astar.hpp, used by astar_search for searches without a
learned value function) must return exactly what the Python search returns.
"""
from functools import partial
from typing import Any

import pytest

from interruption.planner import (
    FFHeuristic, InterruptionSearchProblem, PlannerConfig, SearchStats, astar_search,
)
from interruption.utilities import RandomVariableType, get_task_arrival_prob
from railroad.core import Fluent as F, LiteralGoal, State
from railroad.operators.core import (
    construct_move_operator,
    construct_move_operator_blocking,
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
    "constant_h": dict(discount_by_no_int_prob=False, heuristic_fn=0, discount=1),
    "ff": dict(discount_by_no_int_prob=False, heuristic_fn=FFHeuristic(), discount=1),
    "ff_x2": dict(discount_by_no_int_prob=False, heuristic_fn=FFHeuristic(multiplier=2), discount=1),
    "interruption_discount": dict(
        discount_by_no_int_prob=True, heuristic_fn=FFHeuristic(), planner_interruption_prob_fn=0.1,
    ),
    "arrival_fn_discount": dict(
        discount_by_no_int_prob=True, heuristic_fn=FFHeuristic(),
        planner_interruption_prob_fn=ARRIVAL_FN,
    ),
    "fixed_discount_with_reward": dict(
        discount_by_no_int_prob=False, heuristic_fn=FFHeuristic(weights=(1, 1)), discount=0.95,
        planner_interruption_prob_fn=ARRIVAL_FN, current_task_reward=5,
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


@pytest.mark.parametrize("name", ["constant_h", "ff"])
def test_cpp_search_matches_python_search_when_the_budget_runs_out(name):
    python, cpp = _both(CONFIGS[name], num_steps=3)
    assert not python[2] and cpp == python  # same best-so-far plan and cost


def test_cpp_search_matches_python_search_when_the_goal_is_unreachable():
    python, cpp = _both(CONFIGS["ff"], goal=LiteralGoal(F("at book garage")))
    assert cpp == python == ([], float("inf"), False, python[3])


def test_only_searches_the_cpp_search_computes_identically_go_to_it():
    from interruption.cost_formulations import DefaultFormulation
    from interruption.planner import _cpp_search_applies

    def config(**kwargs: Any) -> PlannerConfig:
        settings: dict[str, Any] = {
            "discount_by_no_int_prob": False, "heuristic_fn": FFHeuristic(), "discount": 1,
        }
        return PlannerConfig(**{**settings, **kwargs})

    class CustomFormulation(DefaultFormulation):
        pass

    assert _cpp_search_applies(config())
    assert _cpp_search_applies(config(heuristic_fn=0))
    assert not _cpp_search_applies(config(interruption_value_fn=lambda graph: 0.0))
    assert not _cpp_search_applies(config(heuristic_fn=lambda *args: 0.0))
    assert not _cpp_search_applies(config(cost_formulation=CustomFormulation(False, 1)))


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
    from interruption.planner import relaxed_plan_problem

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
            PlannerConfig(False, FFHeuristic(), 1, relaxed_plan_filtering=filtering), stats=stats,
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
        PlannerConfig(False, FFHeuristic(), 1, relaxed_plan_filtering=True), stats=stats,
    )
    _, full_cost, _, _ = astar_search(
        (state, None), InterruptionSearchProblem(goal, actions),
        PlannerConfig(False, FFHeuristic(), 1),
    )
    assert success and cost == full_cost and stats.relaxed_plan_actions == 0

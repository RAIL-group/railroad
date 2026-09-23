import random
from types import SimpleNamespace
from typing import Any, cast

import pytest

from interruption import experiments
from railroad.core import Fluent as F, LiteralGoal


class _Env:
    def __init__(self, fluents=()):
        self.fluents = set(fluents)

    @property
    def state(self):
        return SimpleNamespace(fluents=frozenset(self.fluents))

    def get_actions(self):
        return []

    def act(self, action):
        self.fluents |= action.adds

    def update_scene_graph(self, action):
        pass


def _action(name, *adds):
    return SimpleNamespace(name=name, adds={F(a) for a in adds})


@pytest.fixture(autouse=True)
def _stub_action_lookup(monkeypatch):
    monkeypatch.setattr(experiments, "get_action_by_name", lambda _acts, name: _CURRENT[name])
    monkeypatch.setattr(experiments, "get_action_cost", lambda _action: 1.0)


_CURRENT: dict = {}


def _run(goal, plan, fluents=(), last_task_flag=True, interruption_prob=0.0):
    _CURRENT.clear()
    _CURRENT.update({a.name: a for a in plan})
    data = cast(Any, SimpleNamespace(
        env=_Env(fluents),
        search_problem=SimpleNamespace(interruption_prob_fn=interruption_prob),
    ))
    trace: list[str] = []
    completed = experiments._execution_loop((goal, True), plan, data, trace, last_task_flag)
    return completed, trace


def _goal(*literals):
    goals = [LiteralGoal(F(lit)) for lit in literals]
    result = goals[0]
    for g in goals[1:]:
        result = result & g
    return result


def test_actions_after_task_completes_do_not_recount_goals():
    # the anticipatory planner appends actions for a literal outside the task
    plan = [
        _action("place-a", "at a x"),
        _action("place-b", "at b x"),
        _action("place-extra", "at creditcard y"),
    ]
    completed, trace = _run(_goal("at a x", "at b x"), plan)

    assert completed == [LiteralGoal(F("at a x")), LiteralGoal(F("at b x"))]
    assert trace == ["place-a | Goal Complete", "place-b | Goal Complete", "place-extra"]


def test_each_goal_is_counted_once_across_a_multi_goal_plan():
    plan = [
        _action("m1", "at a x"),
        _action("m2"),
        _action("m3", "at b x"),
        _action("m4", "at c x"),
    ]
    completed, _ = _run(_goal("at a x", "at b x", "at c x"), plan)

    assert len(completed) == 3


def test_goal_already_true_with_empty_plan_is_counted_once():
    completed, trace = _run(_goal("at a x"), [], fluents={F("at a x")})

    assert completed == [LiteralGoal(F("at a x"))]
    assert trace == ["Goal Complete"]


def test_goal_true_at_start_is_counted_once_when_plan_runs():
    plan = [_action("m1"), _action("m2", "at b x"), _action("m3")]
    completed, _ = _run(_goal("at a x", "at b x"), plan, fluents={F("at a x")})

    assert len(completed) == 2


def test_empty_plan_does_not_advance_the_rng():
    random.seed(1234)
    before = random.getstate()

    completed, trace = _run(
        _goal("at a x"), [], fluents={F("at a x")}, last_task_flag=False
    )

    assert completed == [LiteralGoal(F("at a x"))]
    assert trace == ["Goal Complete"]
    assert random.getstate() == before


def test_each_executed_action_draws_once_while_tasks_are_arriving():
    plan = [_action("m1", "at a x"), _action("m2"), _action("m3")]
    random.seed(1234)
    _run(_goal("at a x"), plan, last_task_flag=False)
    after_run = random.getstate()

    random.seed(1234)
    for _ in plan:
        random.random()

    assert after_run == random.getstate()


def test_no_draws_once_all_tasks_have_arrived():
    random.seed(1234)
    before = random.getstate()

    _run(_goal("at a x"), [_action("m1", "at a x"), _action("m2")], last_task_flag=True)

    assert random.getstate() == before


def test_interruption_stops_the_plan_and_is_recorded():
    plan = [_action("m1", "at a x"), _action("m2", "at b x")]
    completed, trace = _run(
        _goal("at a x", "at b x"), plan, last_task_flag=False, interruption_prob=1.0
    )

    assert completed == [LiteralGoal(F("at a x"))]
    assert trace == ["m1 | Goal Complete | Interrupt"]

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

import interruption.planners as planners_module
from interruption.cost_formulations import NO_INTERRUPTION, DefaultFormulation
from interruption.planner import FFHeuristic, PlannerConfig
from interruption.planners import (
    AnticipatoryPlanner, InterruptionAPPlanner, InterruptionPlanner, MyopicPlanner,
)
from interruption.planning_framework import ap_heuristic_fn
from railroad.core import Fluent as F, LiteralGoal, State
from railroad.operators.core import construct_move_operator


def _ev(scene_graph) -> float:
    return 0.0


def _batched_ev(scene_graph) -> float:
    return 0.0


_batched_ev.batch = lambda graphs: [0.0] * len(graphs)  # ty: ignore[unresolved-attribute]


def test_myopic_planner_config():
    config = MyopicPlanner(h_multiplier=2).config

    assert config.heuristic_fn == FFHeuristic(multiplier=2)
    assert config.cost_formulation == DefaultFormulation(discount=1.0)
    assert config.planner_interruption_prob_fn is None
    assert config.interruption_value_fn is None
    assert config.relaxed_plan_filtering is True


def test_anticipatory_planner_config_is_myopic_with_the_value_model():
    config = AnticipatoryPlanner(ev_model=_batched_ev).config

    assert config.heuristic_fn == FFHeuristic(multiplier=1)
    assert config.cost_formulation == DefaultFormulation(discount=1.0)
    assert config.planner_interruption_prob_fn is None
    assert config.interruption_value_fn is _batched_ev
    batch = _batched_ev.batch  # ty: ignore[unresolved-attribute]
    assert config.interruption_value_batch_fn is batch
    assert config.relaxed_plan_filtering is True


@pytest.mark.parametrize("planner_cls, weights", [
    (InterruptionPlanner, None), (InterruptionAPPlanner, (1, 1)),
])
def test_interruption_planner_config(planner_cls, weights):
    config = planner_cls(ev_model=_ev, interruption_prob_fn=0.1).config

    assert config.heuristic_fn == FFHeuristic(multiplier=1, weights=weights)
    assert config.cost_formulation == DefaultFormulation(discount=NO_INTERRUPTION)
    assert config.planner_interruption_prob_fn == 0.1
    assert config.interruption_value_fn is _ev
    assert config.interruption_value_batch_fn is None  # the stub has no .batch
    # its value comes from objects the current task doesn't need
    assert config.relaxed_plan_filtering is False
    assert config.interruption_halo_filtering is False


def test_a_planner_variant_is_a_constructor_argument():
    config = InterruptionAPPlanner(ev_model=_ev, discount=1.0, halo_filtering=True).config

    assert config.cost_formulation == DefaultFormulation(discount=1.0)
    assert config.interruption_halo_filtering is True


def test_planner_configs_are_frozen():
    config = MyopicPlanner().config
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, "interruption_value_fn", _ev)
    with pytest.raises(TypeError):  # keyword-only
        PlannerConfig(0)  # ty: ignore[missing-argument, too-many-positional-arguments]


@pytest.mark.parametrize("planner", [
    MyopicPlanner(h_multiplier=3),
    AnticipatoryPlanner(ev_model=_ev, h_multiplier=3),
    InterruptionAPPlanner(ev_model=_ev, h_multiplier=3),
])
def test_myopic_keeps_the_heuristic_multiplier(planner):
    assert planner.myopic() == MyopicPlanner(h_multiplier=3)


@pytest.fixture
def captured_searches(monkeypatch):
    """astar_search stand-in recording each call's scene graph, problem and config."""
    calls = []

    def fake_astar(initial_state, problem, config, stats=None):
        calls.append(SimpleNamespace(scene_graph=initial_state[1], problem=problem, config=config))
        return ["plan"], 1.0, True, None

    monkeypatch.setattr(planners_module, "astar_search", fake_astar)
    monkeypatch.setattr(planners_module, "planning_state", lambda state, mapping: state)
    return calls


def _environment() -> Any:
    return SimpleNamespace(state=State(0, set()), scene=SimpleNamespace(scene_graph="graph"))


def test_myopic_planner_searches_without_a_scene_graph(captured_searches):
    problem: Any = SimpleNamespace(interruption_prob_fn=0.3)

    assert MyopicPlanner().plan(_environment(), problem, {}) == (["plan"], True)
    (call,) = captured_searches
    assert call.scene_graph is None
    assert call.config == MyopicPlanner().config


def test_interruption_planner_defaults_to_the_problems_arrival_model(captured_searches):
    problem: Any = SimpleNamespace(interruption_prob_fn=0.3)

    InterruptionPlanner(ev_model=_ev).plan(_environment(), problem, {})
    InterruptionPlanner(ev_model=_ev, interruption_prob_fn=0.1).plan(_environment(), problem, {})

    assert [call.config.planner_interruption_prob_fn for call in captured_searches] == [0.3, 0.1]
    assert all(call.scene_graph == "graph" for call in captured_searches)


def test_anticipatory_planner_passes_its_sampling_settings(monkeypatch):
    seen = {}

    def fake_anticipatory_planner(initial_state, problem, config, scene, mapping, **kwargs):
        seen.update(kwargs, config=config)
        return ["plan"], 1.0, True

    monkeypatch.setattr(planners_module, "anticipatory_planner", fake_anticipatory_planner)
    monkeypatch.setattr(planners_module, "planning_state", lambda state, mapping: state)
    planner = AnticipatoryPlanner(ev_model=_ev, num_samples=5, seed=3)
    problem: Any = SimpleNamespace()

    assert planner.plan(_environment(), problem, {}) == (["plan"], True)
    assert seen == {"num_samples": 5, "seed": 3, "config": planner.config}


def _move_problem():
    actions = construct_move_operator(5.0).instantiate(
        {"robot": ["r1"], "location": ["start", "target"]}
    )
    state = State(time=0, fluents={F("at r1 start"), F("free r1")})
    return state, LiteralGoal(F("at r1 target")), actions


def test_interruption_planner_heuristic_ignores_the_value():
    state, goal, actions = _move_problem()
    heuristic_fn = InterruptionPlanner(ev_model=_ev).config.heuristic_fn
    assert isinstance(heuristic_fn, FFHeuristic)

    assert heuristic_fn(state, goal, actions, 999.0) == 5.0
    assert heuristic_fn(state, goal, actions, 999.0) == ap_heuristic_fn(
        state, goal, actions, 999.0, h_multi=1
    )


def test_interruption_ap_planner_heuristic_adds_the_value():
    state, goal, actions = _move_problem()
    heuristic_fn = InterruptionAPPlanner(ev_model=_ev).config.heuristic_fn
    assert isinstance(heuristic_fn, FFHeuristic)

    assert heuristic_fn(state, goal, actions, 3.0) == pytest.approx(5.0 + 3.0)
    assert heuristic_fn(state, goal, actions, 3.0) == ap_heuristic_fn(
        state, goal, actions, 3.0, weights=(1, 1), h_multi=1
    )

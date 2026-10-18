"""
Pins astar_search's results to the cost formulation it used before
cost_formulations.py existed: GOLDEN was recorded from the pre-refactor
create_child/astar_search, so the default formulation must reproduce it exactly.
"""
from dataclasses import dataclass, replace
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest

from interruption import planner as planner_module
from interruption.cost_formulations import DefaultFormulation, StepTerms
from interruption.planner import (
    InterruptionSearchProblem,
    InterruptionTrajectory,
    PlannerConfig,
    astar_search,
)
from interruption.planning_framework import ap_heuristic_fn
from interruption.utilities import RandomVariableType, get_next_state, get_task_arrival_prob
from railroad.core import Fluent as F, State, get_next_actions
from railroad.operators.core import (
    construct_move_operator,
    construct_pick_operator,
    construct_place_operator,
)

LOCATIONS = {"kitchen": (0, 0), "living_room": (5, 0), "bedroom": (5, 8), "office": (0, 8)}


def _move_time(robot, start, end):
    del robot
    (x0, y0), (x1, y1) = LOCATIONS[start], LOCATIONS[end]
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5


def _problem():
    objects_by_type = {
        "robot": {"robot1"},
        "location": set(LOCATIONS),
        "object": {"water_bottle", "book"},
    }
    actions = [
        action
        for op in (
            construct_move_operator(_move_time),
            construct_pick_operator(3),
            construct_place_operator(2),
        )
        for action in op.instantiate(objects_by_type)
    ]
    initial_state = State(
        time=0,
        fluents={
            F("at robot1 kitchen"), F("free robot1"),
            F("at water_bottle bedroom"), F("at book office"), ~F("hand-full robot1"),
        },
    )
    goal = F("at water_bottle living_room") & F("at book living_room") & F("free robot1")
    return InterruptionSearchProblem(goal, actions), initial_state


class _StubSceneGraph(SimpleNamespace):
    """Stands in for a SceneGraph: carries the fluents of the state it describes."""

    def shallow_copy(self):
        return _StubSceneGraph(fluents=self.fluents)


def _ev(scene_graph) -> float:
    # a deterministic, state-dependent stand-in for the learned value function
    return float(sum(len(str(f)) for f in scene_graph.fluents) % 17)


def _batch_ev(scene_graphs) -> list[float]:
    return [_ev(sg) for sg in scene_graphs]


ARRIVAL_FN = partial(get_task_arrival_prob, RandomVariableType.CONTINUOUS, -1, 30)

# name -> PlannerConfig kwargs (plus "ev" / "batch" for the learned value function)
SCENARIOS: dict[str, dict[str, Any]] = {
    "myopic_constant_h": dict(discount_by_no_int_prob=False, heuristic_fn=0, discount=1),
    "myopic_ff": dict(
        discount_by_no_int_prob=False, heuristic_fn=partial(ap_heuristic_fn), discount=1
    ),
    "no_int_prob_discount": dict(
        discount_by_no_int_prob=True, heuristic_fn=partial(ap_heuristic_fn),
        planner_interruption_prob_fn=0.1,
    ),
    "arrival_fn_discount": dict(
        discount_by_no_int_prob=True, heuristic_fn=partial(ap_heuristic_fn),
        planner_interruption_prob_fn=ARRIVAL_FN,
    ),
    "fixed_discount_with_reward": dict(
        discount_by_no_int_prob=False, heuristic_fn=partial(ap_heuristic_fn), discount=0.95,
        planner_interruption_prob_fn=ARRIVAL_FN, current_task_reward=5,
    ),
    "ev_no_batch": dict(
        discount_by_no_int_prob=False, heuristic_fn=partial(ap_heuristic_fn, weights=(1, 1)),
        discount=1, planner_interruption_prob_fn=ARRIVAL_FN, ev=True,
    ),
    "ev_batch": dict(
        discount_by_no_int_prob=False, heuristic_fn=partial(ap_heuristic_fn, weights=(1, 1)),
        discount=1, planner_interruption_prob_fn=ARRIVAL_FN, ev=True, batch=True,
    ),
    "ev_batch_int_prob_discount": dict(
        discount_by_no_int_prob=True, heuristic_fn=partial(ap_heuristic_fn, weights=(1, 1)),
        planner_interruption_prob_fn=0.1, ev=True, batch=True,
    ),
}


def _setup(name: str, monkeypatch, **overrides: Any):
    kwargs: dict[str, Any] = {**SCENARIOS[name], **overrides}
    use_ev, use_batch = kwargs.pop("ev", False), kwargs.pop("batch", False)
    if use_ev:
        kwargs["interruption_value_fn"] = _ev
        kwargs["interruption_value_batch_fn"] = _batch_ev if use_batch else None
    search_problem, initial_state = _problem()
    scene_graph = None
    if use_ev:
        # the stub graph tracks the state's fluents instead of parsing action names
        monkeypatch.setattr(
            planner_module, "get_updated_scene_graph",
            lambda sg, state, action: setattr(sg, "fluents", frozenset(state.fluents)),
        )
        scene_graph = _StubSceneGraph(fluents=frozenset(initial_state.fluents))
    config: Any = PlannerConfig(**kwargs)
    return search_problem, initial_state, scene_graph, config


def run_scenario(name: str, monkeypatch, **overrides) -> tuple[list[str], float]:
    """Runs a scenario's search; returns (action names, cost)."""
    search_problem, initial_state, scene_graph, config = _setup(name, monkeypatch, **overrides)
    plan, cost, success, _ = astar_search((initial_state, scene_graph), search_problem, config)
    assert success
    return [action.name for action in plan], cost


def first_children(name: str, monkeypatch, **overrides) -> dict[str, tuple[float, ...]]:
    """
    Every child of the initial state, as create_child builds it:
    action name -> (cost, value, discount, h_value, discounted_h_value).
    """
    search_problem, initial_state, scene_graph, config = _setup(name, monkeypatch, **overrides)
    root = InterruptionTrajectory(
        state=initial_state, action=None, discount_factor=1, scene_graph=scene_graph
    )
    children = {}
    for action in get_next_actions(initial_state, search_problem.actions):
        _, interruption_prob = get_next_state(
            initial_state, action, config.planner_interruption_prob_fn or 0
        )
        child = root.create_child(search_problem, config, action, interruption_prob)
        children[action.name] = (
            child.cost, child.value, child.discount_factor,
            child.h_value, child.discounted_h_value,
        )
    return children


GOLDEN_SEARCH: dict[str, tuple[list[str], float]] = {
    'myopic_constant_h': (
        [
            'move robot1 kitchen office',
            'pick robot1 office book',
            'move robot1 office living_room',
            'place robot1 living_room book',
            'move robot1 living_room bedroom',
            'pick robot1 bedroom water_bottle',
            'move robot1 bedroom living_room',
            'place robot1 living_room water_bottle',
        ],
        43.4339811320566,
    ),
    'myopic_ff': (
        [
            'move robot1 kitchen office',
            'pick robot1 office book',
            'move robot1 office living_room',
            'place robot1 living_room book',
            'move robot1 living_room bedroom',
            'pick robot1 bedroom water_bottle',
            'move robot1 bedroom living_room',
            'place robot1 living_room water_bottle',
        ],
        43.4339811320566,
    ),
    'no_int_prob_discount': (
        [
            'move robot1 kitchen office',
            'pick robot1 office book',
            'move robot1 office living_room',
            'place robot1 living_room book',
            'move robot1 living_room bedroom',
            'pick robot1 bedroom water_bottle',
            'move robot1 bedroom living_room',
            'place robot1 living_room water_bottle',
        ],
        32.02791651696585,
    ),
    'arrival_fn_discount': (
        [
            'move robot1 kitchen office',
            'pick robot1 office book',
            'move robot1 office living_room',
            'place robot1 living_room book',
            'move robot1 living_room bedroom',
            'pick robot1 bedroom water_bottle',
            'move robot1 bedroom living_room',
            'place robot1 living_room water_bottle',
        ],
        25.850360528545018,
    ),
    'fixed_discount_with_reward': (
        [
            'move robot1 kitchen office',
            'pick robot1 office book',
            'move robot1 office living_room',
            'place robot1 living_room book',
            'move robot1 living_room bedroom',
            'pick robot1 bedroom water_bottle',
            'move robot1 bedroom living_room',
            'place robot1 living_room water_bottle',
        ],
        37.19372050136858,
    ),
    'ev_no_batch': (
        [
            'move robot1 kitchen office',
            'pick robot1 office book',
            'move robot1 office living_room',
            'place robot1 living_room book',
            'move robot1 living_room bedroom',
            'pick robot1 bedroom water_bottle',
            'move robot1 bedroom living_room',
            'place robot1 living_room water_bottle',
        ],
        53.7285372637178,
    ),
    'ev_batch': (
        [
            'move robot1 kitchen office',
            'pick robot1 office book',
            'move robot1 office living_room',
            'place robot1 living_room book',
            'move robot1 living_room bedroom',
            'pick robot1 bedroom water_bottle',
            'move robot1 bedroom living_room',
            'place robot1 living_room water_bottle',
        ],
        53.7285372637178,
    ),
    'ev_batch_int_prob_discount': (
        [
            'move robot1 kitchen office',
            'pick robot1 office book',
            'move robot1 office living_room',
            'place robot1 living_room book',
            'move robot1 living_room bedroom',
            'pick robot1 bedroom water_bottle',
            'move robot1 bedroom living_room',
            'place robot1 living_room water_bottle',
        ],
        36.48700035696586,
    ),
}

GOLDEN_FIRST_CHILDREN: dict[str, dict[str, tuple[float, ...]]] = {
    'myopic_constant_h': {
        'move robot1 kitchen bedroom': (9.433981132056603, 9.433981132056603, 1, 0, 0),
        'move robot1 kitchen living_room': (5.0, 5.0, 1, 0, 0),
        'move robot1 kitchen office': (8.0, 8.0, 1, 0, 0),
    },
    'myopic_ff': {
        'move robot1 kitchen bedroom': (9.433981132056603, 30.9339811320566, 1, 21.5, 21.5),
        'move robot1 kitchen living_room': (5.0, 32.4339811320566, 1, 27.4339811320566, 27.4339811320566),
        'move robot1 kitchen office': (8.0, 31.650971698084902, 1, 23.650971698084902, 23.650971698084902),
    },
    'no_int_prob_discount': {
        'move robot1 kitchen bedroom': (9.433981132056603, 28.783981132056603, 0.9, 21.5, 19.35),
        'move robot1 kitchen living_room': (5.0, 29.69058301885094, 0.9, 27.4339811320566, 24.69058301885094),
        'move robot1 kitchen office': (8.0, 29.285874528276413, 0.9, 23.650971698084902, 21.285874528276413),
    },
    'arrival_fn_discount': {
        'move robot1 kitchen bedroom': (9.433981132056603, 25.13282228002035, 0.7301786580448254, 21.5, 15.698841147963746),
        'move robot1 kitchen living_room': (5.0, 28.222363669279837, 0.8464817248906141, 27.4339811320566, 23.222363669279837),
        'move robot1 kitchen office': (8.0, 26.114949453423502, 0.7659283383646487, 23.650971698084902, 18.114949453423502),
    },
    'fixed_discount_with_reward': {
        'move robot1 kitchen bedroom': (9.433981132056603, 34.6089811320566, 0.95, 21.5, 25.174999999999997),
        'move robot1 kitchen living_room': (5.0, 35.812282075453766, 0.95, 27.4339811320566, 30.81228207545377),
        'move robot1 kitchen office': (8.0, 35.218423113180656, 0.95, 23.650971698084902, 27.218423113180656),
    },
    'ev_no_batch': {
        'move robot1 kitchen bedroom': (12.13219455160835, 43.63219455160835, 1, 31.5, 31.5),
        'move robot1 kitchen living_room': (7.149255851531402, 48.583236983588, 1, 41.4339811320566, 41.4339811320566),
        'move robot1 kitchen office': (10.106644954718162, 42.757616652803065, 1, 32.6509716980849, 32.6509716980849),
    },
    'ev_batch': {
        'move robot1 kitchen bedroom': (12.13219455160835, 43.63219455160835, 1, 31.5, 31.5),
        'move robot1 kitchen living_room': (7.149255851531402, 48.583236983588, 1, 41.4339811320566, 41.4339811320566),
        'move robot1 kitchen office': (10.106644954718162, 42.757616652803065, 1, 32.6509716980849, 32.6509716980849),
    },
    'ev_batch_int_prob_discount': {
        'move robot1 kitchen bedroom': (10.433981132056603, 38.7839811320566, 0.9, 31.5, 28.35),
        'move robot1 kitchen living_room': (6.4, 43.690583018850944, 0.9, 41.4339811320566, 37.290583018850946),
        'move robot1 kitchen office': (8.9, 38.285874528276416, 0.9, 32.6509716980849, 29.385874528276414),
    },
}


# exact equality: the default formulation must do the same floating-point
# operations, in the same order, as the code it replaced
@pytest.mark.parametrize("name", SCENARIOS)
def test_search_matches_pre_refactor_results(name, monkeypatch):
    assert run_scenario(name, monkeypatch) == GOLDEN_SEARCH[name]


@pytest.mark.parametrize("name", SCENARIOS)
def test_children_match_pre_refactor_values(name, monkeypatch):
    assert first_children(name, monkeypatch) == GOLDEN_FIRST_CHILDREN[name]


@pytest.mark.parametrize("name", SCENARIOS)
def test_explicit_default_formulation_matches_the_one_built_from_config_fields(name, monkeypatch):
    kwargs = {
        k: v for k, v in SCENARIOS[name].items()
        if k in ("discount_by_no_int_prob", "discount", "current_task_reward")
    }
    explicit = DefaultFormulation(**kwargs)
    assert run_scenario(name, monkeypatch, cost_formulation=explicit) == GOLDEN_SEARCH[name]


@dataclass(frozen=True)
class _DoubledActionCost(DefaultFormulation):
    def accumulated_cost(self, terms: StepTerms) -> float:
        return super().accumulated_cost(replace(terms, action_cost=2 * terms.action_cost))


@pytest.mark.parametrize("name", ["myopic_constant_h", "ev_batch"])
def test_a_custom_formulation_reaches_both_cost_call_sites(name, monkeypatch):
    # both scenarios take astar_search's pruning path, which computes a child's
    # accumulated cost before create_child does; it must use the same formulation
    calls = []
    num_children = 0
    create_child = InterruptionTrajectory.create_child

    def counting_create_child(self, *args, **kwargs):
        nonlocal num_children
        num_children += 1
        return create_child(self, *args, **kwargs)

    monkeypatch.setattr(InterruptionTrajectory, "create_child", counting_create_child)

    @dataclass(frozen=True)
    class _Recording(_DoubledActionCost):
        def accumulated_cost(self, terms: StepTerms) -> float:
            calls.append(terms)
            return super().accumulated_cost(terms)

    base = {
        k: v for k, v in SCENARIOS[name].items()
        if k in ("discount_by_no_int_prob", "discount", "current_task_reward")
    }
    plan, cost = run_scenario(name, monkeypatch, cost_formulation=_Recording(**base))

    golden_plan, golden_cost = GOLDEN_SEARCH[name]
    assert plan == golden_plan
    # undiscounted, EV-free scenario: doubling every action cost doubles the total
    if name == "myopic_constant_h":
        assert cost == pytest.approx(2 * golden_cost)
    else:
        assert cost > golden_cost
    # once per child in create_child, plus once per candidate in the pruning step
    assert len(calls) > num_children > 0

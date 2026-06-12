"""Tests for the lsp-explore / move-to-goal operators and providers."""

from __future__ import annotations

import pytest

from railroad._bindings import Fluent
from railroad.lsp import (
    FrontierProperties,
    OptimisticFrontierPropertyProvider,
    OracleFrontierPropertyProvider,
    OracleFrontierLabel,
    construct_lsp_explore_operator,
    construct_move_to_goal_operator,
)

F = Fluent


def test_optimistic_provider_constants() -> None:
    provider = OptimisticFrontierPropertyProvider(
        prob_feasible=0.7, delta_success_cost=1.0, exploration_cost=12.0
    )
    props = provider.get("robot1", "anything")
    assert props == FrontierProperties(0.7, 1.0, 12.0)


def test_oracle_provider_maps_labels() -> None:
    labels = {
        "f_yes": OracleFrontierLabel("f_yes", 1.0, 30.0, 25.0, None, "h1"),
        "f_no": OracleFrontierLabel("f_no", 0.0, None, None, 8.0, "h2"),
        "f_degenerate": OracleFrontierLabel("f_degenerate", 0.0, None, None, None, "h3"),
    }
    provider = OracleFrontierPropertyProvider(lambda: labels)

    props_yes = provider.get("robot1", "f_yes")
    assert props_yes.prob_feasible == 1.0
    # Delta success cost = true cost - optimistic cost.
    assert props_yes.delta_success_cost == pytest.approx(5.0)

    props_no = provider.get("robot1", "f_no")
    assert props_no.prob_feasible == 0.0
    assert props_no.exploration_cost == 8.0

    # Missing or degenerate labels fall back to the default.
    default = provider.get("robot1", "f_missing")
    assert provider.get("robot1", "f_degenerate") == default
    assert 0.0 < default.prob_feasible < 1.0


def test_lsp_explore_operator_grounding() -> None:
    provider = OracleFrontierPropertyProvider(lambda: {
        "f1": OracleFrontierLabel("f1", 1.0, 30.0, 25.0, None, "h1"),
        "f2": OracleFrontierLabel("f2", 0.0, None, None, 8.0, "h2"),
    })
    operator = construct_lsp_explore_operator(provider, speed_cells_per_sec=2.0)

    actions = operator.instantiate({
        "robot": {"robot1"},
        "frontier": {"f1", "f2"},
    })
    by_name = {a.name: a for a in actions}
    assert set(by_name) == {"lsp-explore robot1 f1", "lsp-explore robot1 f2"}

    action = by_name["lsp-explore robot1 f1"]

    lock_effect = action.effects[0]
    assert F("not free robot1") in lock_effect.resulting_fluents
    assert F("lock-explore f1") in lock_effect.resulting_fluents

    branch_effect = action.effects[1]
    branches = list(branch_effect.prob_effects)
    assert len(branches) == 2
    probs = sorted(p for p, _ in branches)
    assert probs[0] == pytest.approx(0.001)  # clamped failure prob
    assert probs[1] == pytest.approx(0.999)  # clamped success prob

    success_branch = next(
        effects for p, effects in branches if p == pytest.approx(0.999)
    )
    failure_branch = next(
        effects for p, effects in branches if p == pytest.approx(0.001)
    )

    # Success reveals the goal after the delta success cost — it never
    # relocates the robot.
    success_effect = success_branch[0]
    assert success_effect.time == pytest.approx((30.0 - 25.0) / 2.0)
    assert F("revealed goal") in success_effect.resulting_fluents
    assert F("explored f1") in success_effect.resulting_fluents
    assert F("free robot1") in success_effect.resulting_fluents
    assert not any(
        f.name == "at" for f in success_effect.resulting_fluents
    )

    failure_effect = failure_branch[0]
    assert F("explored f1") in failure_effect.resulting_fluents
    assert F("revealed goal") not in failure_effect.resulting_fluents

    # The infeasible frontier's failure duration comes from its
    # exploration cost.
    action_2 = by_name["lsp-explore robot1 f2"]
    branches_2 = list(action_2.effects[1].prob_effects)
    failure_effect_2 = next(
        effects for p, effects in branches_2 if p == pytest.approx(0.999)
    )[0]
    assert failure_effect_2.time == pytest.approx(8.0 / 2.0)


def test_lsp_explore_min_branch_time_floor() -> None:
    provider = OptimisticFrontierPropertyProvider(
        prob_feasible=0.8, delta_success_cost=0.0, exploration_cost=10.0
    )
    operator = construct_lsp_explore_operator(
        provider, speed_cells_per_sec=2.0, min_branch_time=0.1
    )
    (action,) = operator.instantiate({
        "robot": {"robot1"}, "frontier": {"f1"},
    })
    branches = list(action.effects[1].prob_effects)
    success_effect = next(
        effects for p, effects in branches if p == pytest.approx(0.8)
    )[0]
    # delta_success_cost 0 floors at min_branch_time rather than firing instantly
    assert success_effect.time == pytest.approx(0.1)


def test_move_to_goal_operator_grounding() -> None:
    operator = construct_move_to_goal_operator(4.0)
    actions = operator.instantiate({
        "robot": {"robot1"},
        "location": {"start", "f1"},
        "goal": {"goal"},
    })
    by_name = {a.name: a for a in actions}
    assert set(by_name) == {
        "move robot1 start goal",
        "move robot1 f1 goal",
    }

    action = by_name["move robot1 start goal"]
    assert F("revealed goal") in action.preconditions
    assert F("at robot1 start") in action.preconditions
    arrival = action.effects[1]
    assert arrival.time == pytest.approx(4.0)
    assert F("at robot1 goal") in arrival.resulting_fluents
    assert F("free robot1") in arrival.resulting_fluents

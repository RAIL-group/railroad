"""Tests for the lsp-explore / move-to-goal operators."""

from __future__ import annotations

import pytest

from railroad._bindings import Fluent
from railroad.lsp import (
    FixedPriorFrontierStatistics,
    FrontierStatistics,
    FrontierStatisticsEstimator,
    construct_lsp_explore_operator,
    construct_move_to_goal_operator,
)

F = Fluent


class _MappingStatistics(FrontierStatisticsEstimator):
    """Test estimator returning canned per-frontier statistics."""

    def __init__(self, statistics: dict[str, FrontierStatistics]) -> None:
        self._statistics = statistics

    def get(self, robot: str, frontier_id: str) -> FrontierStatistics:
        return self._statistics[frontier_id]


def _zero_goal_cost(robot: str, frontier_id: str) -> float:
    """Optimistic goal cost stub: zero, so success time == delta cost / speed."""
    return 0.0


def test_lsp_explore_operator_grounding() -> None:
    statistics = _MappingStatistics({
        "f1": FrontierStatistics(1.0, 5.0, 10.0),
        "f2": FrontierStatistics(0.0, 0.0, 8.0),
    })
    # Optimistic lower-bound cost (cells) from each frontier to the goal,
    # added back onto delta_success_cost to form the full success cost.
    goal_cost = {"f1": 6.0, "f2": 4.0}
    operator = construct_lsp_explore_operator(
        statistics, lambda r, f: goal_cost[f], speed_cells_per_sec=2.0
    )

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
    assert probs[0] == pytest.approx(0.0)  # failure prob (unclamped: [0, 1])
    assert probs[1] == pytest.approx(1.0)  # success prob

    success_branch = next(
        effects for p, effects in branches if p == pytest.approx(1.0)
    )
    failure_branch = next(
        effects for p, effects in branches if p == pytest.approx(0.0)
    )

    # Full success cost = optimistic bound (6) + delta (5) = 11 cells, so
    # success_time = 11 / 2 = 5.5s; failure_time = exploration 10 / 2 = 5s.
    # The branch point sits at min(5.5, 5.0) = 5.0, and the branch effects
    # fire at the offset from there: success at 0.5, failure at 0.0.
    assert branch_effect.time == pytest.approx(5.0)

    success_effect = success_branch[0]
    assert success_effect.time == pytest.approx(0.5)
    assert F("reachable goal") in success_effect.resulting_fluents
    assert F("revealed goal") not in success_effect.resulting_fluents
    assert F("explored f1") in success_effect.resulting_fluents
    assert F("free robot1") in success_effect.resulting_fluents
    assert not any(
        f.name == "at" for f in success_effect.resulting_fluents
    )

    failure_effect = failure_branch[0]
    assert failure_effect.time == pytest.approx(0.0)
    assert F("explored f1") in failure_effect.resulting_fluents
    assert F("reachable goal") not in failure_effect.resulting_fluents

    # The infeasible frontier's failure (exploration cost 8) completes
    # before its success (optimistic 4 + delta 0 = 4 cells → 2s), so the
    # branch point is the earlier success time and failure fires later.
    action_2 = by_name["lsp-explore robot1 f2"]
    branch_effect_2 = action_2.effects[1]
    assert branch_effect_2.time == pytest.approx(4.0 / 2.0)  # min(2, 4) = 2
    failure_effect_2 = next(
        effects for p, effects in branch_effect_2.prob_effects
        if p == pytest.approx(1.0)
    )[0]
    assert failure_effect_2.time == pytest.approx(8.0 / 2.0 - 4.0 / 2.0)


def test_lsp_explore_min_time_floor() -> None:
    statistics = FixedPriorFrontierStatistics(
        prob_feasible=0.8, delta_success_cost=0.0, exploration_cost=10.0
    )
    operator = construct_lsp_explore_operator(
        statistics, _zero_goal_cost, speed_cells_per_sec=2.0, min_time=0.1
    )
    (action,) = operator.instantiate({
        "robot": {"robot1"}, "frontier": {"f1"},
    })
    branch_effect = action.effects[1]
    # Zero optimistic cost + zero delta floors success_time at min_time;
    # the branch point (min with the 5s failure) is therefore min_time too.
    assert branch_effect.time == pytest.approx(0.1)
    success_effect = next(
        effects for p, effects in branch_effect.prob_effects
        if p == pytest.approx(0.8)
    )[0]
    # success_time floored at 0.1 == branch point, so the offset is 0.
    assert success_effect.time == pytest.approx(0.0)


def test_move_to_goal_operator_grounding() -> None:
    operator = construct_move_to_goal_operator(4.0)
    actions = operator.instantiate({
        "robot": {"robot1"},
        "location": {"start", "f1"},
        "goal": {"goal"},
    })
    by_name = {a.name: a for a in actions}
    assert set(by_name) == {
        "move-to-goal robot1 start goal",
        "move-to-goal robot1 f1 goal",
    }

    action = by_name["move-to-goal robot1 start goal"]
    # Gated on the goal being reachable (explore success) but not yet
    # revealed (directly observed); once revealed the plain move takes over.
    assert F("reachable goal") in action.preconditions
    assert ~F("revealed goal") in action.preconditions
    assert F("at robot1 start") in action.preconditions
    arrival = action.effects[1]
    assert arrival.time == pytest.approx(4.0)
    assert F("at robot1 goal") in arrival.resulting_fluents
    assert F("free robot1") in arrival.resulting_fluents

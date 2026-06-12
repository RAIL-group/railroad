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


def test_lsp_explore_operator_grounding() -> None:
    statistics = _MappingStatistics({
        "f1": FrontierStatistics(1.0, 5.0, 10.0),
        "f2": FrontierStatistics(0.0, 0.0, 8.0),
    })
    operator = construct_lsp_explore_operator(statistics, speed_cells_per_sec=2.0)

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
    assert success_effect.time == pytest.approx(5.0 / 2.0)
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
    statistics = FixedPriorFrontierStatistics(
        prob_feasible=0.8, delta_success_cost=0.0, exploration_cost=10.0
    )
    operator = construct_lsp_explore_operator(
        statistics, speed_cells_per_sec=2.0, min_branch_time=0.1
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

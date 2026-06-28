"""Golden end-to-end replay tests over hand-authored maps.

Ties the driver's bounds back to the pure ``accumulate_bounds`` formula and
asserts the lower-bound soundness property across a few topologies. The
deterministic ``explore_first`` selector keeps these exact; a final test
exercises the real MCTS production path (no scripted selector).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from railroad.lsp.frontier_statistics import FixedPriorFrontierStatistics
from railroad.replay.cost import accumulate_bounds
from railroad.replay.replay_env import run_replay

from .conftest import build_log_from_ascii, explore_first_select

MAPS = {
    "one_frontier": """
        ##########
        #S......G#
        #...?....#
        ##########
    """,
    "two_frontiers": """
        ############
        #S........G#
        #..?....?..#
        ############
    """,
    "long_corridor": """
        #############
        #S.........G#
        #.....?.....#
        #############
    """,
}


def _estimator() -> FixedPriorFrontierStatistics:
    return FixedPriorFrontierStatistics(prob_feasible=0.8)


@pytest.mark.parametrize("name", sorted(MAPS))
def test_golden_replay(name: str) -> None:
    log = build_log_from_ascii(MAPS[name])
    result = run_replay(log, _estimator(), select_action=explore_first_select)

    assert result.goal_reached
    assert result.commits, "explore-first must commit to at least one frontier"

    # The driver's bounds are exactly accumulate_bounds(commits, total_cost).
    assert result.bounds == accumulate_bounds(result.commits, result.total_cost)

    # Every commit's optimistic-to-goal is admissible and finite (goal reachable).
    for commit in result.commits:
        assert math.isfinite(commit.optimistic_to_goal)
        assert commit.optimistic_to_goal > 0.0

    # Lower-bound soundness: the optimistic bound never exceeds the actual cost
    # the (goal-reaching) replayed policy paid.
    assert result.bounds.optimistic_lb <= result.total_cost + 1e-6


@pytest.mark.parametrize("name", sorted(MAPS))
def test_golden_no_double_commit(name: str) -> None:
    log = build_log_from_ascii(MAPS[name])
    result = run_replay(log, _estimator(), select_action=explore_first_select)
    signatures = [c.frontier_signature for c in result.commits]
    assert len(signatures) == len(set(signatures))


def test_mcts_production_path_runs() -> None:
    """run_replay with the default MCTS selector reaches a clean terminal."""
    log = build_log_from_ascii(MAPS["one_frontier"])
    result = run_replay(
        log,
        _estimator(),
        max_planning_iterations=60,
        mcts_iterations=800,
    )
    assert result.termination in {"goal_reached", "no_actions", "planner_none"}
    assert np.isfinite(result.bounds.simply_connected_lb)
    assert result.total_cost >= 0.0

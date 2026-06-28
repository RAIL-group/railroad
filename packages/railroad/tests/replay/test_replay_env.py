"""Tests for the ReplayEnvironment intercept, sensing, and the run_replay driver.

GL-free and torch-free: a ``FixedPriorFrontierStatistics`` policy needs no
images, and the deterministic ``explore_first`` selector removes MCTS
stochasticity. See replay_design.md §14.
"""

from __future__ import annotations

import numpy as np
import pytest

from railroad.navigation.constants import OBSTACLE_THRESHOLD
from railroad.lsp.frontier_statistics import FixedPriorFrontierStatistics
from railroad.replay.replay_env import build_replay_env, goal_fluent, run_replay

from .conftest import build_log_from_ascii, explore_first_select

# Goal reachable through free space; one unobserved pocket -> one frontier.
MAP_ONE_FRONTIER = """
##########
#S......G#
#...?....#
##########
"""

# Two separated unobserved pockets -> two frontiers.
MAP_TWO_FRONTIERS = """
############
#S........G#
#..?....?..#
############
"""


def _estimator() -> FixedPriorFrontierStatistics:
    return FixedPriorFrontierStatistics(prob_feasible=0.8)


def _corruption_mask(env) -> np.ndarray:
    """Cells that are obstacles in the observed grid but not in the pristine map."""
    observed_obstacle = env.observed_grid >= OBSTACLE_THRESHOLD
    pristine_obstacle = env._pristine_grid >= OBSTACLE_THRESHOLD
    return observed_obstacle & ~pristine_obstacle


def _drive(env, robots, select, max_iter=120) -> str:
    """Run the plan→act loop on *env*; return the termination reason."""
    goal = goal_fluent(robots)
    termination = "max_iterations"
    for _ in range(max_iter):
        if goal.evaluate(env.state.fluents):
            return "goal_reached"
        actions = env.get_actions()
        if not actions:
            return "no_actions"
        name = select(env, actions, goal)
        if name in ("NONE", "", None):
            return "planner_none"
        from railroad.core import get_action_by_name

        env.act(get_action_by_name(actions, name))
    if goal.evaluate(env.state.fluents):
        return "goal_reached"
    return termination


# --------------------------------------------------------------------------
# Intercept & retirement
# --------------------------------------------------------------------------


def test_intercept_records_commits() -> None:
    log = build_log_from_ascii(MAP_ONE_FRONTIER)
    result = run_replay(log, _estimator(), select_action=explore_first_select)
    assert result.commits, "exploring a frontier must record a commit"
    commit = result.commits[0]
    assert commit.robot == "robot1"
    assert np.isfinite(commit.optimistic_to_goal)
    assert commit.optimistic_to_goal > 0.0
    assert commit.frontier_signature


def test_retirement_is_per_signature() -> None:
    log = build_log_from_ascii(MAP_TWO_FRONTIERS)
    result = run_replay(log, _estimator(), select_action=explore_first_select)
    signatures = [c.frontier_signature for c in result.commits]
    assert len(signatures) == len(set(signatures)), "no frontier committed twice"
    assert len(result.commits) >= 2


def test_reaches_goal_and_bounds_are_finite() -> None:
    log = build_log_from_ascii(MAP_ONE_FRONTIER)
    result = run_replay(log, _estimator(), select_action=explore_first_select)
    assert result.goal_reached
    assert np.isfinite(result.bounds.optimistic_lb)
    assert np.isfinite(result.bounds.simply_connected_lb)
    assert result.total_cost > 0.0


def test_lower_bound_soundness() -> None:
    """When the policy reaches the goal, optimistic_lb <= its actual cost."""
    log = build_log_from_ascii(MAP_TWO_FRONTIERS)
    result = run_replay(log, _estimator(), select_action=explore_first_select)
    assert result.goal_reached
    assert result.bounds.optimistic_lb <= result.total_cost + 1e-6


def test_determinism() -> None:
    log = build_log_from_ascii(MAP_TWO_FRONTIERS)
    a = run_replay(log, _estimator(), select_action=explore_first_select)
    b = run_replay(log, _estimator(), select_action=explore_first_select)
    assert a.bounds == b.bounds
    assert a.total_cost == b.total_cost
    assert [c.frontier_signature for c in a.commits] == [
        c.frontier_signature for c in b.commits
    ]


# --------------------------------------------------------------------------
# §5.1.1 sensing quirk: no map corruption, frontiers survive
# --------------------------------------------------------------------------


def test_no_corruption_after_construction() -> None:
    """Initial sensing must not paint any non-obstacle cell as an obstacle."""
    log = build_log_from_ascii(MAP_ONE_FRONTIER)
    env = build_replay_env(log, _estimator())
    assert not _corruption_mask(env).any()


def test_no_corruption_after_full_replay() -> None:
    log = build_log_from_ascii(MAP_TWO_FRONTIERS)
    env = build_replay_env(log, _estimator())
    _drive(env, log.robots, explore_first_select)
    assert not _corruption_mask(env).any()


def test_replay_uses_recorded_config_from_log() -> None:
    """The replay env rebuilds the deployment's NavigationConfig from the log,
    so it senses/maps exactly as the deployment did — there is no default."""
    import dataclasses

    from railroad.experimental.unknown_search import NavigationConfig

    log = build_log_from_ascii(MAP_ONE_FRONTIER)
    # A config distinct from the fixture's, including an int and a bool field.
    log.config = dataclasses.asdict(
        NavigationConfig(
            sensor_range=23.0,
            sensor_num_rays=91,
            move_execution_use_theta_star=False,
        )
    )
    env = build_replay_env(log, _estimator())
    assert env.config.sensor_range == 23.0
    assert env.config.sensor_num_rays == 91
    assert env.config.move_execution_use_theta_star is False


def test_missing_recorded_config_raises() -> None:
    """A log with no recorded config is an error: replay config must come from
    the deployment, never a default."""
    log = build_log_from_ascii(MAP_ONE_FRONTIER)
    log.config = {}
    with pytest.raises(ValueError, match="no recorded config"):
        build_replay_env(log, _estimator())


def test_missing_config_accepts_explicit_override() -> None:
    """The no-default rule still allows an explicit config= override."""
    from railroad.experimental.unknown_search import NavigationConfig

    log = build_log_from_ascii(MAP_ONE_FRONTIER)
    log.config = {}
    env = build_replay_env(
        log, _estimator(), config=NavigationConfig(sensor_range=42.0)
    )
    assert env.config.sensor_range == 42.0


def test_frontier_survives_confinement_sensing() -> None:
    """An unobserved pocket adjacent to free space yields a (surviving) frontier."""
    log = build_log_from_ascii(MAP_ONE_FRONTIER)
    env = build_replay_env(log, _estimator())
    assert env.frontiers, "the '?' pocket must be detected as a frontier"


def test_observed_grid_is_value_subset_of_pristine() -> None:
    """Every observed (non-unknown) cell equals the pristine map there."""
    from railroad.navigation.constants import UNOBSERVED_VAL

    log = build_log_from_ascii(MAP_TWO_FRONTIERS)
    env = build_replay_env(log, _estimator())
    _drive(env, log.robots, explore_first_select)
    observed_mask = env.observed_grid != UNOBSERVED_VAL
    np.testing.assert_array_equal(
        env.observed_grid[observed_mask], env._pristine_grid[observed_mask]
    )


# --------------------------------------------------------------------------
# Termination
# --------------------------------------------------------------------------


def test_terminates_without_hang() -> None:
    """A bounded loop always returns a definite termination reason."""
    log = build_log_from_ascii(MAP_TWO_FRONTIERS)
    result = run_replay(
        log, _estimator(), select_action=explore_first_select, max_planning_iterations=200
    )
    assert result.termination in {"goal_reached", "no_actions", "planner_none"}

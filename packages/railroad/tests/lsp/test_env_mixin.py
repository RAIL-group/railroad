"""Tests for LSPEnvironmentMixin over UnknownSpaceEnvironment (GL-free)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from railroad._bindings import Fluent, State
from railroad.environment.skill import NavigationMoveSkill
from railroad.environment.symbolic import LocationRegistry
from railroad.experimental.unknown_search import (
    NavigationConfig,
    Pose,
    UnknownSpaceEnvironment,
)
from railroad.experimental.unknown_search.operators import (
    construct_move_navigable_operator,
)
from railroad.lsp import (
    LSPEnvironmentMixin,
    OracleFrontierPropertyProvider,
    construct_lsp_explore_operator,
    construct_move_to_goal_operator,
)
from railroad.navigation.constants import COLLISION_VAL, FREE_VAL

F = Fluent

GOAL_CELL = (5, 25)


from typing import Any


class _Env(LSPEnvironmentMixin, UnknownSpaceEnvironment):
    def __init__(self, goal_cell: tuple[int, int], **kwargs: Any) -> None:
        self._lsp_goal_cell = goal_cell
        super().__init__(**kwargs)


def _branching_corridor_grid() -> np.ndarray:
    """A corridor (rows 4-6, cols 4-26) with a dead-end branch (col 9, up)."""
    grid = COLLISION_VAL * np.ones((30, 30))
    grid[4:7, 4:27] = FREE_VAL
    grid[1:4, 9] = FREE_VAL
    return grid


def _make_env(
    grid: np.ndarray,
    goal_cell: tuple[int, int] = GOAL_CELL,
    sensor_range: float = 8.0,
) -> _Env:
    env_ref: list[_Env | None] = [None]
    provider = OracleFrontierPropertyProvider(
        lambda: env_ref[0].oracle_labels if env_ref[0] is not None else {}
    )

    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        if env_ref[0] is None:
            return 5.0
        return env_ref[0].estimate_move_time_safe(robot, loc_from, loc_to)

    env = _Env(
        goal_cell=goal_cell,
        state=State(0.0, {
            F("at robot1 start"),
            F("free robot1"),
            F("revealed start"),
        }, []),
        objects_by_type={
            "robot": {"robot1"},
            "location": {"start"},
            "frontier": set(),
            "object": set(),
            "goal": set(),
        },
        operators=[
            construct_move_navigable_operator(move_time_fn),
            construct_move_to_goal_operator(move_time_fn),
            construct_lsp_explore_operator(provider, speed_cells_per_sec=2.0),
        ],
        true_grid=grid,
        robot_initial_poses={"robot1": Pose(5.0, 5.0, 0.0)},
        location_registry=LocationRegistry({"start": np.array([5, 5])}),
        skill_overrides={"move": NavigationMoveSkill},
        config=NavigationConfig(
            sensor_range=sensor_range,
            sensor_fov_rad=2 * math.pi,
            interrupt_min_new_cells=30000,
            interrupt_min_dt=30000.0,
        ),
    )
    env_ref[0] = env
    return env


def _frontier_ids_by_kind(env: _Env) -> tuple[str, str]:
    """Return (east corridor frontier id, dead-end branch frontier id)."""
    east = next(
        fid for fid, f in env.frontiers.items() if f.centroid_col > 10
    )
    branch = next(
        fid for fid, f in env.frontiers.items() if f.centroid_col <= 10
    )
    return east, branch


def _place_robot_at_frontier(env: _Env, frontier_id: str) -> None:
    frontier = env.frontiers[frontier_id]
    for fluent in list(env._fluents):
        if fluent.name == "at" and not fluent.negated and fluent.args[0] == "robot1":
            env._fluents.discard(fluent)
    env._fluents.add(F("at", "robot1", frontier_id))
    env.set_robot_pose(
        "robot1",
        Pose(float(frontier.centroid_row), float(frontier.centroid_col), 0.0),
    )


def test_oracle_labels_populated_after_init() -> None:
    env = _make_env(_branching_corridor_grid())
    assert len(env.frontiers) == 2
    east, branch = _frontier_ids_by_kind(env)

    labels = env.oracle_labels
    assert labels[east].prob_feasible == 1.0
    assert labels[east].success_cost is not None
    assert labels[branch].prob_feasible == 0.0
    assert labels[branch].exploration_cost is not None


def test_goal_object_registered_with_known_coords() -> None:
    env = _make_env(_branching_corridor_grid())
    # The goal object and its coordinates are known from the start...
    assert env.objects_by_type["goal"] == {"goal"}
    registry = env._location_registry
    assert registry is not None
    np.testing.assert_array_equal(
        registry.get("goal"), np.array(GOAL_CELL, dtype=float)
    )
    # ...but it is not yet revealed nor a regular location.
    assert F("revealed", "goal") not in env.state.fluents
    assert "goal" not in env.objects_by_type["location"]
    # Moves to the unobserved goal are filtered from the robot's current
    # location (no observed path exists).
    assert "move robot1 start goal" not in {a.name for a in env.get_actions()}


def test_resolve_probabilistic_effect_is_oracle_deterministic() -> None:
    env = _make_env(_branching_corridor_grid())
    east, branch = _frontier_ids_by_kind(env)

    for frontier_id, expect_success in ((east, True), (branch, False)):
        _place_robot_at_frontier(env, frontier_id)
        action = next(
            a for a in env.get_actions()
            if a.name == f"lsp-explore robot1 {frontier_id}"
        )
        effects, _ = env.resolve_probabilistic_effect(
            action.effects[1], env._fluents
        )
        reveals_goal = any(
            f.name == "revealed" and not f.negated and f.args[0] == "goal"
            for eff in effects
            for f in eff.resulting_fluents
        )
        assert reveals_goal == expect_success


def test_explore_success_reveals_goal_without_moving_robot() -> None:
    env = _make_env(_branching_corridor_grid())
    east, _ = _frontier_ids_by_kind(env)
    label = env.oracle_labels[east]
    assert label.success_cost is not None and label.optimistic_cost is not None
    delta = max(0.0, label.success_cost - label.optimistic_cost)

    _place_robot_at_frontier(env, east)
    pose_before = env.robot_poses["robot1"]
    action = next(
        a for a in env.get_actions()
        if a.name == f"lsp-explore robot1 {east}"
    )
    env.act(action)

    # The goal is revealed but the robot has not moved: no teleport.
    assert F("revealed", "goal") in env.state.fluents
    assert F("at", "robot1", "goal") not in env.state.fluents
    assert F("explored", east) in env.state.fluents
    assert F("free", "robot1") in env.state.fluents
    pose = env.robot_poses["robot1"]
    assert (pose.x, pose.y) == (pose_before.x, pose_before.y)
    # Action duration reflects the delta success cost (speed = 2.0).
    assert env.state.time == pytest.approx(0.1 + max(0.1, delta / 2.0))


def test_explore_failure_marks_explored_without_revealing() -> None:
    env = _make_env(_branching_corridor_grid())
    _, branch = _frontier_ids_by_kind(env)
    exploration_cost = env.oracle_labels[branch].exploration_cost
    assert exploration_cost is not None

    _place_robot_at_frontier(env, branch)
    action = next(
        a for a in env.get_actions()
        if a.name == f"lsp-explore robot1 {branch}"
    )
    env.act(action)

    assert F("revealed", "goal") not in env.state.fluents
    assert F("explored", branch) in env.state.fluents
    assert F("free", "robot1") in env.state.fluents
    assert env.state.time == pytest.approx(0.1 + exploration_cost / 2.0)


def test_observed_goal_is_revealed_and_reached_by_moving() -> None:
    # Goal within the initial sensor sweep: revealed at init.
    env = _make_env(_branching_corridor_grid(), goal_cell=(5, 10))
    assert env.goal_observed
    assert F("revealed", "goal") in env.state.fluents

    action = next(
        a for a in env.get_actions() if a.name == "move robot1 start goal"
    )
    env.act(action)

    # The robot physically navigated to the goal cell; the fluent
    # survives the post-act frontier refresh.
    assert F("at", "robot1", "goal") in env.state.fluents
    pose = env.robot_poses["robot1"]
    assert (round(pose.x), round(pose.y)) == (5, 10)
    assert env.state.time > 0.0

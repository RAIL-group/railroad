"""Tests for the explore-frontier operator and exploration-complete fluent."""

from __future__ import annotations

import math

import numpy as np

from railroad._bindings import Fluent, State
from railroad.environment.skill import NavigationMoveSkill
from railroad.environment.symbolic import LocationRegistry
from railroad.experimental.unknown_search import (
    NavigationConfig,
    Pose,
    UnknownSpaceEnvironment,
    construct_explore_frontier_operator,
)
from railroad.experimental.unknown_search.operators import (
    construct_move_navigable_operator,
)
from railroad.navigation.constants import COLLISION_VAL, FREE_VAL

from env_helpers import env_with_operators

F = Fluent


def _make_env(grid: np.ndarray, sensor_range: float) -> UnknownSpaceEnvironment:
    return env_with_operators(UnknownSpaceEnvironment,
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
        },
        operators=[
            construct_move_navigable_operator(5.0),
            construct_explore_frontier_operator(explore_time=2.0, completion_prob=0.5),
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


def _corridor_grid() -> np.ndarray:
    """A long free corridor: rows 4-6, cols 4-26 in a 30x30 grid."""
    grid = COLLISION_VAL * np.ones((30, 30))
    grid[4:7, 4:27] = FREE_VAL
    return grid


def test_explore_action_grounds_for_frontiers() -> None:
    """With unknown space, frontiers exist and explore actions ground."""
    env = _make_env(_corridor_grid(), sensor_range=8.0)

    frontiers = env.objects_by_type.get("frontier", set())
    assert frontiers, "short sensor range should leave a frontier"
    assert not F("exploration-complete").evaluate(env.state.fluents)

    explore_actions = [
        a for a in env.get_actions() if a.name.startswith("explore robot1")
    ]
    assert {a.name for a in explore_actions} == {
        f"explore robot1 {f}" for f in frontiers
    }


def test_explore_operator_effect_fluents() -> None:
    """Explore locks the frontier, then marks it explored and frees the robot."""
    env = _make_env(_corridor_grid(), sensor_range=8.0)
    frontier = next(iter(env.objects_by_type["frontier"]))
    action = next(
        a for a in env.get_actions() if a.name == f"explore robot1 {frontier}"
    )

    start_fluents = action.effects[0].resulting_fluents
    assert F(f"lock-explore {frontier}") in start_fluents
    assert F("not free robot1") in start_fluents

    end_effect = action.effects[1]
    assert F(f"explored {frontier}") in end_effect.resulting_fluents
    assert F("free robot1") in end_effect.resulting_fluents
    probs = [p for p, _ in end_effect.prob_effects]
    assert sum(probs) == 1.0


def test_exploration_complete_when_fully_observed() -> None:
    """A sensor covering the whole grid leaves no frontiers at init."""
    grid = COLLISION_VAL * np.ones((12, 12))
    grid[4:7, 4:9] = FREE_VAL
    env = _make_env(grid, sensor_range=30.0)

    assert env.objects_by_type.get("frontier", set()) == set()
    assert F("exploration-complete").evaluate(env.state.fluents)

"""Integration tests for VisualUnknownSpaceEnvironment (requires OpenGL)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from railroad._bindings import Fluent, State
from railroad.environment.railsim import (
    GuidedMazeConfig,
    RailsimScene,
    VisualUnknownSpaceEnvironment,
)
from railroad.environment.skill import NavigationMoveSkill
from railroad.environment.symbolic import LocationRegistry
from railroad.experimental.unknown_search import NavigationConfig, Pose
from railroad.experimental.unknown_search.operators import (
    construct_move_navigable_operator,
)

F = Fluent

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def maze_scene() -> RailsimScene:
    scene = RailsimScene.maze(
        seed=13, config=GuidedMazeConfig(width_cells=3, height_cells=3)
    )
    start = scene.locations["start_loc"]
    try:
        scene.get_pano_image(Pose(float(start[0]), float(start[1]), 0.0))
    except Exception:
        pytest.skip("no working OpenGL context for railsim rendering")
    yield scene
    scene.release()


def _make_env(scene: RailsimScene, **env_kwargs: Any) -> VisualUnknownSpaceEnvironment:
    start = scene.locations["start_loc"]
    return VisualUnknownSpaceEnvironment(
        scene=scene,
        state=State(0.0, {
            F("at robot1 start_loc"),
            F("free robot1"),
            F("revealed start_loc"),
        }, []),
        objects_by_type={
            "robot": {"robot1"},
            "location": {"start_loc"},
            "frontier": set(),
            "object": set(),
        },
        operators=[construct_move_navigable_operator(5.0)],
        skill_overrides={"move": NavigationMoveSkill},
        robot_initial_poses={"robot1": Pose(float(start[0]), float(start[1]), 0.0)},
        location_registry=LocationRegistry({
            "start_loc": np.array(start, dtype=float)
        }),
        config=NavigationConfig(
            sensor_range=60.0,
            sensor_fov_rad=2 * math.pi,
            interrupt_min_new_cells=30000,
            interrupt_min_dt=30000.0,
        ),
        **env_kwargs,
    )


def test_initial_observation_captures_pano(maze_scene: RailsimScene) -> None:
    env = _make_env(maze_scene)
    assert len(env.pano_records) == 1
    record = env.pano_records[0]
    assert record.robot == "robot1"
    assert record.time == 0.0
    assert record.image.shape == (256, 512, 3)
    assert record.image.dtype == np.uint8


def test_panos_collected_during_move(maze_scene: RailsimScene) -> None:
    env = _make_env(maze_scene)
    moves = [a for a in env.get_actions() if a.name.startswith("move robot1")]
    assert moves, "expected a move action toward a frontier"
    env.act(moves[0])

    assert len(env.pano_records) > 1
    times = [r.time for r in env.pano_records]
    assert times == sorted(times)
    # Poses-in-meters must be the cell poses scaled by resolution.
    res = maze_scene.resolution
    for record in env.pano_records:
        assert record.pose_meters[0] == pytest.approx(record.pose_cells.x * res)
        assert record.pose_meters[1] == pytest.approx(record.pose_cells.y * res)
        assert record.pose_meters[2] == pytest.approx(record.pose_cells.yaw)


def test_capture_panos_disabled(maze_scene: RailsimScene) -> None:
    env = _make_env(maze_scene, capture_panos=False)
    moves = [a for a in env.get_actions() if a.name.startswith("move robot1")]
    assert moves
    env.act(moves[0])
    assert env.pano_records == []

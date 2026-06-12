"""Integration tests for LSPVisualEnvironment (requires OpenGL)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from railroad._bindings import Fluent, State
from railroad.environment.railsim import GuidedMazeConfig, RailsimScene
from railroad.environment.skill import NavigationMoveSkill
from railroad.environment.symbolic import LocationRegistry
from railroad.experimental.unknown_search import NavigationConfig, Pose
from railroad.experimental.unknown_search.operators import (
    construct_move_navigable_operator,
)
from railroad.lsp import (
    OracleFrontierPropertyProvider,
    TrainingDataWriter,
    construct_lsp_explore_operator,
    construct_move_to_goal_operator,
    load_datum,
    read_index,
)
from railroad.lsp.environment import LSPVisualEnvironment

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


def _make_env(
    scene: RailsimScene, data_writer: TrainingDataWriter | None
) -> LSPVisualEnvironment:
    start = scene.locations["start_loc"]
    env_ref: list[LSPVisualEnvironment | None] = [None]
    provider = OracleFrontierPropertyProvider(
        lambda: env_ref[0].oracle_labels if env_ref[0] is not None else {}
    )

    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        if env_ref[0] is None:
            return 5.0
        return env_ref[0].estimate_move_time_safe(robot, loc_from, loc_to)

    def goal_move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        if env_ref[0] is None:
            return 5.0
        return env_ref[0].estimate_goal_move_time(robot, loc_from, loc_to)

    env = LSPVisualEnvironment(
        scene=scene,
        data_writer=data_writer,
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
            "goal": set(),
        },
        operators=[
            construct_move_navigable_operator(move_time_fn),
            construct_move_to_goal_operator(goal_move_time_fn),
            construct_lsp_explore_operator(provider),
        ],
        skill_overrides={"move": NavigationMoveSkill},
        robot_initial_poses={"robot1": Pose(float(start[0]), float(start[1]), 0.0)},
        location_registry=LocationRegistry({
            "start_loc": np.array(start, dtype=float)
        }),
        config=NavigationConfig(
            sensor_range=10.0,
            sensor_fov_rad=2 * math.pi,
            interrupt_min_new_cells=30000,
            interrupt_min_dt=30000.0,
        ),
    )
    env_ref[0] = env
    return env


def test_pano_records_carry_visibility_polygons(maze_scene: RailsimScene) -> None:
    env = _make_env(maze_scene, data_writer=None)
    assert env.pano_records
    for record in env.pano_records:
        polygon = record.visibility_polygon
        assert polygon is not None
        assert polygon.shape[0] == 2
        np.testing.assert_allclose(
            polygon[:, 0], [record.pose_cells.x, record.pose_cells.y]
        )


def test_training_data_emitted(tmp_path, maze_scene: RailsimScene) -> None:  # noqa: ANN001
    writer = TrainingDataWriter(tmp_path / "data", {"test": True})
    env = _make_env(maze_scene, data_writer=writer)

    assert not env.goal_observed, "test needs an undiscovered goal"
    assert env.num_data_written >= 1

    index = read_index(tmp_path / "data")
    assert len(index) == env.num_data_written

    labels = env.oracle_labels
    entry = index[0]
    datum = load_datum(tmp_path / "data" / entry["file"])
    assert datum.image.dtype == np.uint8
    assert datum.image.ndim == 3 and datum.image.shape[2] == 3
    assert datum.image.shape == env.pano_records[0].image.shape

    label = labels[entry["frontier_id"]]
    assert datum.label == (label.prob_feasible >= 0.5)
    writer.close()

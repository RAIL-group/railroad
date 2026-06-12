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
from railroad.lsp import (
    FrontierStatistics,
    FrontierStatisticsEstimator,
    LearnedFrontierStatistics,
    OracleFrontierStatistics,
    TrainingDataWriter,
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
    scene: RailsimScene,
    data_writer: TrainingDataWriter | None,
    frontier_statistics: FrontierStatisticsEstimator | None = None,
) -> LSPVisualEnvironment:
    start = scene.locations["start_loc"]
    return LSPVisualEnvironment(
        scene=scene,
        frontier_statistics=frontier_statistics or OracleFrontierStatistics(),
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
        },
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


def test_learned_statistics_drive_explore_actions(
    maze_scene: RailsimScene,
) -> None:
    """A model plugged in as the estimator parameterizes lsp-explore."""
    batch_sizes: list[int] = []

    def model(observations):  # noqa: ANN001, ANN202
        batch_sizes.append(len(observations))
        for obs in observations:
            assert obs.image.dtype == np.uint8 and obs.image.ndim == 3
        return [FrontierStatistics(0.9, 2.0, 5.0)] * len(observations)

    env = _make_env(
        maze_scene,
        data_writer=None,
        frontier_statistics=LearnedFrontierStatistics(model),
    )

    # The model was batch-evaluated on real panorama observations during
    # the initial refresh...
    assert batch_sizes and max(batch_sizes) >= 1
    assert max(batch_sizes) <= len(env.frontiers)

    # ...and its predictions parameterize the grounded explore actions.
    predicted_ids = {
        fid for fid in env.frontiers
        if env.frontier_statistics.get("robot1", fid)
        == FrontierStatistics(0.9, 2.0, 5.0)
    }
    assert predicted_ids, "expected at least one frontier with a vantage"
    explore = next(
        a for a in env.get_actions()
        if a.name.startswith("lsp-explore")
        and a.name.split()[-1] in predicted_ids
    )
    probs = sorted(p for p, _ in explore.effects[1].prob_effects)
    assert probs[1] == pytest.approx(0.9)

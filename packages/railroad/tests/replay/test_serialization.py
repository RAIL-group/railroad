"""Round-trip tests for RolloutLog on-disk serialization."""

from __future__ import annotations

import numpy as np

from railroad.environment.types import Pose
from railroad.replay.serialization import (
    LoadedPanoRecord,
    load_rollout_log,
    save_rollout_log,
)
from railroad.replay.types import RolloutLog, StepRecord, SubgoalRecord


def _grid() -> np.ndarray:
    return np.array(
        [[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=float
    )


def test_round_trip_minimal(tmp_path) -> None:
    log = RolloutLog(
        recorded_grid=_grid(),
        goal_cell=(2, 0),
        robot_starts={"robot1": (1.0, 0.0, 0.5)},
    )
    save_rollout_log(log, tmp_path)
    loaded = load_rollout_log(tmp_path)

    np.testing.assert_array_equal(loaded.recorded_grid, log.recorded_grid)
    assert loaded.goal_cell == (2, 0)
    assert loaded.robot_starts == {"robot1": (1.0, 0.0, 0.5)}
    assert loaded.problem_class == "navigation"
    assert loaded.subgoals == []
    assert loaded.steps == []


def test_round_trip_with_subgoals_and_steps(tmp_path) -> None:
    log = RolloutLog(
        recorded_grid=_grid(),
        goal_cell=(0, 2),
        robot_starts={"robot1": (1.0, 1.0, 0.0), "robot2": (2.0, 2.0, 1.0)},
        env_name="maze",
        seed=7,
        problem_class="navigation",
        actual_total_cost=42.5,
        config={"sensor_range": 60.0, "speed_cells_per_sec": 2.0},
        subgoals=[
            SubgoalRecord(
                signature="abc",
                centroid=(1, 1),
                cells=np.array([[1, 1, 2], [0, 1, 1]], dtype=int),
                contents=("Knife", "Fork"),
            ),
            SubgoalRecord(
                signature="def",
                centroid=(0, 2),
                cells=np.array([[0], [2]], dtype=int),
            ),
        ],
        steps=[
            StepRecord(
                time=1.0,
                robot_poses={"robot1": (1.0, 1.0, 0.0)},
                chosen_action="move robot1 start_loc frontier_0",
                net_motion={"robot1": 3.0},
            )
        ],
    )
    save_rollout_log(log, tmp_path)
    loaded = load_rollout_log(tmp_path)

    assert loaded.seed == 7
    assert loaded.env_name == "maze"
    assert loaded.actual_total_cost == 42.5
    assert loaded.config == {"sensor_range": 60.0, "speed_cells_per_sec": 2.0}
    assert loaded.robot_starts["robot2"] == (2.0, 2.0, 1.0)

    assert len(loaded.subgoals) == 2
    assert loaded.subgoals[0].signature == "abc"
    assert loaded.subgoals[0].centroid == (1, 1)
    assert loaded.subgoals[0].contents == ("Knife", "Fork")
    np.testing.assert_array_equal(
        loaded.subgoals[0].cells, np.array([[1, 1, 2], [0, 1, 1]])
    )
    np.testing.assert_array_equal(loaded.subgoals[1].cells, np.array([[0], [2]]))

    assert len(loaded.steps) == 1
    assert loaded.steps[0].chosen_action == "move robot1 start_loc frontier_0"
    assert loaded.steps[0].net_motion == {"robot1": 3.0}


def test_config_round_trip_preserves_field_types(tmp_path) -> None:
    """The recorded config must survive disk as int/bool/float, not coerced to
    float — otherwise NavigationConfig fields (e.g. sensor_num_rays, the bool
    toggles) reconstruct with the wrong type on replay."""
    import dataclasses

    from railroad.experimental.unknown_search import NavigationConfig

    log = RolloutLog(
        recorded_grid=_grid(),
        goal_cell=(0, 2),
        robot_starts={"robot1": (1.0, 1.0, 0.0)},
        config=dataclasses.asdict(
            NavigationConfig(
                sensor_range=23.0,
                sensor_num_rays=91,
                move_execution_use_theta_star=False,
            )
        ),
    )
    save_rollout_log(log, tmp_path)
    loaded = load_rollout_log(tmp_path)

    assert loaded.config["sensor_range"] == 23.0
    assert loaded.config["sensor_num_rays"] == 91
    assert isinstance(loaded.config["sensor_num_rays"], int)
    assert loaded.config["move_execution_use_theta_star"] is False
    # And the dict reconstructs a NavigationConfig identical to the original.
    assert NavigationConfig(**loaded.config) == NavigationConfig(
        sensor_range=23.0,
        sensor_num_rays=91,
        move_execution_use_theta_star=False,
    )


def test_pano_records_round_trip(tmp_path) -> None:
    rec = LoadedPanoRecord(
        robot="robot1",
        time=2.0,
        pose_cells=Pose(3.0, 4.0, 0.5),
        pose_meters=(0.3, 0.4, 0.5),
        image=np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3),
        visibility_polygon=np.array([[0.0, 0.0, 5.0, 5.0, 0.0], [0.0, 5.0, 5.0, 0.0, 0.0]]),
    )
    rec_no_poly = LoadedPanoRecord(
        robot="robot1",
        time=3.0,
        pose_cells=Pose(1.0, 1.0, 0.0),
        pose_meters=(0.1, 0.1, 0.0),
        image=np.zeros((4, 8, 3), dtype=np.uint8),
        visibility_polygon=None,
    )
    log = RolloutLog(
        recorded_grid=_grid(),
        goal_cell=(0, 0),
        robot_starts={"robot1": (0.0, 0.0, 0.0)},
        pano_records=[rec, rec_no_poly],
    )
    save_rollout_log(log, tmp_path)
    loaded = load_rollout_log(tmp_path)

    assert len(loaded.pano_records) == 2
    p0 = loaded.pano_records[0]
    np.testing.assert_array_equal(p0.image, rec.image)
    assert (p0.pose_cells.x, p0.pose_cells.y, p0.pose_cells.yaw) == (3.0, 4.0, 0.5)
    assert p0.pose_meters == (0.3, 0.4, 0.5)
    np.testing.assert_array_equal(p0.visibility_polygon, rec.visibility_polygon)
    assert loaded.pano_records[1].visibility_polygon is None


def test_empty_pano_records_no_file(tmp_path) -> None:
    log = RolloutLog(
        recorded_grid=_grid(), goal_cell=(0, 0), robot_starts={"robot1": (0.0, 0.0, 0.0)}
    )
    save_rollout_log(log, tmp_path)
    assert not (tmp_path / "panos.npz").exists()
    assert load_rollout_log(tmp_path).pano_records == []


def test_directory_is_created(tmp_path) -> None:
    target = tmp_path / "nested" / "log"
    log = RolloutLog(
        recorded_grid=_grid(),
        goal_cell=(0, 0),
        robot_starts={"robot1": (0.0, 0.0, 0.0)},
    )
    save_rollout_log(log, target)
    assert (target / "meta.json").exists()
    assert (target / "grid.npz").exists()
    assert load_rollout_log(target).goal_cell == (0, 0)

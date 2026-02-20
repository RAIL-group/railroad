"""Test 1: Mapping growth — repeated scans monotonically increase observed fraction."""

import numpy as np

from railroad.navigation.constants import (
    COLLISION_VAL,
    FREE_VAL,
    UNOBSERVED_VAL,
)
from railroad.environment.unknown_space.laser import (
    get_laser_scanner_directions,
    simulate_sensor_measurement,
)
from railroad.environment.unknown_space.mapping import insert_scan
from railroad.environment.unknown_space.types import Pose


def _make_corridor_grid(size: int = 30) -> np.ndarray:
    """Create a simple corridor grid: walls on boundary, free interior."""
    grid = FREE_VAL * np.ones((size, size))
    grid[0, :] = COLLISION_VAL
    grid[-1, :] = COLLISION_VAL
    grid[:, 0] = COLLISION_VAL
    grid[:, -1] = COLLISION_VAL
    # Internal wall with a gap
    grid[size // 2, :] = COLLISION_VAL
    grid[size // 2, size // 2] = FREE_VAL
    grid[size // 2, size // 2 + 1] = FREE_VAL
    return grid


def test_mapping_growth_monotonic():
    """Repeated insert_scan calls monotonically increase observed fraction."""
    true_grid = _make_corridor_grid(30)
    observed_grid = UNOBSERVED_VAL * np.ones_like(true_grid)

    directions = get_laser_scanner_directions(181, 2 * np.pi)
    max_range = 9.0

    poses = [
        Pose(5.0, 5.0, 0.0),
        Pose(10.0, 10.0, 0.0),
        Pose(5.0, 15.0, 1.57),
        Pose(20.0, 15.0, 3.14),
    ]

    prev_observed_count = 0
    for pose in poses:
        laser_ranges = simulate_sensor_measurement(
            true_grid, directions, max_range, pose
        )
        observed_grid, newly_observed = insert_scan(
            occupancy_grid=observed_grid,
            laser_scanner_directions=directions,
            laser_ranges=laser_ranges,
            max_range=max_range,
            sensor_pose=pose,
            connect_neighbor_distance=2,
        )
        current_observed = int(np.count_nonzero(observed_grid != UNOBSERVED_VAL))
        assert current_observed >= prev_observed_count, (
            f"Observed count decreased: {current_observed} < {prev_observed_count}"
        )
        prev_observed_count = current_observed

    # Should have observed some cells
    assert prev_observed_count > 0


def test_observed_never_reverts_to_unobserved():
    """Once a cell is observed, it never goes back to UNOBSERVED."""
    true_grid = _make_corridor_grid(20)
    observed_grid = UNOBSERVED_VAL * np.ones_like(true_grid)

    directions = get_laser_scanner_directions(91, 2 * np.pi)
    max_range = 7.0

    poses = [
        Pose(5.0, 5.0, 0.0),
        Pose(10.0, 10.0, 1.0),
        Pose(15.0, 5.0, -1.0),
    ]

    for pose in poses:
        prev_observed = observed_grid.copy()
        laser_ranges = simulate_sensor_measurement(
            true_grid, directions, max_range, pose
        )
        observed_grid, _ = insert_scan(
            occupancy_grid=observed_grid,
            laser_scanner_directions=directions,
            laser_ranges=laser_ranges,
            max_range=max_range,
            sensor_pose=pose,
        )
        # Any cell that was previously observed should still be observed
        was_observed = prev_observed != UNOBSERVED_VAL
        still_observed = observed_grid[was_observed] != UNOBSERVED_VAL
        assert still_observed.all(), "Some previously observed cells reverted to UNOBSERVED"

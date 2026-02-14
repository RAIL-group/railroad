"""Occupancy grid mapping utilities for local scan fusion."""

from __future__ import annotations

import math

from matplotlib.path import Path
import numpy as np
import scipy.ndimage

from .constants import (
    COLLISION_VAL,
    FREE_VAL,
    OBSTACLE_THRESHOLD,
    UNOBSERVED_VAL,
)
from . import laser
from .types import PoseLike


def _transform_rays(rays: np.ndarray, sensor_pose: PoseLike) -> np.ndarray:
    """Transform (rotate and offset) a laser scan according to pose."""
    origin = np.array([[float(sensor_pose.x)], [float(sensor_pose.y)]])
    rotation_mat = np.array(
        [
            [math.cos(float(sensor_pose.yaw)), -math.sin(float(sensor_pose.yaw))],
            [math.sin(float(sensor_pose.yaw)), math.cos(float(sensor_pose.yaw))],
        ]
    )

    return np.matmul(rotation_mat, rays) + origin


def _get_poly_for_scan(transformed_rays: np.ndarray, sensor_pose: PoseLike) -> Path:
    """Return a polygon path for transformed rays."""
    origin_np = np.array([[float(sensor_pose.x)], [float(sensor_pose.y)]])
    path_points = np.concatenate((origin_np, transformed_rays, origin_np), axis=1)
    return Path(path_points.T, closed=True)


def _set_points_inside_poly(
    grid: np.ndarray,
    poly: Path,
    value: float,
    max_range: float | None = None,
    sensor_pose: PoseLike | None = None,
) -> np.ndarray:
    """Set grid cells inside a polygon to a value."""
    vertices = np.asarray(poly.vertices, dtype=float)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    bounds_min = np.floor(bounds_min)
    bounds_max = np.ceil(bounds_max)
    bounds_min[0] = max(bounds_min[0], 0)
    bounds_min[1] = max(bounds_min[1], 0)
    bounds_max[0] = min(bounds_max[0], grid.shape[0] - 1)
    bounds_max[1] = min(bounds_max[1], grid.shape[1] - 1)

    x = np.arange(bounds_min[0], bounds_max[0] + 1) + 0.5
    y = np.arange(bounds_min[1], bounds_max[1] + 1) + 0.5
    xv, yv = np.meshgrid(x, y)
    xr = np.reshape(xv, (xv.size, 1))
    yr = np.reshape(yv, (yv.size, 1))
    grid_points = np.concatenate((xr, yr), axis=1)

    if max_range is not None and sensor_pose is not None:
        origin_np = np.array([[float(sensor_pose.x)], [float(sensor_pose.y)]])
        is_within_range = np.sum((grid_points - origin_np.T) ** 2, axis=1) < (max_range**2)
        grid_points = grid_points[is_within_range, :]

    inside_points = poly.contains_points(grid_points)

    out = grid.copy()
    out[
        grid_points[inside_points, 0].astype(int),
        grid_points[inside_points, 1].astype(int),
    ] = value
    return out


def _set_collision_boundary(
    grid: np.ndarray,
    transformed_rays: np.ndarray,
    is_within_max_range: np.ndarray,
    connect_neighbor_distance: int | None = None,
) -> np.ndarray:
    """Insert collision observations at transformed ray endpoints."""
    coll_points = transformed_rays.astype(int)
    coll_points = coll_points[:, is_within_max_range]
    coll_points = coll_points[:, coll_points[0, :] >= 0]
    coll_points = coll_points[:, coll_points[1, :] >= 0]
    coll_points = coll_points[:, coll_points[0, :] < grid.shape[0]]
    coll_points = coll_points[:, coll_points[1, :] < grid.shape[1]]

    out = grid.copy()
    if connect_neighbor_distance is None:
        out[coll_points[0], coll_points[1]] = COLLISION_VAL
        return out

    for ii in range(max(0, coll_points.shape[1] - 1)):
        start = coll_points[:, ii]
        end = coll_points[:, ii + 1]
        if np.linalg.norm(start - end) < connect_neighbor_distance:
            bpoints = laser.bresenham_points(start, end)
            out[bpoints[0, :], bpoints[1, :]] = COLLISION_VAL
        else:
            out[start[0], start[1]] = COLLISION_VAL

    return out


def _update_grid_with_projected_measurement(
    grid: np.ndarray,
    measurement_grid: np.ndarray,
    occupied_prob: float,
    unoccupied_prob: float,
) -> np.ndarray:
    """Fuse prior occupancy grid with a projected scan measurement grid."""
    free_spaces = measurement_grid == FREE_VAL
    coll_spaces = measurement_grid == COLLISION_VAL
    unobserved_spaces = grid == UNOBSERVED_VAL

    out = grid.copy()

    out[unobserved_spaces] = measurement_grid[unobserved_spaces]

    out[free_spaces] = (
        unoccupied_prob * FREE_VAL + (1 - unoccupied_prob) * out[free_spaces]
    )
    out[coll_spaces] = (
        occupied_prob * COLLISION_VAL + (1 - occupied_prob) * out[coll_spaces]
    )

    return out


def insert_scan(
    occupancy_grid: np.ndarray,
    laser_scanner_directions: np.ndarray,
    laser_ranges: np.ndarray,
    max_range: float,
    sensor_pose: PoseLike,
    connect_neighbor_distance: int | None = None,
    occupied_prob: float = 0.9,
    unoccupied_prob: float = 0.1,
    do_only_compute_visibility: bool = False,
) -> tuple[np.ndarray, int]:
    """Insert a simulated scan into an occupancy grid.

    Returns:
        tuple of (updated_grid, newly_observed_cells_count)
    """
    truncated_ranges = laser_ranges.copy()
    truncated_ranges[truncated_ranges > max_range] = max_range

    rays = truncated_ranges * laser_scanner_directions
    transformed_rays = _transform_rays(rays, sensor_pose)

    measurement_grid = UNOBSERVED_VAL * np.ones(occupancy_grid.shape)
    poly = _get_poly_for_scan(transformed_rays, sensor_pose)
    measurement_grid = _set_points_inside_poly(
        measurement_grid,
        poly,
        value=FREE_VAL,
        max_range=max_range,
        sensor_pose=sensor_pose,
    )
    measurement_grid = _set_collision_boundary(
        grid=measurement_grid,
        transformed_rays=transformed_rays,
        is_within_max_range=(laser_ranges < max_range),
        connect_neighbor_distance=connect_neighbor_distance,
    )

    newly_observed = np.logical_and(
        occupancy_grid == UNOBSERVED_VAL,
        measurement_grid != UNOBSERVED_VAL,
    )
    newly_observed_count = int(np.count_nonzero(newly_observed))

    if do_only_compute_visibility:
        return measurement_grid, newly_observed_count

    updated = _update_grid_with_projected_measurement(
        grid=occupancy_grid,
        measurement_grid=measurement_grid,
        occupied_prob=occupied_prob,
        unoccupied_prob=unoccupied_prob,
    )

    return updated, newly_observed_count


def get_fully_connected_observed_grid(occupancy_grid: np.ndarray, pose: PoseLike) -> np.ndarray:
    """Prune observed regions not connected to the robot's observed component."""
    observed = np.logical_and(
        occupancy_grid < OBSTACLE_THRESHOLD,
        occupancy_grid >= FREE_VAL,
    )
    labels, _ = scipy.ndimage.label(observed)

    out = occupancy_grid.copy()
    robot_row = int(round(float(pose.x)))
    robot_col = int(round(float(pose.y)))
    if not (0 <= robot_row < out.shape[0] and 0 <= robot_col < out.shape[1]):
        return out

    robot_label = labels[robot_row, robot_col]
    mask = np.logical_and(labels > 0, labels != robot_label)
    out[mask] = UNOBSERVED_VAL
    return out

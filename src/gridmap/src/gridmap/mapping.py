"""A set of functions useful for manipulating occupancy grids."""

import math
import matplotlib
import numpy as np
import scipy

from . import laser

from .constants import (COLLISION_VAL, FREE_VAL, UNOBSERVED_VAL,
                        OBSTACLE_THRESHOLD)


def _transform_rays(rays, sensor_pose):
    """Transform (rotate and offset) a laser scan according to pose."""
    origin = np.array([[sensor_pose.x], [sensor_pose.y]])
    rotation_mat = np.array(
        [[math.cos(sensor_pose.yaw), -math.sin(sensor_pose.yaw)],
         [math.sin(sensor_pose.yaw),
          math.cos(sensor_pose.yaw)]])

    return np.matmul(rotation_mat, rays) + origin


def _get_poly_for_scan(transformed_rays, sensor_pose):
    """Returns a polygon obj for a transformed scan.

    For the polygon to completely contain the region defined by the scan,
    this function requires that the sensor_pose be used to close the path.
    """
    origin_np = np.array([[sensor_pose.x], [sensor_pose.y]])
    path_points = np.concatenate((origin_np, transformed_rays, origin_np),
                                 axis=1)
    return matplotlib.path.Path(path_points.T, closed=True)


def _set_points_inside_poly(grid,
                            poly,
                            value,
                            max_range=None,
                            sensor_pose=None):
    """Sets the value of any grid points defined by a polygon.

    This function is suitable for setting all free points defined by a
    laser scan converted into a matplotlib.path.Path via the
    _get_poly_for_scan function.
    """

    # Compute the bounds
    bounds_min = poly.vertices.min(axis=0)
    bounds_max = poly.vertices.max(axis=0)
    bounds_min = np.floor(bounds_min)
    bounds_max = np.ceil(bounds_max)
    bounds_min[0] = max(bounds_min[0], 0)
    bounds_min[1] = max(bounds_min[1], 0)
    bounds_max[0] = min(bounds_max[0], grid.shape[0] - 1)
    bounds_max[1] = min(bounds_max[1], grid.shape[1] - 1)

    # Get list of points
    x = np.arange(bounds_min[0], bounds_max[0] + 1) + 0.5
    y = np.arange(bounds_min[1], bounds_max[1] + 1) + 0.5
    xv, yv = np.meshgrid(x, y)
    xr = np.reshape(xv, (xv.size, 1))
    yr = np.reshape(yv, (yv.size, 1))
    grid_points = np.concatenate((xr, yr), axis=1)

    # Prune points outside the max range if necessary
    if max_range is not None:
        origin_np = np.array([[sensor_pose.x], [sensor_pose.y]])
        is_within_range = np.sum(
            (grid_points - origin_np.T)**2, axis=1) < (max_range**2)
        grid_points = grid_points[is_within_range, :]

    # Compute inside points and set grid value
    inside_points = poly.contains_points(grid_points)

    grid = grid.copy()
    grid[grid_points[inside_points, 0].astype(int),
         grid_points[inside_points, 1].astype(int)] = value

    return grid


def _set_collision_boundary(grid,
                            transformed_rays,
                            is_within_max_range,
                            connect_neighbor_distance=None):
    """Adds obstacles defined by the transformed_rays.

    Obstacles are inserted at the endpoints of each of the transformed rays
    for which is_within_max_range is True. If connect_neighbor_distance is not
    None, lines of 'occupied' are drawn between neighboring points of a fixed
    distance (in units of grid cells). Rays must already be transformed.
    """

    # Filter out points that are above max range or outside the grid
    coll_points = transformed_rays.astype(int)
    coll_points = coll_points[:, is_within_max_range]
    coll_points = coll_points[:, coll_points[0, :] >= 0]
    coll_points = coll_points[:, coll_points[1, :] >= 0]
    coll_points = coll_points[:, coll_points[0, :] < grid.shape[0]]
    coll_points = coll_points[:, coll_points[1, :] < grid.shape[1]]

    if connect_neighbor_distance is None:
        grid[coll_points[0], coll_points[1]] = COLLISION_VAL
    else:
        for ii in range(coll_points.shape[1] - 1):
            start = coll_points[:, ii]
            end = coll_points[:, ii + 1]
            if np.linalg.norm(start - end) < connect_neighbor_distance:
                bpoints = laser.bresenham_points(start, end)
                grid[bpoints[0, :], bpoints[1, :]] = COLLISION_VAL
            else:
                grid[start[0], start[1]] = COLLISION_VAL

    return grid


def _update_grid_with_projected_measurement(grid, measurement_grid,
                                            occupied_prob, unoccupied_prob):
    """Fuses an old grid and a new 'measurement_grid'.

    The measurement_grid should correspond to a new laser scan projected
    into 2D space. The new measurement is fused into the old according to the
    likelihood of observing occupied or unoccupied space.
    """
    free_spaces = (measurement_grid == FREE_VAL)
    coll_spaces = (measurement_grid == COLLISION_VAL)
    unobserved_spaces = (grid == UNOBSERVED_VAL)

    grid = grid.copy()
    # If the occupancy grid was unobserved, add the new value (no
    # "probablity")
    grid[unobserved_spaces] = measurement_grid[unobserved_spaces]

    # Otherwise, a probabilistic approach is used to join them
    grid[free_spaces] = (unoccupied_prob * FREE_VAL +
                         (1 - unoccupied_prob) * grid[free_spaces])
    grid[coll_spaces] = (occupied_prob * COLLISION_VAL +
                         (1 - occupied_prob) * grid[coll_spaces])

    return grid


def insert_scan(occupancy_grid,
                laser_scanner_directions,
                laser_ranges,
                max_range,
                sensor_pose,
                connect_neighbor_distance=None,
                occupied_prob=0.9,
                unoccupied_prob=0.1,
                do_only_compute_visibility=False):
    """Add a new scan to an occupancy grid.

    We follow the Octomap Server convention for inserting scans: if the
    grid is unoccupied, the value from the laser scan is trusted as the
    ground truth. However, if another sensor observation has already
    been revealed in that region, the new points are weighted by their
    respective probabilities and averaged into the new grid. See
    "_update_grid_with_projected_measurement" for implementation.

    Args:
        occupancy_grid (np.Array): 2D grid of currently observed map.
        laser_scanner_directions (np.Array): list of unit vectors
            indicating direction of each element of 'range'.
        laser_ranges (np.Array): 1D list of laser ranges/distances.
        max_range (float): maximum range of sensor. Ranges above this distance
            will not be inserted as obstacles and the insertion of free space
            is limited to this value.
        sensor_pose (Pose): pose at which the measurement will be inserted.
        connect_neighbor_distance (int): Distance (in grid cells) that
            neighboring obstacles should be connected. Important when the
            max_range is large compared to the number of ranges.
        occupied_prob (float): likelihood an observed obstacle is an obstacle.
        unoccupied_prob (float): likelihood a free cell is actually free.

    Returns:
        occupancy_grid (np.Array): grid with new scan inserted.
    """

    # Truncate the points by the max range (for efficiency purposes)
    truncated_ranges = laser_ranges.copy()
    truncated_ranges[truncated_ranges > max_range] = max_range

    # Compute the rays
    rays = truncated_ranges * laser_scanner_directions
    transformed_rays = _transform_rays(rays, sensor_pose)

    # Populate the new grid with the free/collision points
    new_measurement_grid = UNOBSERVED_VAL * np.ones(occupancy_grid.shape)
    poly = _get_poly_for_scan(transformed_rays, sensor_pose)
    new_measurement_grid = _set_points_inside_poly(new_measurement_grid,
                                                   poly,
                                                   value=FREE_VAL,
                                                   max_range=max_range,
                                                   sensor_pose=sensor_pose)
    new_measurement_grid = _set_collision_boundary(
        grid=new_measurement_grid,
        transformed_rays=transformed_rays,
        is_within_max_range=(laser_ranges < max_range),
        connect_neighbor_distance=connect_neighbor_distance)

    if do_only_compute_visibility:
        return new_measurement_grid

    # Update grid with new measurement
    occupancy_grid = _update_grid_with_projected_measurement(
        grid=occupancy_grid.copy(),
        measurement_grid=new_measurement_grid,
        occupied_prob=occupied_prob,
        unoccupied_prob=unoccupied_prob)

    return occupancy_grid


def get_fully_connected_observed_grid(occupancy_grid, pose):
    """Returns a version of the observed grid in which components
    unconnected to the region containing the robot are set to 'unobserved'.
    This useful for preventing the system from generating any frontiers
    that cannot be planned to. Also, certain geometrys may cause frontiers
    to be erroneously ruled out if "stray" observed space exists.
    """
    # Group the frontier points into connected components
    labels, _ = scipy.ndimage.label(
        np.logical_and(occupancy_grid < OBSTACLE_THRESHOLD,
                       occupancy_grid >= FREE_VAL))

    occupancy_grid = occupancy_grid.copy()
    robot_label = labels[int(pose.x), int(pose.y)]
    mask = np.logical_and(labels > 0, labels != robot_label)
    occupancy_grid[mask] = UNOBSERVED_VAL

    return occupancy_grid

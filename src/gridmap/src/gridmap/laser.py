"""
Functions for simulating a laser sensor and ray casting in a grid.
"""
import numpy as np
import math


def get_laser_scanner_directions(num_points, field_of_view_rad):
    """Creates an array of direction vectors in the sensor frame."""
    angles_rad = np.linspace(-field_of_view_rad / 2, field_of_view_rad / 2,
                             num_points)
    directions = np.vstack((np.cos(angles_rad), np.sin(angles_rad)))
    return directions


def simulate_sensor_measurement(occupancy_grid,
                                laser_scanner_directions,
                                max_range,
                                sensor_pose,
                                subsample=1):
    """Ray cast to get a simulated laser measurement from a grid.

    Args:
        occupancy_grid (np.Array): the ground-truth map.
        laser_scanner_directions (np.Array): list of direction unit vectors
            that define the laser (see 'get_laser_scanner_directions').
        max_range (float): maximum range of ray casting.
        sensor_pose (Pose): pose (x, y, yaw) of the sensor.
        subsample (int) [DEPRECATED]: scale factor for subsampling grid during
            ray casting; should give higher-precision.
    """

    # Apply the pose transformation to the laser scanner rays
    origin = [
        sensor_pose.x - 0.5 * (subsample - 1),
        sensor_pose.y - 0.5 * (subsample - 1)
    ]
    rotation_mat = np.array(
        [[math.cos(sensor_pose.yaw), -math.sin(sensor_pose.yaw)],
         [math.sin(sensor_pose.yaw),
          math.cos(sensor_pose.yaw)]])
    rotated_directions = np.matmul(rotation_mat, laser_scanner_directions)
    transformed_rays = max_range * rotated_directions
    transformed_rays[0, :] += origin[0]
    transformed_rays[1, :] += origin[1]

    # Loop through the rays and cast them into the grid
    num_scans = laser_scanner_directions.shape[1]
    outputs = np.zeros((2, num_scans))
    for [target, out] in np.nditer([transformed_rays, outputs],
                                   order='F',
                                   flags=['external_loop'],
                                   op_flags=[['readonly'], ['readwrite']],
                                   op_dtypes=['float64', 'float64']):
        _, end = cast_ray(occupancy_grid, origin, target, scale=4)
        out[0] = end[0] - origin[0]
        out[1] = end[1] - origin[1]

    # Project the points onto rotated_directions (and add minor correction
    # factor)
    val = np.zeros(num_scans)
    for ii in range(num_scans):
        val[ii] = (outputs[0, ii] * rotated_directions[0, ii] +
                   outputs[1, ii] * rotated_directions[1, ii])
        outputs[0,
                ii] = (val[ii] + 0.5 * subsample) * rotated_directions[0, ii]
        outputs[1,
                ii] = (val[ii] + 0.5 * subsample) * rotated_directions[1, ii]

    return np.linalg.norm(outputs, axis=0)


def bresenham_points(start, target, scale=4, do_floor=True):
    """Get the points along a line using Bresenham's Algorithm.

    Using Bresenham's algorithm, this function returns a list of points foom the
    starting location to the target location. An optional scale argument can be
    passed to compute points at a fractional grid resolution. For example, if
    requesting the points between start=[0.0, 0.9] and target=[5.0, 1.9], we
    might want the line to "jump" earlier. Using scale=2 allows us to accomplish
    this:

    >>> bresenham_points(start=[0.0, 0.9], target=[5.0, 1.9], \
                         scale=1, do_floor=True)
    array([[0, 1, 2, 3, 4, 5],
           [0, 0, 0, 1, 1, 1]])
    >>> bresenham_points(start=[0.0, 0.9], target=[5.0, 1.9], \
                         scale=2, do_floor=True)
    array([[0, 1, 2, 3, 4, 5],
           [0, 1, 1, 1, 1, 1]])

    If the values are not "floored", all the sub-pixel point are returned:
    >>> bresenham_points(start=[0.0, 0.9], target=[5.0, 1.9], \
                         scale=2, do_floor=False)
    array([[ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ],
           [ 0.5,  0.5,  0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  1.5,  1.5,  1.5]])

    Args:
        start (2x float): 2D starting point
        target (2x float): 2D ending point
        scale (int): Optional sub-pixel scale argument
        do_floor (Bool): Optional argument to return integer coordinates
            or fractional coordinages (for scale != 1)

    Returns: 2xN numpy array of N coordinates corresponding to points given by
        Bresenham's algorithm. (This includes the endpoints start and target.)
    """

    # Convert to integers
    upscaled_start_int = [int(scale * start[0]), int(scale * start[1])]
    upscaled_target_int = [int(scale * target[0]), int(scale * target[1])]

    upscaled_point = upscaled_start_int

    dx = upscaled_target_int[0] - upscaled_start_int[0]
    xstep = 1
    if dx < 0:
        dx = -dx
        xstep = -1

    dy = upscaled_target_int[1] - upscaled_start_int[1]
    ystep = 1
    if dy < 0:
        dy = -dy
        ystep = -1

    if dx == 0:
        # Vertical
        upsampled_points = np.zeros([2, dy + 1])
        for ii in range(dy + 1):
            upsampled_points[0, ii] = upscaled_point[0]
            upsampled_points[1, ii] = upscaled_point[1]
            upscaled_point[1] += ystep
    elif dy == 0:
        # Horizontal
        upsampled_points = np.zeros([2, dx + 1])
        for ii in range(dx + 1):
            upsampled_points[0, ii] = upscaled_point[0]
            upsampled_points[1, ii] = upscaled_point[1]
            upscaled_point[0] += xstep
    elif dx > dy:
        n = dx
        dy += dy
        e = dy - dx
        dx += dx

        upsampled_points = np.zeros([2, n + 1])
        for ii in range(n + 1):
            upsampled_points[0, ii] = upscaled_point[0]
            upsampled_points[1, ii] = upscaled_point[1]
            if e >= 0:
                upscaled_point[1] += ystep
                e -= dx
            e += dy
            upscaled_point[0] += xstep
    else:
        n = dy
        dx += dx
        e = dx - dy
        dy += dy

        upsampled_points = np.zeros([2, n + 1])
        for ii in range(n + 1):
            upsampled_points[0, ii] = upscaled_point[0]
            upsampled_points[1, ii] = upscaled_point[1]
            if e >= 0:
                upscaled_point[0] += xstep
                e -= dy
            e += dx
            upscaled_point[1] += ystep

    # Now return the collision state and the current pose
    points = 1.0 * upsampled_points / scale

    if do_floor is True:
        points = points.astype(int)
        indices = np.unique(points, axis=1, return_index=True)[1]
        points = np.array([points[:, ind] for ind in sorted(indices)]).T

    return points


def cast_ray(occupancy_grid, start, target, scale=4, obstacle_threshold=1):
    """Cast a ray through an occupancy grid, returning at collision. This
    function uses Bresenham's line algorithm to determine if a ray passing
    through a grid collides with obstacles in the grid.
    """

    # Convert to integers
    upscaled_start_int = [int(scale * start[0]), int(scale * start[1])]
    upscaled_target_int = [int(scale * target[0]), int(scale * target[1])]
    upscaled_point = upscaled_start_int
    did_collide = False

    dx = upscaled_target_int[0] - upscaled_start_int[0]
    xstep = 1
    if dx < 0:
        dx = -dx
        xstep = -1

    dy = upscaled_target_int[1] - upscaled_start_int[1]
    ystep = 1
    if dy < 0:
        dy = -dy
        ystep = -1

    if dx == 0:
        # Vertical
        for ii in range(dy):
            # Check for collision
            # If outside the grid, the ray will not collide
            point = [upscaled_point[0] // scale, upscaled_point[1] // scale]
            if not (0 <= point[0] < occupancy_grid.shape[0]
                    and 0 <= point[1] < occupancy_grid.shape[1]):
                return (False, upscaled_target_int)
            elif occupancy_grid[point[0], point[1]] >= obstacle_threshold:
                did_collide = True
                break

            # Update current position
            upscaled_point[1] += ystep
    elif dy == 0:
        # Horizontal
        for ii in range(dx):
            # Check for collision
            # If outside the grid, the ray will not collide
            point = [upscaled_point[0] // scale, upscaled_point[1] // scale]
            if not (0 <= point[0] < occupancy_grid.shape[0]
                    and 0 <= point[1] < occupancy_grid.shape[1]):
                return (False, upscaled_target_int)
            elif occupancy_grid[point[0], point[1]] >= obstacle_threshold:
                did_collide = True
                break

            # Update current position
            upscaled_point[0] += xstep
    elif dx > dy:
        n = dx
        dy += dy
        e = dy - dx
        dx += dx

        for ii in range(n):
            # Check for collision
            # If outside the grid, the ray will not collide
            point = [upscaled_point[0] // scale, upscaled_point[1] // scale]
            if not (0 <= point[0] < occupancy_grid.shape[0]
                    and 0 <= point[1] < occupancy_grid.shape[1]):
                return (False, upscaled_target_int)
            elif occupancy_grid[point[0], point[1]] >= obstacle_threshold:
                did_collide = True
                break

            # Update current position
            if e >= 0:
                upscaled_point[1] += ystep
                e -= dx
            e += dy
            upscaled_point[0] += xstep
    else:
        n = dy
        dx += dx
        e = dx - dy
        dy += dy

        for ii in range(n):
            # Check for collision
            # If outside the grid, the ray will not collide
            point = [upscaled_point[0] // scale, upscaled_point[1] // scale]
            if not (0 <= point[0] < occupancy_grid.shape[0]
                    and 0 <= point[1] < occupancy_grid.shape[1]):
                return (False, upscaled_target_int)
            elif occupancy_grid[point[0], point[1]] >= obstacle_threshold:
                did_collide = True
                break

            # Update current position
            if e >= 0:
                upscaled_point[0] += xstep
                e -= dy
            e += dx
            upscaled_point[1] += ystep

    # Now return the collision state and the current pose
    point = [1.0 * upscaled_point[0] / scale, 1.0 * upscaled_point[1] / scale]
    return (did_collide, [point[0] + 0.0 / scale, point[1] + 0.0 / scale])

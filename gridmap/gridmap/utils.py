import math
import numpy as np
import scipy.ndimage

from .constants import COLLISION_VAL, OBSTACLE_THRESHOLD


def inflate_grid(grid,
                 inflation_radius,
                 obstacle_threshold=OBSTACLE_THRESHOLD,
                 collision_val=COLLISION_VAL):
    """Inflates obstacles in an occupancy grid

    Creates a mask for all grid cells exceeding the obstacle threshold and
    uses a convolution to compute how much the obstacles should inflate. The
    inflated mask is used to set the cells of a copy of the initial grid to
    occupied. All other cells are unchanged: free space and unnoccupied cells
    outside of the inflation radius are preserved (therefore, frontiers can
    still be computed from an inflated grid).

    Args:
        grid (np.Array): occupancy grid
        inflation_radius (float): radius (in grid units) to inflate obstacles.
            Note that this is a float; a fractional inflation radius can be
            used to determine whether or not corners of a box are included.
        obstacle_threshold (float): value above which a cell is an obstacle.
        collision_val (float): value obstacles are given after inflation.

    Returns:
        inflated_grid (np.Array): grid with inflated obstacles.
    """

    obstacle_grid = np.zeros(grid.shape)
    obstacle_grid[grid >= obstacle_threshold] = 1

    kernel_size = int(1 + 2 * math.ceil(inflation_radius))
    cind = int(math.ceil(inflation_radius))
    y, x = np.ogrid[-cind:kernel_size - cind, -cind:kernel_size - cind]
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[y * y + x * x <= inflation_radius * inflation_radius] = 1
    inflated_mask = scipy.ndimage.convolve(obstacle_grid,
                                           kernel,
                                           mode='constant',
                                           cval=0)
    inflated_mask = inflated_mask >= 1.0
    inflated_grid = grid.copy()
    inflated_grid[inflated_mask] = collision_val

    return inflated_grid

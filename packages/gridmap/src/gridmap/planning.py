import numpy as np
import skimage.graph

from . import laser
from .constants import OBSTACLE_THRESHOLD
from .utils import inflate_grid


def compute_cost_grid_from_position(occupancy_grid,
                                    start,
                                    use_soft_cost=False,
                                    obstacle_cost=-1,
                                    ends=None,
                                    only_return_cost_grid=False):
    """Get the cost grid and planning function for a grid/start position.

    The primary purpose of this function is to plan using Dijkstra's algorithm.
    If use_soft_cost == True, a soft inflation cost is added to the grid before
    planning; This is useful for planning in practice, but increases planning
    time, so defaults to False. The cost grid is computed by thresholding the
    input occupancy_grid using OBSTACLE_THRESHOLD: anything below this value,
    including free space, is treated as unoccupied. In addition to returning
    the cost grid from the start position, this function also provides a
    helper utility for generating a plan to a 'target' position in the grid.

    Args:
        occupancy_grid (np.Array): input grid over which cost is computed
        start (Pose): location of the start position for cost computation
        use_soft_cost (Bool): do use soft costs.

    Returns:
        cost_grid (np.Array): the Dijkstra's algorithm cost of traveling from
            the 'start' position to other positions in the grid. If the point
            is unreachable, the cost_grid is infinite for that grid cell.
        get_plan (function): Returns the plan from 'start' to a provided
            'target' argument (also a Pose). Optional argumnent 'do_sparsify'
            will return a sparse version of the plan, computed via the cost
            grid; the sparsification will only happen a fixed number of points
            into the future, specified by 'bound'. Optional argument 'do_flip'
            will reverse the order of the computed plan and the order of the
            sparsification.
    """
    if len(np.array(start).shape) > 1:
        starts = start.T
    else:
        starts = [start]

    if use_soft_cost is not None and use_soft_cost:
        scale_factor = 50
        input_cost_grid = np.ones(occupancy_grid.shape) * scale_factor
        g1 = inflate_grid(occupancy_grid, 1.5)
        g2 = inflate_grid(g1, 1.0)
        g3 = inflate_grid(g2, 1.5)
        soft_cost_grid = 8 * g1 + 5 * g2 + g3
        input_cost_grid += soft_cost_grid
    else:
        scale_factor = 1
        soft_cost_grid = None
        input_cost_grid = np.ones(occupancy_grid.shape)

    input_cost_grid[occupancy_grid >= OBSTACLE_THRESHOLD] = obstacle_cost

    mcp = skimage.graph.MCP_Geometric(input_cost_grid)
    if ends is None:
        cost_grid = mcp.find_costs(starts=starts)[0] / (1.0 * scale_factor)
    else:
        cost_grid = mcp.find_costs(starts=starts,
                                   ends=ends)[0] / (1.0 * scale_factor)

    if only_return_cost_grid:
        return cost_grid

    def get_path(target, do_sparsify=False, do_flip=False, bound=25):
        try:
            path_list = mcp.traceback(target)
        except ValueError:
            # Planning failed
            return False, np.array([[]])
        path = np.zeros((2, len(path_list)))
        if not do_flip:
            for ii in range(len(path_list)):
                path[:, ii] = path_list[ii]
        else:
            for ii in range(len(path_list)):
                path[:, -1 - ii] = path_list[ii]

        if not do_sparsify:
            return True, path.astype(int)

        # Sparsify the path
        pstart = path[:, 0]
        keep = [0]
        inf_grid = np.isinf(cost_grid)

        if soft_cost_grid is not None:
            tmp_grid = np.logical_or(
                soft_cost_grid >= 0.05 * scale_factor * OBSTACLE_THRESHOLD,
                inf_grid)
        else:
            tmp_grid = inf_grid

        if bound is not None:
            bound = min(path.shape[1] - 1, bound)
        else:
            bound = path.shape[1] - 1

        for ii in range(2, bound):
            # Cast ray to the current terminal point
            did_collide, _ = laser.cast_ray(tmp_grid, pstart, path[:, ii])

            # If the ray tracing collides, add this point to the 'keep' list
            # and update pstart
            if did_collide:
                keep.append(ii)
                pstart = path[:, ii]

        for ii in range(bound, path.shape[1] - 1):
            keep.append(ii)

        keep.append(path.shape[1] - 1)
        path = path[:, keep]

        return True, path.astype(int)

    return cost_grid, get_path

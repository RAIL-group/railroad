import numpy as np
import gridmap
from common import Pose


def get_cost_between_two_coords(grid, start, end, return_path=False):
    occ_grid = np.copy(grid)
    occ_grid[int(start[0])][int(start[1])] = 0

    occ_grid[end[0], end[1]] = 0

    cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
        occ_grid,
        start=[start[0], start[1]],
        use_soft_cost=True)
    cost = cost_grid[end[0], end[1]]

    if return_path:
        _, path = get_path(target=[end[0], end[1]])
        return cost, path
    return cost


def get_coordinates_at_time(path, time):
    diffs = np.diff(path, axis=1)
    segment_lengths = np.linalg.norm(diffs, axis=0)
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
    idx = np.searchsorted(cumulative_lengths, time, side='left')
    idx = idx if idx < len(cumulative_lengths) else -1
    return path[:, idx]


def compute_cost_and_trajectory(grid, path, resolution=0.05, use_robot_model=False):
    '''This function returns the path cost, robot trajectory
    given the occupancy grid and the container poses the
    robot explored during object search.
    '''
    if use_robot_model:
        pass
        # cost, trajectory = compute_cost_and_robot_trajectory(grid, path)
    else:
        cost, trajectory = compute_cost_and_dijkstra_trajectory(grid, path)

    return resolution * cost, trajectory


def compute_cost_and_dijkstra_trajectory(grid, path):
    total_cost = 0
    trajectory = None
    occ_grid = np.copy(grid)

    for pose in path:
        occ_grid[int(pose.x), int(pose.y)] = 0

    for idx, pose in enumerate(path[:-1]):
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            occ_grid,
            start=[pose.x, pose.y],
            use_soft_cost=True,
            only_return_cost_grid=False)
        next_pose = path[idx + 1]

        cost = cost_grid[int(next_pose.x), int(next_pose.y)]

        total_cost += cost
        _, robot_path = get_path([next_pose.x, next_pose.y],
                                 do_sparsify=False,
                                 do_flip=False)
        if trajectory is None:
            trajectory = robot_path
        else:
            trajectory = np.concatenate((trajectory, robot_path), axis=1)

    return total_cost, trajectory

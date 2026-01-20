import numpy as np
import gridmap
import common.robot
import common.primitive as primitive
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
        cost, trajectory = compute_cost_and_robot_trajectory(grid, path)
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


def compute_cost_and_robot_trajectory(grid, path):
    robot = common.robot.Turtlebot_Robot(pose=Pose(path[0].x, path[0].y, yaw=0))

    for _, next_pose in enumerate(path[1:]):
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            grid, start=[next_pose.x, next_pose.y],
            use_soft_cost=True,
            only_return_cost_grid=False
        )

        reached = False
        while not reached:
            _, robot_path = get_path([robot.pose.x, robot.pose.y],
                                     do_sparsify=True,
                                     do_flip=True)
            motion_primitives = robot.get_motion_primitives()
            costs, _ = primitive.get_motion_primitive_costs(grid,
                                                            cost_grid,
                                                            robot.pose,
                                                            robot_path,
                                                            motion_primitives,
                                                            do_use_path=True)
            robot.move_primitive(motion_primitives, np.argmin(costs))
            dist = Pose.cartesian_distance(robot.pose, next_pose)
            if dist <= 2.0:
                reached = True

    trajectory = [[], []]
    for pose in robot.all_poses:
        trajectory[0].append(pose.x)
        trajectory[1].append(pose.y)

    return robot.net_motion, np.array(trajectory)


def extract_robot_poses(
    actions: list[str],
    initial_locations: dict[str, str],
    location_coords: dict[str, tuple[float, float]]
) -> dict[str, list]:
    from common import Pose  # Import locally to avoid circular imports if any

    robot_poses = {}

    # Initialize with start poses
    for robot_name, start_loc in initial_locations.items():
        if start_loc in location_coords:
            robot_poses[robot_name] = [Pose(*location_coords[start_loc])]
        else:
             # Handle cases where location might be 'start' or similar if mapped differently
             # assuming location_coords has 'start' if it's a valid key
             pass

    for action in actions:
        if not action.startswith('move'):
            continue

        parts = action.split()
        if len(parts) == 4: # move robot from to
            robot_name = parts[1]
            to_loc = parts[3]

            if robot_name not in robot_poses:
                # Should have been initialized, but safe fallback or error?
                # If distinct robot names are used that weren't in initial_locations
                pass

            if to_loc in location_coords:
                robot_poses[robot_name].append(Pose(*location_coords[to_loc]))

    return robot_poses

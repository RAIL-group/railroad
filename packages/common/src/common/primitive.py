"""
Functions and Classes useful for determining useful properties of motion
primitives.

A Motion_Primitive is, at its core, a set of poses and a cost. Other functions,
including 'is_primitive_in_collision' and 'get_motion_primitive_costs' are used
to determine properties of the primitives as they relate to the cost grid.
"""

import math
import numpy as np

from . import Pose

COLLISION_VAL = 1
FREE_VAL = 0
UNOBSERVED_VAL = -1
assert (COLLISION_VAL > FREE_VAL)
assert (FREE_VAL > UNOBSERVED_VAL)
OBSTACLE_THRESHOLD = 0.5 * (FREE_VAL + COLLISION_VAL)


class Motion_Primitive():
    def __init__(self, poses, cost, map_data=None):
        self.poses = poses
        self.cost = cost
        self.map_data = map_data

    def transform(self, pose):
        """Transform the motion primitive by a given pose.

        This is equivalent to 'right multiplying' the pose/transform.
        The resulting primitive should be such that the output poses
        are the input poses applied to the pose.
        """
        new_poses = [p * pose for p in self.poses]
        return Motion_Primitive(poses=new_poses, cost=self.cost)


def is_primitive_in_collision(occupancy_grid, motion_primitive):
    """Loop through points in the motion primitive and determines if any are
    in collision. (Note, motion primitive points must be in the world frame,
    as is obtained from the 'get_motion_primitives' method.)"""

    for pose in motion_primitive.poses:
        if occupancy_grid[int(pose.x), int(pose.y)] > OBSTACLE_THRESHOLD:
            return True

    return False


def get_motion_primitive_costs(occupancy_grid,
                               cost_grid,
                               robot_pose,
                               path,
                               motion_primitive_list,
                               do_use_path=True):
    # Initialize the costs
    costs = np.zeros(len(motion_primitive_list))

    def get_cost_via_grid(primitive_pose):
        """Get the cost via the grid"""
        # Subsample the grid to compute the cost
        # Compute the cost of each motion primitive
        # Interpolate over the cost grid
        x = primitive_pose.x - 0.5
        y = primitive_pose.y - 0.5
        ix = int(x)
        iy = int(y)
        dx = x - ix
        dy = y - iy
        cost_x0y0 = min(cost_grid[ix, iy],
                        cost_grid[ix + 1, iy + 1] + 1 + math.sqrt(2))
        cost_x1y0 = min(cost_grid[ix + 1, iy + 0], cost_x0y0 + 1 + 1)
        cost_x0y1 = min(cost_grid[ix + 0, iy + 1], cost_x0y0 + 1 + 1)
        cost_x1y1 = min(cost_grid[ix + 1, iy + 1],
                        cost_x0y0 + 1 + math.sqrt(2))

        cost = ((1 - dx) * (1 - dy) * cost_x0y0 + (dx) * (1 - dy) * cost_x1y0 +
                (1 - dx) * (dy) * cost_x0y1 + (dx) * (dy) * cost_x1y1)
        return cost

    # Determine the 'target pose' for the path method
    try:
        target_pose = Pose(x=path[0, 1], y=path[1, 1])
    except:  # noqa
        do_use_path = False

    def get_cost_via_path(primitive_pose):
        """Get cost for pose via the path method. A correcton factor is added
        via the cost grid method to compensate for the fact that the path
        does not give the exact cost. Using this correction factor allows us
        to more directly switch between the different methods (mostly for
        the purposes of computing training data).
        """
        grid_cost = get_cost_via_grid(robot_pose)
        target_robot_dist = Pose.cartesian_distance(target_pose, robot_pose)
        target_primitive_dist = Pose.cartesian_distance(
            target_pose, primitive_pose)

        return grid_cost - target_robot_dist + target_primitive_dist

    # This check ensures that the sparse plan won't encourage the robot to
    # travel backwards.
    for primitive in motion_primitive_list:
        do_use_path = (
            do_use_path and Pose.cartesian_distance(target_pose, robot_pose) >
            10 * Pose.cartesian_distance(robot_pose, primitive.poses[-1]))

    for c, primitive in enumerate(motion_primitive_list):
        if is_primitive_in_collision(occupancy_grid,
                                     primitive) or get_cost_via_grid(
                                         primitive.poses[-1]) == float('inf'):
            # Determine if any are in collision with the grid.
            costs[c] = 1e10
        elif do_use_path:
            costs[c] = get_cost_via_path(primitive.poses[-1]) + primitive.cost
        else:
            costs[c] = get_cost_via_grid(primitive.poses[-1]) + primitive.cost

    robot_cost = get_cost_via_grid(robot_pose)

    return (costs - robot_cost, robot_cost)

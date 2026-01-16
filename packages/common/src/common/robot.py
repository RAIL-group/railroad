"""
Includes definitions for various 'Robot' classes which are useful for
storing the agent's state and getting the agent's available actions during
navigation. The primary robot class is the "Turtlebot_Robot", which defines
a basic set of motion primitives relevant to differential drive robots.
"""

import copy
import math

from common import Pose
from . import primitive


class Robot:
    """Base robot class.

    All robots must define the 'move' function, which defines how they can
    move through space. The Robot class is also useful for defining how an
    embodied agent travels through space and for storing a history of poses.
    """
    def __init__(self, pose, map_data=None):
        self.pose = copy.copy(pose)
        self.all_poses = [copy.copy(self.pose)]
        self.does_use_motion_primitives = 0
        self.net_motion = 0
        self.map_data = map_data

    def move(self, plan, distance=1.0, stationary=False):
        # Start at 1 so that the "rounded current robot pose" is not used
        if stationary:
            self.all_poses.append(copy.copy(self.all_poses[-1]))
            return

        ii = 1
        remaining_dist = distance
        while remaining_dist > 0 and ii < plan.shape[1]:
            subgoal = plan[:, ii]
            dx = subgoal[0] - self.pose.x
            dy = subgoal[1] - self.pose.y
            step_dist = math.sqrt(dx * dx + dy * dy)
            if step_dist < remaining_dist:
                remaining_dist -= step_dist
                self.pose.x = subgoal[0]
                self.pose.y = subgoal[1]
                self.net_motion += step_dist
                ii += 1
            else:
                self.pose.x += remaining_dist * dx / step_dist
                self.pose.y += remaining_dist * dy / step_dist
                self.net_motion += remaining_dist
                self.all_poses += [copy.copy(self.pose)]
                return

    def max_travel_distance(self, num_recent_poses):
        """Returns the farthest distance between two poses out of the last N poses"""
        max_dist = 0.0
        if len(self.all_poses) < num_recent_poses:
            return 1e10
        for pose_a in self.all_poses[-num_recent_poses:]:
            for pose_b in self.all_poses[-num_recent_poses:]:
                max_dist = max(max_dist,
                               Pose.cartesian_distance(pose_a, pose_b))
        return max_dist

    @property
    def pose_m(self):
        if self.map_data is None:
            return self.pose

        return Pose(x=self.pose.x * self.map_data['resolution'] +
                    self.map_data['x_offset'],
                    y=self.pose.y * self.map_data['resolution'] +
                    self.map_data['y_offset'],
                    yaw=self.pose.yaw)


class Motion_Primitive_Robot(Robot):
    """A simple robot that moves using motion primitives.

    This is the base class for robots that use motion primitives to move.
    As such, the 'get_motion_primitives' function must be defined and the
    move class has been updated to include the selected primitive for motion.
    """
    def __init__(self,
                 pose,
                 num_primitives=32,
                 primitive_length=2.8,
                 map_data=None):
        self.pose = copy.copy(pose)
        self.all_poses = [copy.copy(self.pose)]
        self.net_motion = 0
        self.does_use_motion_primitives = 1
        self.num_primitives = num_primitives
        self.primitive_length = primitive_length
        self.map_data = map_data

    def get_motion_primitives(self):
        """Returns the motion primitives available to the robot at the current
        time (can be a function of robot state)."""
        # Create motion primitives
        return [
            primitive.Motion_Primitive(poses=[
                Pose(x=self.primitive_length *
                     math.cos(2 * math.pi * i / self.num_primitives),
                     y=self.primitive_length *
                     math.sin(2 * math.pi * i / self.num_primitives),
                     yaw=0) * self.pose
            ], cost=self.primitive_length)
            for i in range(self.num_primitives)
        ]

    def move(self, motion_primitives, ind, stationary=False):
        if stationary:
            self.all_poses.append(self.all_poses[-1])
            return

        primitive = motion_primitives[ind]
        self.net_motion += primitive.cost
        p = primitive.poses[-1]
        self.pose = Pose(p.x, p.y, p.yaw)
        for pose in primitive.poses:
            self.all_poses.append(Pose(p.x, p.y, p.yaw))


class Turtlebot_Robot(Motion_Primitive_Robot):
    """A simple 'turtlebot' robot class."""
    def __init__(self,
                 pose,
                 num_primitives=10,
                 max_yaw=math.pi / 2,
                 primitive_length=1.0,
                 map_data=None):
        self.pose = copy.copy(pose)
        self.all_poses = [copy.copy(self.pose)]
        self.net_motion = 0
        self.does_use_motion_primitives = 1
        self.num_primitives = num_primitives
        self.max_yaw = max_yaw
        self.primitive_length = primitive_length
        self.map_data = map_data

    def get_motion_primitives(self):
        """Returns the motion primitives available to the robot at the current
        time (can be a function of robot state)."""
        self.pose = Pose(self.pose.x, self.pose.y, self.pose.yaw)
        r0 = self.primitive_length
        N = self.num_primitives
        angles = [(i * 1.0 / N - 1) * self.max_yaw / 2
                  for i in range(2 * N + 1)]
        primitive_list = [
            primitive.Motion_Primitive(poses=[
                Pose(x=r0 * math.cos(angle) * (math.cos(angle)),
                     y=r0 * math.sin(angle) * (math.cos(angle)),
                     yaw=2 * angle) * self.pose
            ], cost=r0) for angle in angles
        ]
        primitive_list += [
            primitive.Motion_Primitive(
                poses=[Pose(x=0, y=0, yaw=math.pi) * self.pose], cost=2 * r0)
        ]
        return primitive_list

import numpy as np
from sctp.param import VEL_RATIO, RobotType, APPROX_TIME

class Robot:
    _id_counter = 0
    def __init__(self, position, cur_node=None, at_node=False, robot_type=RobotType.Ground, edge=None):
        self.id = Robot._id_counter
        Robot._id_counter += 1
        self.robot_type = robot_type
        self.cur_pose = np.array(position)
        self.last_node = cur_node
        self.unfinished_action = None
        if at_node:
            self.edge = []
            self.at_node = True
        else:
            self.edge = edge
            self.at_node = False
            # assert self.last_node == None
        if self.robot_type == RobotType.Ground:
            self.vel = 1.0
        elif self.robot_type == RobotType.Drone:
            self.vel = 1.0*VEL_RATIO
            self.edge = []
        self.need_action = True 
        self.action = None
        self.remaining_time = 0.0
        self.direction = np.array([0.0, 0.0])
        self._cost_to_target = 0.0
        self.visited_vertices = [self.last_node]
        self.pl_vertex = self.last_node
        self.net_time = 0.0
        self.all_poses = [[self.cur_pose[0],self.cur_pose[1]]]

    def advance_time(self, delta_time):
        advance_distance = self.vel * delta_time
        self._cost_to_target -= advance_distance
        self.remaining_time -= delta_time
        if self.remaining_time < APPROX_TIME:
            self.remaining_time = 0.0
            self._cost_to_target = 0.0
        assert self.remaining_time >= 0.0, 'Remaining time cannot be negative'
        if self.remaining_time == 0.0:
            self.need_action = True
            self.at_node = True
            self.edge = []
            if self.last_node != self.action.target:
                self.pl_vertex = self.last_node
                self.last_node = self.action.target
        else:
            self.at_node = False
            self.edge = [self.last_node, self.action.target]
        
        self._get_coordinates_after_distance(advance_distance)
        self.net_time += delta_time


    def _get_coordinates_after_distance(self, distance):
        self.cur_pose += self.direction * distance
        self.all_poses.append([self.cur_pose[0],self.cur_pose[1]])

    def copy(self):
        new_robot = Robot(position=self.cur_pose.copy(), cur_node=self.last_node, 
                          at_node=self.at_node, robot_type=self.robot_type, edge=self.edge)
        new_robot.need_action = self.need_action
        new_robot.action = self.action
        new_robot.remaining_time = self.remaining_time
        new_robot._cost_to_target = self._cost_to_target
        new_robot.id = self.id
        new_robot.pl_vertex = self.pl_vertex
        new_robot.direction = self.direction.copy()
        new_robot.visited_vertices = self.visited_vertices.copy()
        new_robot.unfinished_action = self.unfinished_action
        return new_robot

    def retarget(self, new_action, distance, direction):
        if not self.remaining_time == 0.0:
            raise NotImplementedError('Time remaining must be 0 for now. '
                                      'Drones cannot switch mid-action')
        self.direction = direction
        self._update_time_to_target(distance)
        # Store the new action
        self.action = new_action
        self.need_action = False

    def _update_time_to_target(self, distance):
        self._cost_to_target = distance
        self.remaining_time = self._cost_to_target / self.vel

    

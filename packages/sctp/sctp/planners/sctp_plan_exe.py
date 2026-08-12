import numpy as np
import sctp
from sctp.param import VEL_RATIO

class SCTPPlanExecution(object):
    def __init__(self, graph, reached_goal, goalID, robot, drones=[], verbose=True):
        self.robot = robot
        self.drones = drones
        self.graph = graph
        self.counter = 0
        self.verbose = verbose
        self.goalID = goalID
        self.goal_reached_fn = reached_goal
        self.action_cost = 0.0
        self.joint_actions = []
        self.costs = []
        self.max_counter = 100
        self.counter = 0
        self.success = False
        self.vertices_status = {}

    def __iter__(self):
        while True:
            # print(f"The drone unfinished action is {self.drones[0].unfinished_action}")
            if len(self.drones) > 0:
                yield {
                    "robot": ([self.robot.cur_pose, self.robot.at_node, self.robot.edge, self.robot.last_node, self.robot.pl_vertex]),
                    "drones": ([[drone.cur_pose, drone.at_node, drone.last_node, drone.unfinished_action] for drone in self.drones]),
                    "observed_pois": (self.vertices_status)
                }
            else:
                yield {
                    "robot": ([self.robot.cur_pose, self.robot.at_node, self.robot.edge, self.robot.last_node,self.robot.pl_vertex]),
                    "drones": None,
                    "observed_pois": (self.vertices_status)
                }
            if self.goal_reached_fn():
                print("----------------- The ground robot reaches its goal ---------------- ")
                self.success = True
                if self.verbose:
                    print(f"Robot position: {self.robot.cur_pose}")
                    if len(self.drones) > 0:
                        print(f"Drone position: ", [drone.cur_pose for drone in self.drones])
                break
            if self.counter > self.max_counter:
                self.success = False
                print("################# Robot failed to find path to goal ##########################")
                break
            self.vertices_status.clear()
            actions_list = [action for action in self.joint_actions]
            self.counter += 1
            while True:
                if self.drones == []:
                    need_replan, actions_list = self.baseline_move(actions_list)
                else:
                    need_replan = self.team_move(actions_list[:len(self.drones)+1])
                if need_replan or len(actions_list) == 0:
                    break
            
    def team_move(self, actions_list):
        self.action_cost = self.update_joint_action(actions_list)    
        self.transition_robot()
        self.transition_drones()
        # Reset the robot and drones
        self.robot.need_action = True
        self.robot.remaining_time = 0.0
        for drone in self.drones:
            drone.need_action = True 
            drone.remaining_time = 0.0
        return True
        
    def baseline_move(self, actions_list):
        self.action_cost = self.update_action(actions_list[0])
        need_replan_robot = self.transition_robot()
        return need_replan_robot, actions_list[1:]
    
    def is_anyrobot_needaction(self):
        if self.robot.need_action:
            return True 
        if len(self.drones) > 0 and any([drone.need_action for drone in self.drones]):
            return True
        return False
    
    def save_joint_actions(self, joint_actions, costs):
        self.joint_actions = joint_actions
        self.costs = costs

    def update_action(self, action, droneID = None):        
        # if droneID is None:
        # for the robot
        assert self.robot.remaining_time == 0.0
        assert self.robot.need_action == True
        end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == action.target][0].coord
        distance = np.linalg.norm(np.array(self.robot.cur_pose) - np.array(end_pos))
        if distance != 0.0:
            robot_direction = (np.array([end_pos[0], end_pos[1]]) - self.robot.cur_pose)/distance
        else:
            robot_direction = np.array([1.0, 1.0])
        self.robot.retarget(action, distance, robot_direction)
        return distance
            
    def transition_robot(self):
        self.robot.advance_time(self.action_cost)
        # sense the current point
        if self.robot.at_node:
            self.robot.need_action = True
            self.robot.remaining_time = 0.0
            vertex_id = self.robot.last_node
            v = [node for node in self.graph.pois if node.id == vertex_id]
            if v:
                self.vertices_status[vertex_id] = v[0].block_status
                v[0].block_prob = float(v[0].block_status)
                if v[0].block_status == 1:
                    return True
        return False

    def transition_drones(self):
        new_block_found = False
        for i, drone in enumerate(self.drones):
            if drone.at_node and drone.last_node ==self.goalID:
                continue
            drone.advance_time(self.action_cost)
            if drone.at_node: # sense the node
                drone.need_action = True 
                drone.remaining_time = 0.0
                vertex_id = drone.last_node
                v = [node for node in self.graph.pois if node.id == vertex_id]
                if v:
                    # sense the node
                    v[0].block_prob = float(v[0].block_status)
                    self.vertices_status[vertex_id] = v[0].block_status
                    new_block_found = True if v[0].block_status == 1 else False
                drone.unfinished_action = None
            else:
                # print(f"Drone {drone.id} is not at node {drone.last_node}, so it has unfinished action {drone.unfinished_action}")
                drone.unfinished_action = drone.action
        return new_block_found
             

    def update_joint_action(self, joint_action):
        # in this function, all robots need action -
        if joint_action is None:
            return
        # for the robot
        if self.robot.need_action:
            assert self.robot.remaining_time == 0.0
            end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == joint_action[-1].target][0].coord
            distance = np.linalg.norm(np.array(self.robot.cur_pose) - np.array(end_pos))
            if distance != 0.0:
                robot_direction = (np.array([end_pos[0], end_pos[1]]) - self.robot.cur_pose)/distance
            else:
                robot_direction = np.array([1.0, 1.0])
            self.robot.retarget(joint_action[-1], distance, robot_direction)
            min_time = distance
        else:
            min_time = self.robot.remaining_time
            
        # for drones
        for i, drone in enumerate(self.drones):
            if drone.last_node == self.goalID:
                continue
            assert drone.remaining_time == 0.0
            assert drone.need_action == True
            end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == joint_action[i].target][0].coord
            distance = np.linalg.norm(np.array(drone.cur_pose) - np.array(end_pos))            
            if distance != 0.0:
                direction = (np.array([end_pos[0], end_pos[1]]) - drone.cur_pose)/distance
            else:
                direction = np.array([1.0, 1.0])
            drone.retarget(joint_action[i], distance, direction)
            if distance > 0.0 and (distance/VEL_RATIO) < min_time:
                min_time = distance/VEL_RATIO
        return min_time
        # self.action_cost = min_time

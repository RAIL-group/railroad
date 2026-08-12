import numpy as np
import pouct_planner
import sctp
from sctp import param
from sctp.core import Action
from sctp.param import RobotType


class SCTPPlanner(object):
    def __init__(self, args, init_graph, goalID, robot, drones=[], rollout_fn = None, 
                tree_depth = 500, n_maps=100, verbose=False):
        self.args = args
        self.verbose = verbose
        self.observed_graph = init_graph
        self.robot = robot 
        self.drones = drones 
        self.goalID = goalID
        self.rollout_fn = rollout_fn
        self.goalNeighbors = []
        self.max_depth = tree_depth
        self.n_maps = n_maps
        # print(f"The number of iterations of MCTS is {self.args.num_iterations} with revisit_pen is {param.REVISIT_PEN}")
        
    def reached_goal(self):
        if not self.robot.at_node:
            return False
        return self.robot.last_node == self.goalID    
    
    def update(self, observations, robot_data, drone_data=None):
        if observations:
            self.observed_graph.update(observations)
        self.robot.cur_pose = np.array([robot_data[0][0],robot_data[0][1]])
        self.robot.at_node = robot_data[1]
        self.robot.edge = robot_data[2].copy()
        self.robot.last_node = robot_data[3]
        self.robot.pl_vertex = robot_data[4]
        self.robot.remaining_time = 0.0
        self.robot.need_action = True
        if drone_data:
            for i, drone in enumerate(self.drones):
                drone.cur_pose = np.array([drone_data[i][0][0], drone_data[i][0][1]])
                drone.at_node = drone_data[i][1]
                drone.edge = []
                drone.last_node = drone_data[i][2]
                drone.unfinished_action = drone_data[i][3]
                drone.remaining_time = 0.0
                drone.need_action = True    

        if self.verbose:
            print('------------robots poses after updating ---------------')
            if self.drones != []:
                print(f"Drones: ", [(drone.cur_pose, drone.at_node, drone.last_node, drone.remaining_time) for drone in self.drones])
            print(f"Robot: {self.robot.cur_pose} at node? {self.robot.at_node} /on edge? {self.robot.edge}/ last node: {self.robot.last_node}")
        
    
    def compute_joint_action(self):
        if self.reached_goal():
            return None, 0.0       
        robot = self.robot.copy()
        if self.drones == []:
            drones = []
        else:
            drones = [drone.copy() for drone in self.drones]
                
        sctpstate = sctp.core.SCTPState(graph=self.observed_graph, goalID=self.goalID, 
                                        robot=robot,
                                        drones=drones,
                                        n_maps=self.n_maps)
        action, cost, [ordering, costs] = pouct_planner.core.po_mcts(
            sctpstate, n_iterations=self.args.num_iterations, C=self.args.C, depth= self.max_depth, rollout_fn=self.rollout_fn)
        # because replanning, so just take some first n+1 action
        if len(ordering) < 1+len(self.drones):
            ordering += [Action(target=self.goalID, rtype=RobotType.Drone) for _ in range(1+len(self.drones) - len(ordering))]
        if self.verbose:
            print("action ordering=", [f"{action}" for action in ordering[:1+len(self.drones)]])
        return ordering, costs

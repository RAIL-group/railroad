import random
import numpy as np
import argparse
from sctp import sctp_graphs as graphs
from sctp import core
from pouct_planner import core as policy
from sctp.robot import Robot
from sctp.param import EventOutcome, RobotType
from sctp.planners import sctp_planner

def test_sctp_planner_lg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='base')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--C', type=int, default=100)
    parser.add_argument('--resolution', type=float, default=0.05)
    
    args = parser.parse_args()
    args.current_seed = args.seed
    # print(args.num_iterations)
    exp_param=50.0
    start, goal, l_graph = graphs.linear_graph_unc()
    for poi in l_graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
    robot = Robot(position=[0.0, 0.0], cur_node=start.id)
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=RobotType.Drone)]
    sctpplanner = sctp_planner.SCTPPlanner(args=args, init_graph=l_graph, 
                                    goalID=goal.id, robot=robot, drones=drones) 
    actions, costs = sctpplanner.compute_joint_action()
    vertices_status = {4: 0}
    
    robot_data = ([[15.0, 0], True, None, 3])
    drone_data = ([[[15.0,0], True, 3]])
    test_data = {"observed_pois": (vertices_status), "robot": robot_data, "drones": drone_data}
    sctpplanner.update(test_data['observed_pois'], test_data['robot'], test_data['drones'])
    if sctpplanner.reached_goal():
        print("The robot reaches it goals")

def test_sctp_planner_sg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='base')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--C', type=int, default=100)
    parser.add_argument('--resolution', type=float, default=0.05)
    
    args = parser.parse_args()
    args.current_seed = args.seed
    # print(args.num_iterations)
    exp_param=50.0
    start, goal, s_graph = graphs.s_graph_unc()
    for poi in s_graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
    robot = Robot(position=[0.0, 0.0], cur_node=start.id)
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=RobotType.Drone)]
    sctpplanner = sctp_planner.SCTPPlanner(args=args, init_graph=s_graph, 
                                    goalID=goal.id, robot=robot, drones=drones) 
    actions, costs = sctpplanner.compute_joint_action()
    for action in actions:
        print(action)
    vertices_status = {5: 0}
    
    robot_data = ([[8.0, 0], True, None, 4])
    drone_data = ([[[8.0,0], True, 4]])
    test_data = {"observed_pois": (vertices_status), "robot": robot_data, "drones": drone_data}
    sctpplanner.update(test_data['observed_pois'], test_data['robot'], test_data['drones'])
    if sctpplanner.reached_goal():
        print("The robot reaches it goals")


if __name__ == '__main__':
    # test_sctp_planner_lg()
    test_sctp_planner_sg()

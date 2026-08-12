import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sctp import sctp_graphs as graphs
from sctp.utils import plotting 
from sctp.robot import Robot
from sctp import core
from sctp.param import EventOutcome, RobotType
from sctp.planners import sctp_planner as planner
from sctp.planners import sctp_plan_exe as plan_loop


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--planner', type=str, default='base')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--C', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=500)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--n_vertex', type=int, default=14)

    args = parser.parse_args(['--save_dir', ''])
    args.seed = 2007
    args.save_dir = '/data/sctp'
    args.planner = 'sctp'
    args.num_drones = 1
    args.num_iterations = 1000
    args.C = 30
    args.max_depth = 35
    args.current_seed = args.seed
    
    return args


def test_sctp_plan_exec_lg():
    # args = parser.parse_args()
    args = _get_args()
    args.planner = 'sctp'
    random.seed(args.seed)
    np.random.seed(args.seed)

    start, goal, l_graph = graphs.linear_graph_unc()
    for poi in l_graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        poi.block_status = 0
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]
    sctpplanner = planner.SCTPPlanner(args=args, init_graph=l_graph, goalID=goal.id, robot=robot, 
                                      drones=drones, rollout_fn=core.sctp_rollout3,verbose=True) 
    
    plan_exec = plan_loop.SCTPPlanExecution(robot=robot, drones=drones, goalID=goal.id,\
                                                   graph=l_graph, reached_goal=sctpplanner.reached_goal)

    for step_data in plan_exec:
        sctpplanner.update(
            step_data['observed_pois'],
            step_data['robot'],
            step_data['drones']
        )
        
        joint_action, cost = sctpplanner.compute_joint_action()
        plan_exec.update_joint_action(joint_action, cost)
    
    cost = robot.net_time
    # print(f"The cost (time) to reach the goal is: {cost}")

    fig = plt.figure(figsize=(10, 10), dpi=300)
    # ax1 = plt.subplot(121)
    plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
    plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
    plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
    plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
    plotting.plot_sctpgraph(l_graph, plt)
    x = [pose[0] for pose in robot.all_poses]
    y = [pose[1] for pose in robot.all_poses]
    plt.scatter(x, y, marker='P', alpha=0.5)
    plt.plot(x, y, color="green")
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        plt.plot(x,y, color="yellow")
        plt.scatter(x, y, marker='s', alpha=0.5)
    plt.title(f'Seed: {args.seed} | Planner: {args.planner} | Cost: {cost:.2f}')
    plt.show()


def test_sctp_plan_exec_dg():
    args = _get_args()
    args.planner = 'sctp'
    random.seed(args.seed)
    np.random.seed(args.seed)
    start, goal, graph = graphs.disjoint_unc()
    for poi in graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        if poi.id == 6 or poi.id==8:
            poi.block_status = int(0)
        if poi.id == 7:
            poi.block_status = int(1)
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]

    planner_robot = robot.copy()
    planner_drones = [drone.copy() for drone in drones]
    sctpplanner = planner.SCTPPlanner(args=args, init_graph=graph, goalID=goal.id, robot=planner_robot, 
                                      drones=planner_drones, rollout_fn=core.sctp_rollout3,verbose=True) 
    plan_exec = plan_loop.SCTPPlanExecution(robot=robot, drones=drones, goalID=goal.id,\
                                                   graph=graph, reached_goal=sctpplanner.reached_goal)
    for step_data in plan_exec:
        print("####################### New navigation #######################################")
        sctpplanner.update(
            step_data['observed_pois'],
            step_data['robot'],
            step_data['drones']
        )
        
        joint_action, cost = sctpplanner.compute_joint_action()
        plan_exec.update_joint_action(joint_action, cost)
    
    cost = robot.net_time

    fig = plt.figure(figsize=(10, 10), dpi=300)
    # ax1 = plt.subplot(121)
    plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
    plt.text(start.coord[0]-0.5, start.coord[1], 'start', fontsize=7)
    plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
    plt.text(goal.coord[0]+0.1, goal.coord[1], 'goal', fontsize=7)
    plotting.plot_sctpgraph(graph, plt)
    # print(robot.all_poses)
    x = [pose[0] for pose in robot.all_poses]
    y = [pose[1] for pose in robot.all_poses]
    plt.scatter(x, y, marker='P', s=4.5, alpha=1.0)
    plt.plot(x, y, color="green")
    for i, (x,y) in enumerate(zip(x,y)):
        xs = [x-0.1, x+0.1]
        ys = [y-0.1, y+0.15]
        plt.text(np.random.choice(xs),np.random.choice(ys), f'gs{i+1}', fontsize=5)
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        plt.plot(x,y, color='yellow', alpha=0.6)
        plt.scatter(x, y, marker='s', s=4.5)
        for i, (x,y) in enumerate(zip(x,y)):
            plt.text(x-0.1,y-0.2, f'ds{i+1}', fontsize=5)
    plt.title(f'Seed: {args.seed} | Planner: {args.planner} | Cost: {cost:.2f}')
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')

    plt.show()

    logfile = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED : {args.seed} | PLANNER : {args.planner} | COST : {cost:0.3f}\n")


def test_sctp_plan_exec_sg():
    args = _get_args()
    args.planner = 'sctp'
    args.seed = 2009
    random.seed(args.seed)
    np.random.seed(args.seed)
    start, goal, graph = graphs.s_graph_unc()
    for poi in graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        if poi.id == 6 or poi.id==8 or poi.id==7:
            poi.block_status = int(0)
        if poi.id == 5 or poi.id==9:
            poi.block_status = int(1)
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]

    planner_robot = robot.copy()
    planner_drones = [drone.copy() for drone in drones]
    sctpplanner = planner.SCTPPlanner(args=args, init_graph=graph, goalID=goal.id,robot=planner_robot, 
                                      drones=planner_drones, rollout_fn=core.sctp_rollout3,verbose=True) 
    plan_exec = plan_loop.SCTPPlanExecution(robot=robot, drones=drones, goalID=goal.id,\
                                                   graph=graph, reached_goal=sctpplanner.reached_goal)
    for step_data in plan_exec:
        print("####################### New navigation #######################################")
        sctpplanner.update(
            step_data['observed_pois'],
            step_data['robot'],
            step_data['drones']
        )
        
        joint_action, cost = sctpplanner.compute_joint_action()
        plan_exec.update_joint_action(joint_action, cost)
    
    cost = robot.net_time

    fig = plt.figure(figsize=(10, 10), dpi=300)
    plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
    plt.text(start.coord[0]-0.5, start.coord[1], 'start', fontsize=7)
    plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
    plt.text(goal.coord[0]+0.1, goal.coord[1], 'goal', fontsize=7)
    plotting.plot_sctpgraph(graph, plt)
    x = [pose[0] for pose in robot.all_poses]
    y = [pose[1] for pose in robot.all_poses]
    plt.scatter(x, y, marker='P', s=4.5, alpha=1.0)
    plt.plot(x, y, color="green")
    for i, (x,y) in enumerate(zip(x,y)):
        xs = [x-0.1, x+0.1]
        ys = [y-0.1, y+0.15]
        plt.text(np.random.choice(xs),np.random.choice(ys), f'gs{i+1}', fontsize=5)
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        plt.plot(x,y, color='yellow', alpha=0.6)
        plt.scatter(x, y, marker='s', s=4.5)
        for i, (x,y) in enumerate(zip(x,y)):
            plt.text(x-0.1,y-0.2, f'ds{i+1}', fontsize=5)
    plt.title(f'Seed: {args.seed} | Planner: {args.planner} | Cost: {cost:.2f}')
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')

    plt.show()

    logfile = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED : {args.seed} | PLANNER : {args.planner} | COST : {cost:0.3f}\n")


def test_sctp_plan_exec_mg():
    args = _get_args()
    args.planner = 'sctp'
    random.seed(args.seed)
    np.random.seed(args.seed)
    start, goal, graph = graphs.m_graph_unc()
    for poi in graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        if poi.id == 8 or poi.id==9 or poi.id==14 or poi.id==19 or poi.id==18:
            poi.block_status = int(0)
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]

    planner_robot = robot.copy()
    planner_drones = [drone.copy() for drone in drones]
    sctpplanner = planner.SCTPPlanner(args=args, init_graph=graph, goalID=goal.id,robot=planner_robot, 
                                      drones=planner_drones, rollout_fn=core.sctp_rollout3, verbose=True) 
    plan_exec = plan_loop.SCTPPlanExecution(robot=robot, drones=drones, goalID=goal.id,\
                                                   graph=graph, reached_goal=sctpplanner.reached_goal)
    for step_data in plan_exec:
        print("####################### New navigation #######################################")
        sctpplanner.update(
            step_data['observed_pois'],
            step_data['robot'],
            step_data['drones']
        )
        
        joint_action, cost = sctpplanner.compute_joint_action()
        plan_exec.update_joint_action(joint_action, cost)
    
    cost = robot.net_time

    fig = plt.figure(figsize=(10, 10), dpi=300)
    plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
    plt.text(start.coord[0]-0.5, start.coord[1], 'start', fontsize=7)
    plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
    plt.text(goal.coord[0]+0.1, goal.coord[1], 'goal', fontsize=7)
    plotting.plot_sctpgraph(graph, plt, verbose=True)
    x = [pose[0] for pose in robot.all_poses]
    y = [pose[1] for pose in robot.all_poses]
    plt.scatter(x, y, marker='P', s=4.5, alpha=1.0)
    plt.plot(x, y, color="green")
    for i, (x,y) in enumerate(zip(x,y)):
        xs = [x-0.1, x+0.1]
        ys = [y-0.1, y+0.15]
        plt.text(np.random.choice(xs),np.random.choice(ys), f'gs{i+1}', fontsize=5)
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        plt.plot(x,y, color='yellow', alpha=0.6)
        plt.scatter(x, y, marker='s', s=4.5)
        for i, (x,y) in enumerate(zip(x,y)):
            plt.text(x-0.1,y-0.2, f'ds{i+1}', fontsize=5)
    plt.title(f'Seed: {args.seed} | Planner: {args.planner} | Cost: {cost:.2f}')
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')

    plt.show()

    logfile = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED : {args.seed} | PLANNER : {args.planner} | COST : {cost:0.3f}\n")


def test_sctp_plan_exec_rg():
    print("")
    args = _get_args()
    args.planner = 'sctp'
    args.seed = 3000
    args.num_iterations = 500
    args.n_samples = 1000
    args.n_vertex = 8
    args.max_depth = 30
    verbose = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    start, goal, graph = graphs.random_graph(n_vertex=args.n_vertex)
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]

    planner_robot = robot.copy()
    graph_forPlot = graph.copy()
    planner_drones = [drone.copy() for drone in drones]
    sctpplanner = planner.SCTPPlanner(args=args, init_graph=graph, goalID=goal.id,robot=planner_robot, 
                                      drones=planner_drones, tree_depth=args.max_depth, 
                                      n_samples=args.n_samples, rollout_fn=core.sctp_rollout3, verbose=True) 
    plan_exec = plan_loop.SCTPPlanExecution(robot=robot, drones=drones, goalID=goal.id,\
                                                   graph=graph, reached_goal=sctpplanner.reached_goal)
        
    plan_exec.max_counter = 25
    for step_data in plan_exec:
        print("####################### New navigation #######################################")
        # print(step_data['drones'])
        sctpplanner.update(
            step_data['observed_pois'],
            step_data['robot'],
            step_data['drones']
        )
        joint_actions, cost = sctpplanner.compute_joint_action()
        plan_exec.save_joint_actions(joint_actions, cost)
    
    cost = robot.net_time
    # fig = plt.figure(figsize=(10, 10), dpi=300)
    x_g = [pose[0] for pose in robot.all_poses]
    y_g = [pose[1] for pose in robot.all_poses]
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpath=[x_g, y_g], dpaths=dpaths, graph_plot=graph_forPlot,\
                             start_coord=start.coord, goal_coord=goal.coord, seed=args.seed, cost=cost, verbose=verbose)
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')
    plt.show()

    logfile = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED : {args.seed} | PLANNER : {args.planner} | COST : {cost:0.3f}\n")


def test_baseline_plan_exec_lg():
    args = _get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    start, goal, graph = graphs.linear_graph_unc()
    for poi in graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        poi.block_status = 0
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    drones = []
    sctpplanner = planner.SCTPPlanner(args=args, init_graph=graph, goalID=goal.id, robot=robot, 
                                      drones=drones, rollout_fn=core.sctp_rollout3, verbose=True) 
    
    plan_exec = plan_loop.SCTPPlanExecution(robot=robot, drones=drones, goalID=goal.id,\
                                                   graph=graph, reached_goal=sctpplanner.reached_goal)

    for step_data in plan_exec:
        sctpplanner.update(
            step_data['observed_pois'],
            step_data['robot'],
            step_data['drones']
        )
        
        joint_action, cost = sctpplanner.compute_joint_action()
        plan_exec.update_joint_action(joint_action, cost)
    
    cost = robot.net_time

    fig = plt.figure(figsize=(10, 10), dpi=300)
    # ax1 = plt.subplot(121)
    plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
    plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
    plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
    plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
    plotting.plot_sctpgraph(graph, plt)
    x = [pose[0] for pose in robot.all_poses]
    y = [pose[1] for pose in robot.all_poses]
    plt.scatter(x, y, marker='P', alpha=0.5)
    plt.plot(x, y, color="green")
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        plt.plot(x,y, color="yellow")
        plt.scatter(x, y, marker='s', alpha=0.5)
    plt.title(f'Seed: {args.seed} | Planner: {args.planner} | Cost: {cost:.2f}')
    plt.show()


def test_baseline_plan_exec_dg():
    args = _get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    start, goal, graph = graphs.disjoint_unc()
    for poi in graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        if poi.id == 6:
            poi.block_status = int(0)
            poi.block_prob = 0.6
        if poi.id == 8:
            poi.block_status = int(0)
        if poi.id == 7:
            poi.block_status = int(1)
            poi.block_prob = 0.9
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    drones = []
    planner_robot = robot.copy()
    planner_drones = [drone.copy() for drone in drones]
    sctpplanner = planner.SCTPPlanner(args=args, init_graph=graph, goalID=goal.id,robot=planner_robot, 
                                      drones=planner_drones, rollout_fn=core.sctp_rollout3, verbose=True) 
    plan_exec = plan_loop.SCTPPlanExecution(robot=robot, drones=drones, goalID=goal.id,\
                                                   graph=graph, reached_goal=sctpplanner.reached_goal)
    for step_data in plan_exec:
        print("####################### Action Execution #######################################")
        sctpplanner.update(
            step_data['observed_pois'],
            step_data['robot'],
            step_data['drones']
        )
        
        joint_action, cost = sctpplanner.compute_joint_action()
        plan_exec.update_joint_action(joint_action, cost)
    
    cost = robot.net_time

    # fig = plt.figure(figsize=(10, 10), dpi=300)
    x_g = [pose[0] for pose in robot.all_poses]
    y_g = [pose[1] for pose in robot.all_poses]
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpath=[x_g, y_g], dpaths=dpaths, 
                             start_coord=start.coord, goal_coord=goal.coord, seed=args.seed, cost=cost, verbose=True)
    
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')
    plt.show()

    logfile = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED : {args.seed} | PLANNER : {args.planner} | COST : {cost:0.3f}\n")


def test_baseline_plan_exec_sg():
    args = _get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    start, goal, graph = graphs.s_graph_unc()
    for poi in graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        if poi.id == 6 or poi.id==8 or poi.id==7:
            poi.block_status = int(0)
        if poi.id == 5 or poi.id==9:
            poi.block_status = int(1)
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    # drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]
    drones = []
    planner_robot = robot.copy()
    planner_drones = [drone.copy() for drone in drones]
    sctpplanner = planner.SCTPPlanner(args=args, init_graph=graph, goalID=goal.id,robot=planner_robot, 
                                      drones=planner_drones, rollout_fn=core.sctp_rollout3,verbose=True) 
    plan_exec = plan_loop.SCTPPlanExecution(robot=robot, drones=drones, goalID=goal.id,\
                                                   graph=graph, reached_goal=sctpplanner.reached_goal)
    for step_data in plan_exec:
        print("####################### New navigation #######################################")
        sctpplanner.update(
            step_data['observed_pois'],
            step_data['robot'],
            step_data['drones']
        )
        
        joint_action, cost = sctpplanner.compute_joint_action()
        plan_exec.update_joint_action(joint_action, cost)
    
    cost = robot.net_time

    # fig = plt.figure(figsize=(10, 10), dpi=300)
    x_g = [pose[0] for pose in robot.all_poses]
    y_g = [pose[1] for pose in robot.all_poses]
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpath=[x_g, y_g], dpaths=dpaths, 
                             start_coord=start.coord, goal_coord=goal.coord, seed=args.seed, cost=cost, verbose=True)
    
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')
    plt.show()

    logfile = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED : {args.seed} | PLANNER : {args.planner} | COST : {cost:0.3f}\n")


def test_baseline_plan_exec_mg():
    args = _get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    start, goal, graph = graphs.m_graph_unc()
    for poi in graph.pois:
        assert poi.block_prob != 0.0
        assert poi.block_prob != 1.0
        if poi.id == 8 or poi.id==9 or poi.id==14 or poi.id==19 or poi.id==18:
            poi.block_status = int(0)
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    # drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]
    drones = []
    planner_robot = robot.copy()
    planner_drones = [drone.copy() for drone in drones]
    sctpplanner = planner.SCTPPlanner(args=args, init_graph=graph, goalID=goal.id,robot=planner_robot, 
                                      drones=planner_drones, rollout_fn=core.sctp_rollout3,verbose=True) 
    plan_exec = plan_loop.SCTPPlanExecution(robot=robot, drones=drones, goalID=goal.id,\
                                                   graph=graph, reached_goal=sctpplanner.reached_goal)
    for step_data in plan_exec:
        print("####################### Action Execution #######################################")
        sctpplanner.update(
            step_data['observed_pois'],
            step_data['robot'],
            step_data['drones']
        )
        
        joint_action, cost = sctpplanner.compute_joint_action()
        plan_exec.update_joint_action(joint_action, cost)
    
    cost = robot.net_time
    x_g = [pose[0] for pose in robot.all_poses]
    y_g = [pose[1] for pose in robot.all_poses]
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpath=[x_g, y_g], dpaths=dpaths, 
                             start_coord=start.coord, goal_coord=goal.coord, seed=args.seed, cost=cost, verbose=True)    
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')
    plt.show()

    logfile = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED : {args.seed} | PLANNER : {args.planner} | COST : {cost:0.3f}\n")



def test_baseline_plan_exec_rg():
    args = _get_args()
    args.planner = 'base'
    args.seed = 3000
    args.num_iterations = 10000
    args.n_samples = 400
    args.max_depth = 20
    args.n_vertex = 8
    verbose = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    start, goal, graph = graphs.random_graph(n_vertex=args.n_vertex)
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    if args.planner == 'base':
        drones = []
    else:
        drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]
    init_graph = graph.copy()
    graph_forPlot = graph.copy()
    planner_robot = robot.copy()
    planner_drones = [drone.copy() for drone in drones]
    sctpplanner = planner.SCTPPlanner(args=args, init_graph=init_graph, goalID=goal.id,robot=planner_robot, 
                                      drones=planner_drones, n_samples=args.n_samples, tree_depth=args.max_depth,
                                      rollout_fn=core.sctp_rollout3,verbose=True) 
    plan_exec = plan_loop.SCTPPlanExecution(robot=robot, drones=drones, goalID=goal.id,\
                                                   graph=graph, reached_goal=sctpplanner.reached_goal)
    plan_exec.max_counter = 30
    for step_data in plan_exec:
        print(f"####################### Action Execution with count {plan_exec.counter} #######################################")
        sctpplanner.update(
            step_data['observed_pois'],
            step_data['robot'],
            step_data['drones']
        )
        
        joint_actions, cost = sctpplanner.compute_joint_action()
        plan_exec.save_joint_actions(joint_actions, cost)
    
    cost = robot.net_time
    x_g = [pose[0] for pose in robot.all_poses]
    y_g = [pose[1] for pose in robot.all_poses]
    dpaths = []
    for drone in drones:
        x = [pose[0] for pose in drone.all_poses]
        y = [pose[1] for pose in drone.all_poses]
        dpaths.append([x, y])
    plotting.plot_plan_exec(graph=graph, plt=plt, name=args.planner, gpath=[x_g, y_g], dpaths=dpaths, graph_plot=graph_forPlot,\
                             start_coord=start.coord, goal_coord=goal.coord, seed=args.seed, cost=cost, verbose=verbose)
    plt.savefig(f'{args.save_dir}/sctp_eval_planner_{args.planner}_seed_{args.seed}.png')
    plt.show()

    logfile = Path(args.save_dir) / f'log_{args.num_drones}.txt'
    with open(logfile, "a+") as f:
        f.write(f"SEED : {args.seed} | PLANNER : {args.planner} | COST : {cost:0.3f}\n")


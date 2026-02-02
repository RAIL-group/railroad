import pytest
import random
import matplotlib.pyplot as plt
from pathlib import Path
from functools import reduce
from types import SimpleNamespace

import procthor
import environments.procthor
from common import Pose
from railroad import operators
from railroad.planner import MCTSPlanner
from railroad.core import Fluent as F, State, get_action_by_name, LiteralGoal
from environments import plotting, utils
from railroad.experimental.environment import EnvironmentInterface
from environments.utils import extract_robot_poses
from operator import and_


def get_args():
    args = SimpleNamespace(
        num_robots=1,
        current_seed=7005,
        resolution=0.05,
        save_dir='./data/test_logs',
    )
    random.seed(args.current_seed)
    return args

def test_single_robot_plotting():
    args = get_args()
    robot_locations = {'robot1': 'start_loc'}
    env = environments.procthor.ProcTHOREnvironment(args, robot_locations)

    objects_by_type = {
        "robot": robot_locations.keys(),
        "location": env.locations.keys(),
        "object": ['egg_55', 'watch_81']
        # "object": env.all_objects  # this stalls in the first mcts call
    }

    init_state = State(
        time=0,
        fluents={
            F("revealed start_loc"),
            F("at robot1 start_loc"), F("free robot1"),
        },
    )

    move_time_fn = env.get_skills_time_fn(skill_name='move')
    pick_time = env.get_skills_time_fn(skill_name='pick')
    place_time = env.get_skills_time_fn(skill_name='place')
    search_time = env.get_skills_time_fn(skill_name='search')
    no_op_time = env.get_skills_time_fn(skill_name='no_op')
    object_find_prob = lambda r, loc, o: 1.0

    move_op = operators.construct_move_operator_blocking(move_time_fn)
    search_op = operators.construct_search_operator(object_find_prob, search_time)
    pick_op = operators.construct_pick_operator_blocking(pick_time)
    place_op = operators.construct_place_operator_blocking(place_time)
    no_op = operators.construct_no_op_operator(no_op_time=no_op_time, extra_cost=100.0)

    sim = EnvironmentInterface(
        init_state,
        objects_by_type,
        [search_op, move_op, pick_op, place_op, no_op],
        env
    )

    all_actions = sim.get_actions()
    mcts = MCTSPlanner(all_actions)

    goal = reduce(and_, [F("at egg_55 safe_19"), F("at watch_81 fridge_12")])
    print(f"Goal: {goal}")

    actions_taken = []
    for _ in range(50):
        action_name = mcts(sim.state, goal, max_iterations=4000, c=300, heuristic_multiplier=2)
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action)
            actions_taken.append(action_name)

        if goal.evaluate(sim.state.fluents):
            print("Goal reached!")
            break

    with open(Path(args.save_dir) / f'vis_single_robot_plan_outputs_{args.current_seed}.txt', 'a') as f:
        f.write(f"Goal: {goal}\n")
        # f.write(f"Max iterations: {max_iterations}\n")
        f.write("Actions taken:\n")
        print(f"Actions taken:")
        for action in actions_taken:
            f.write(f"{action}\n")
            print(action)

        f.write("------------\n")

    # Use the new extraction logic
    robot_poses_dict = extract_robot_poses(actions_taken, robot_locations, env.locations)
    robot_all_poses = robot_poses_dict['robot1']

    cost, trajectory = utils.compute_cost_and_trajectory(
        env.grid, robot_all_poses, 1.0, use_robot_model=True)

    plt.figure(figsize=(8, 8))

    ax = plt.subplot(111)
    plotting.plot_grid_with_robot_trajectory(ax, env.grid, robot_all_poses, trajectory, env.known_graph)
    # add goal to title
    plt.title(f"Single Robot Trajectory Cost: {cost:0.1f}\nGoal: {goal}")

    figpath = Path(args.save_dir) / f'test_visualization_single_robot_{args.current_seed}.png'
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=300)
    print(f"Saved plot to {figpath}")
    assert goal.evaluate(sim.state.fluents)


def test_multi_robot_unknown_plotting():
    args = get_args()
    args.num_robots = 2
    robot_locations = {'robot1': 'start_loc', 'robot2': 'start_loc'}
    env = environments.procthor.ProcTHOREnvironment(args, robot_locations)

    objects_by_type = {
        "robot": robot_locations.keys(),
        "location": env.locations.keys(),
        "object": ['egg_55', 'watch_81']  # env.all_objects,
    }

    init_state = State(
        time=0,
        fluents={
            F("revealed start_loc"),
            F("at robot1 start_loc"), F("free robot1"),
            F("at robot2 start_loc"), F("free robot2"),
        }
    )

    # Create operators
    move_time_fn = env.get_skills_time_fn(skill_name='move')
    search_time = env.get_skills_time_fn(skill_name='search')
    pick_time = env.get_skills_time_fn(skill_name='pick')
    place_time = env.get_skills_time_fn(skill_name='place')
    no_op_time = env.get_skills_time_fn(skill_name='no_op')
    object_find_prob = lambda r, l, o: 1.0
    move_op = operators.construct_move_operator_blocking(move_time_fn)
    search_op = operators.construct_search_operator(object_find_prob, search_time)
    pick_op = operators.construct_pick_operator_blocking(pick_time)
    place_op = operators.construct_place_operator_blocking(place_time)
    no_op = operators.construct_no_op_operator(no_op_time=no_op_time, extra_cost=100.0)

    sim = EnvironmentInterface(init_state, objects_by_type,
                               [search_op, move_op, pick_op, place_op, no_op], env)

    all_actions = sim.get_actions()
    mcts = MCTSPlanner(all_actions)
    goal = reduce(and_, [F("at egg_55 safe_19"), F("at watch_81 fridge_12")])
    print(f"Goal: {goal}")

    consecutive_no_op = 0
    actions_taken = []
    # Increase iterations/steps for multi-robot to actually do something
    for _ in range(60):
        action_name = mcts(sim.state, goal, max_iterations=4000, c=300, heuristic_multiplier=2)
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action)
            actions_taken.append(action_name)
            if action_name.split(' ')[0] == 'no_op':
                consecutive_no_op += 1
            if consecutive_no_op > 4:
                break
        if goal.evaluate(sim.state.fluents):
            print("Goal reached!")
            break

    with open(Path(args.save_dir) / f'vis_multi_robot_unknown_plan_outputs_{args.current_seed}.txt', 'a') as f:
        f.write(f"Goal: {goal}\n")
        f.write("Actions taken:\n")
        print(f"Actions taken:")
        for action in actions_taken:
            f.write(f"{action}\n")
            print(action)
        f.write("------------\n")

    robot_poses_dict = extract_robot_poses(actions_taken, robot_locations, env.locations)

    # Prepare data for plotting
    robots_data = {}
    total_cost = 0

    for robot_name, poses in robot_poses_dict.items():
        cost, trajectory = utils.compute_cost_and_trajectory(
            env.grid, poses, 1.0, use_robot_model=True)
        robots_data[robot_name] = (poses, trajectory)
        total_cost += cost

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)

    plotting.plot_multi_robot_trajectories(ax, env.grid, robots_data, env.known_graph)

    plt.title(f"Multi Robot Trajectory Cost: {total_cost:.1f}\nGoal: {goal}")

    figpath = Path(args.save_dir) / f'test_visualization_unknown_multi_robot_{args.current_seed}.png'
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=300)
    print(f"Saved plot to {figpath}")
    assert goal.evaluate(sim.state.fluents)


def test_multi_robot_known_plotting():
    args = get_args()
    args.num_robots = 2
    robot_locations = {'robot1': 'start_loc', 'robot2': 'start_loc'}
    env = environments.procthor.ProcTHOREnvironment(args, robot_locations)

    objects_by_type = {
        "robot": robot_locations.keys(),
        "location": env.locations.keys(),
        "object": ['egg_55', 'watch_81']  # env.all_objects,
    }

    init_state = State(
        time=0,
        fluents={
            F("revealed start_loc"),
            F("at robot1 start_loc"), F("free robot1"),
            F("at robot2 start_loc"), F("free robot2"),
            F("at watch_81 tvstand_18"),
            F("at egg_55 fridge_12"),
        }
    )

    # Create operators
    move_time_fn = env.get_skills_time_fn(skill_name='move')
    pick_time = env.get_skills_time_fn(skill_name='pick')
    place_time = env.get_skills_time_fn(skill_name='place')
    no_op_time = env.get_skills_time_fn(skill_name='no_op')
    object_find_prob = lambda r, l, o: 1.0
    move_op = operators.construct_move_operator_blocking(move_time_fn)
    pick_op = operators.construct_pick_operator_blocking(pick_time)
    place_op = operators.construct_place_operator_blocking(place_time)
    no_op = operators.construct_no_op_operator(no_op_time=no_op_time, extra_cost=100.0)

    sim = EnvironmentInterface(init_state, objects_by_type,
                               [move_op, pick_op, place_op, no_op], env)

    all_actions = sim.get_actions()
    mcts = MCTSPlanner(all_actions)

    goal = reduce(and_, [F("at egg_55 safe_19"), F("at watch_81 fridge_12")])
    print(f"Goal: {goal}")

    actions_taken = []
    # Increase iterations/steps for multi-robot to actually do something
    for _ in range(20):
        action_name = mcts(sim.state, goal, max_iterations=4000, c=300, heuristic_multiplier=2)
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action)
            actions_taken.append(action_name)

        if goal.evaluate(sim.state.fluents):
            print("Goal reached!")
            break

    with open(Path(args.save_dir) / f'vis_multi_robot_known_plan_outputs_{args.current_seed}.txt', 'a') as f:
        f.write(f"Goal: {goal}\n")
        f.write("Actions taken:\n")
        print(f"Actions taken:")
        for action in actions_taken:
            f.write(f"{action}\n")
            print(action)
        f.write("------------\n")

    robot_poses_dict = extract_robot_poses(actions_taken, robot_locations, env.locations)

    # Prepare data for plotting
    robots_data = {}
    total_cost = 0

    for robot_name, poses in robot_poses_dict.items():
        cost, trajectory = utils.compute_cost_and_trajectory(
            env.grid, poses, 1.0, use_robot_model=True)
        robots_data[robot_name] = (poses, trajectory)
        total_cost += cost

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)

    plotting.plot_multi_robot_trajectories(ax, env.grid, robots_data, env.known_graph)

    plt.title(f"Multi Robot Trajectory Cost: {total_cost:.1f}\nGoal: {goal}")

    figpath = Path(args.save_dir) / f'test_visualization_known_multi_robot_{args.current_seed}.png'
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=300)
    print(f"Saved plot to {figpath}")
    assert goal.evaluate(sim.state.fluents)

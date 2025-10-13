import pytest
from common import Pose
import environments
import environments.procthor
from mrppddl.planner import MCTSPlanner
import random
from mrppddl.core import Fluent as F, State, get_action_by_name
import procthor
import matplotlib.pyplot as plt
from pathlib import Path
from environments import plotting, utils
from environments import Simulator


def get_args():
    args = lambda: None
    args.num_robots = 2
    args.current_seed = 4001
    args.resolution = 0.05
    args.save_dir = '/data/test_logs'
    return args

def get_objects_from_goal_fluents(goal_fluents):
    objects = set()
    for fluent in goal_fluents:
        objects.update({fluent.args[0]})
    return objects

# def get_move_cost_fn(env):
#     def get_move_time(robot, loc_from, loc_to):
#         if robot == "r1":
#             return 50.0
#         return 100.0
#     return get_move_time

def get_ground_truth_action_name(env, robot="r1"):
    target_container_idx = env.target_object_info['container_idxs'][0]
    target_container_name = env.graph.get_node_name_by_idx(target_container_idx)
    action_name = f"search {robot} start {target_container_name}_{target_container_idx} {env.target_object}"
    return action_name

def test_procthor_environment_initialization():
    args = get_args()
    env = environments.procthor.ProcTHOREnvironment(args)
    print(env.locations)
    print(env.target_object)

def test_goal_fluents_to_objects():
    objects = {"objA"}
    goal_fluents = {F("found objA")}
    objects_goal_fluents = get_objects_from_goal_fluents(goal_fluents)
    assert objects == objects_goal_fluents


def test_simulator_with_procthor_map():
    args = get_args()
    env = environments.procthor.ProcTHOREnvironment(args)

    objects_by_type = {
        "robot": [f'r{i+1}' for i in range(args.num_robots)] ,
        "location": env.locations.keys(),
        "object": [env.target_object],
    }

    init_state = State(
            time=0,
            fluents={
                F("revealed start"),
                F("at r1 start"), F("free r1"),
                # F("at r2 start"), F("free r2"),
            },
    )

    move_time_fn = env.get_move_cost_fn()
    search_time = lambda r, l: 10 if r == "r1" else 15
    object_find_prob = lambda r, l, o: 1.0
    move_op = environments.simulator.actions.construct_move_operator(move_time_fn)
    search_op = environments.simulator.actions.construct_search_operator(object_find_prob, search_time)

    sim = Simulator(init_state, objects_by_type, [search_op, move_op], env)

    all_actions = sim.get_actions()
    mcts = MCTSPlanner(all_actions)
    goal_fluents = {F(f"found {env.target_object}")}

    actions_taken = []
    for _ in range(5):
        action_name = mcts(sim.state, goal_fluents, max_iterations=1000, c=10)
        print(f'{action_name=}')
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action)
            print(sim.state.fluents)
            actions_taken.append(action_name)

        if sim.is_goal_reached(goal_fluents):
            print("Goal reached!")
            break

    print(f"Actions taken: {actions_taken}")

    robot_all_poses = [Pose(*env.locations['start'])]
    for action in actions_taken:
        if not action.startswith('move'):
            continue
        _, _, _, to = action.split()
        robot_all_poses.append(Pose(*env.locations[to]))
    print(robot_all_poses)


    cost, trajectory = utils.compute_cost_and_trajectory(env.grid, robot_all_poses, 1.0)

    plt.figure(figsize=(8, 8))
    known_locations = [env.known_graph.get_node_name_by_idx(idx) for idx in env.thor_interface.target_objs_info['container_idxs']]
    plt.suptitle(f"Seed: {args.current_seed} | Target object: {env.target_object}\n"
                 f"Known locations: {known_locations} ")

    ax = plt.subplot(221)
    plt.title('Whole scene graph')
    procthor.plotting.plot_graph(ax, env.known_graph.nodes, env.known_graph.edges)

    ax = plt.subplot(222)
    procthor.plotting.plot_graph_on_grid(ax, env.grid, env.known_graph)
    plt.text(env.robot_pose.x, env.robot_pose.y, '+', color='red', size=6, rotation=45)
    plt.title('Graph over occupancy grid')

    plt.subplot(223)
    top_down_image = env.thor_interface.get_top_down_image()
    plt.imshow(top_down_image)
    plt.title('Top-down view of the map')
    plt.axis('off')

    plt.subplot(224)
    ax = plt.subplot(224)
    plotting.plot_grid_with_robot_trajectory(ax, env.grid, robot_all_poses, trajectory, env.known_graph)
    plt.title(f"Cost: {cost:0.1f}")

    plt.savefig(Path(args.save_dir) / f'object_search_optimistic_{args.current_seed}.png', dpi=1000)

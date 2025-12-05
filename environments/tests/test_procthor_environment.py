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
from environments.core import EnvironmentInterface as Simulator


def get_args():
    args = lambda: None
    args.num_robots = 1
    args.current_seed = 4001
    args.resolution = 0.05
    args.save_dir = './data/test_logs'
    return args


def test_procthor_add_remove_objects():
    args = get_args()
    env = environments.procthor.ProcTHOREnvironment(args)

    # pick a random object
    obj1 = random.choice(list(env.all_objects))
    obj1_idx = int(obj1.split('_')[1])
    loc1_idx = list(env.known_graph.get_adjacent_nodes_idx(obj1_idx, filter_by_type=2))[0]

    loc1 = f"{env.known_graph.get_node_name_by_idx(loc1_idx)}_{loc1_idx}"
    loc1_obj_idxs = env.partial_graph.get_adjacent_nodes_idx(loc1_idx, filter_by_type=3)
    assert len(loc1_obj_idxs) == 0  # nothing is revealed yet

    # reveal loc1
    objects = env.get_objects_at_location(loc1)
    loc1_obj_idxs = env.partial_graph.get_adjacent_nodes_idx(loc1_idx, filter_by_type=3)
    assert len(loc1_obj_idxs) == len(objects['object']) # objects revealed

    # pick a random object from loc1
    obj1 = random.choice(list(objects['object']))
    # remove the object from loc1
    env.remove_object_from_location(obj1, loc1)
    loc1_obj_idxs = env.partial_graph.get_adjacent_nodes_idx(loc1_idx, filter_by_type=3)
    assert len(loc1_obj_idxs) == len(objects['object']) - 1 # one object removed
    # ensure obj1 is not in loc1 anymore
    for obj_idx in loc1_obj_idxs:
        object_name = env.partial_graph.nodes[obj_idx]['object_name']
        assert object_name != obj1

    # place it at another random location
    loc2 = random.choice(list(env.locations.keys() - {'start', loc1}))
    loc2_idx = int(loc2.split('_')[1])
    loc2_obj_idxs = env.partial_graph.get_adjacent_nodes_idx(loc2_idx, filter_by_type=3)
    assert len(loc2_obj_idxs) == 0  # nothing is there in loc2 yet
    env.add_object_at_location(obj1, loc2) # add obj1 to loc2
    loc2_obj_idxs = env.partial_graph.get_adjacent_nodes_idx(loc2_idx, filter_by_type=3)
    assert len(loc2_obj_idxs) == 1  # something is there in loc2
    object_name = env.partial_graph.nodes[loc2_obj_idxs[0]]['object_name']
    assert object_name == obj1  # obj1 is  at loc2



def test_procthor_move_and_search():
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
    move_op = environments.operators.construct_move_operator(move_time_fn)
    search_op = environments.operators.construct_search_operator(object_find_prob, search_time)

    sim = Simulator(init_state, objects_by_type, [search_op, move_op], env)

    all_actions = sim.get_actions()
    mcts = MCTSPlanner(all_actions)
    goal_fluents = {F(f"found {env.target_object}")}
    print(f"Goal fluents: {goal_fluents}")

    actions_taken = []
    for _ in range(15):
        action_name = mcts(sim.state, goal_fluents, max_iterations=1000, c=10)
        print(f'{action_name=}')
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action)
            print(sim.state.fluents)
            actions_taken.append(action_name)

        if sim.goal_reached(goal_fluents):
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

    figpath = Path(args.save_dir) / f'object_search_optimistic_{args.current_seed}.png'
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=1000)


def test_procthor_move_search_pick_place():
    args = get_args()
    args.current_seed = 4001
    env = environments.procthor.ProcTHOREnvironment(args)
    objects = ['teddybear_6', 'pencil_17']
    to_loc = 'garbagecan_5'

    objects_by_type = {
        "robot": [f'r{i+1}' for i in range(args.num_robots)],
        "location": env.locations.keys(),
        "object": objects,
    }

    init_state = State(
            time=0,
            fluents={
                F("revealed start"),
                F("at r1 start"), F("free r1"),
            },
    )
    # Task: Place all objects at random_location
    # goal_fluents = {F(f"at {obj} {to_loc}") for obj in env.all_objects}
    # goal_fluents = {F(f'found {objects[0]}')}
    # goal_fluents = {F(f'holding r1 {objects[0]}')}
    goal_fluents = {F(f"at {objects[0]} {to_loc}")}
    # goal_fluents = {F(f"at {obj} {to_loc}") for obj in objects}

    move_time_fn = env.get_move_cost_fn()
    search_time = lambda r, l: 0 if r == "r1" else 15
    pick_time = lambda r, l, o: 5 if r == "r1" else 7
    place_time = lambda r, l, o: 5 if r == "r1" else 7
    object_find_prob = lambda r, l, o: 1.0
    move_op = environments.operators.construct_move_operator(move_time_fn)
    search_op = environments.operators.construct_search_operator(object_find_prob, search_time)
    pick_op = environments.operators.construct_pick_operator(pick_time)
    place_op = environments.operators.construct_place_operator(place_time)

    sim = Simulator(init_state, objects_by_type, [move_op, search_op, pick_op, place_op], env)

    all_actions = sim.get_actions()
    mcts = MCTSPlanner(all_actions)
    actions_taken = []
    for _ in range(500):
        action_name = mcts(sim.state, goal_fluents, max_iterations=2000, c=10)
        print(f'{action_name=}')
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action)
            print(sim.state.fluents)
            actions_taken.append(action_name)

        if sim.goal_reached(goal_fluents):
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
    procthor.plotting.plot_graph(ax, env.partial_graph.nodes, env.partial_graph.edges)

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

    figpath = Path(args.save_dir) / f'object_search_optimistic_{args.current_seed}.png'
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=1000)

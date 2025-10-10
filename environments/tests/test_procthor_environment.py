import pytest
from common import Pose
import environments
import environments.actions
import environments.procthor
from mrppddl.planner import MCTSPlanner
import random
from mrppddl.core import Fluent as F, State, get_action_by_name
import procthor
import matplotlib.pyplot as plt
from pathlib import Path
from environments import plotting, utils



def get_args():
    args = lambda: None
    args.num_robots = 1
    args.current_seed = 1002
    args.resolution = 0.05
    args.save_dir = '/data/test_logs'
    return args

def get_objects_from_goal_fluents(goal_fluents):
    objects = set()
    for fluent in goal_fluents:
        objects.update({fluent.args[0]})
    return objects

def get_object_likelihood_fn(env):
    def get_object_likelihood(robot, object, location):
        return 1.0
    return get_object_likelihood

def get_move_time_fn(env):
    def get_move_time(robot, loc_from, loc_to):
        if robot == "r1":
            return 50.0
        return 100.0
    return get_move_time

def get_ground_truth_action_name(env, robot="r1"):
    target_container_idx = env.target_object_info['container_idxs'][0]
    target_container_name = env.graph.get_node_name_by_idx(target_container_idx)
    action_name = f"search {robot} start {target_container_name}_{target_container_idx} {env.target_object}"
    return action_name

def test_procthor_environment_initialization():
    args = get_args()
    env = environments.procthor.ProcTHOREnvironment(args)
    print(env.objects_at_locations)
    print(env.target_object)

def test_goal_fluents_to_objects():
    objects = {"objA"}
    goal_fluents = {F("found objA")}
    objects_goal_fluents = get_objects_from_goal_fluents(goal_fluents)
    assert objects == objects_goal_fluents

def test_procthor_environment_upcoming_action():
    args = get_args()
    env = environments.procthor.ProcTHOREnvironment(args)
    target_container_idx = env.target_object_info['container_idxs'][0]
    target_container_name = env.graph.get_node_name_by_idx(target_container_idx) + f"_{target_container_idx}"
    target_object_name = env.target_object

    objects_by_type = {
        "robot": {f'r{i+1}' for i in range(args.num_robots)} ,
        "location": env.objects_at_locations.keys(),
        "object": {target_object_name},
    }

    object_likelihood_fn = get_object_likelihood_fn(env)
    move_time_fn = get_move_time_fn(env)
    search_op = environments.actions.construct_search_operator(object_likelihood_fn, move_time_fn)

    init_state = State(
        time=0,
        fluents={
            F("at r1 start"), F("at r2 start"),
            F("free r1"), F("free r2"),
            F("revealed start")
        }
    )

    sim = environments.simulator.Simulator(init_state, objects_by_type, [search_op], env)
    actions = sim.get_actions()
    action_name = f'search r1 start {target_container_name} {target_object_name}'
    a1 = get_action_by_name(actions, action_name)
    sim.advance(a1)
    assert F("free r1") not in sim.state.fluents
    assert F("free r2") in sim.state.fluents
    assert F("at r1 start") not in sim.state.fluents
    assert F("at r2 start") in sim.state.fluents
    assert F(f"lock-search {target_container_name}") in sim.state.fluents

    random_container_name = random.choice(list(env.objects_at_locations.keys() - {target_container_name, 'start'}))
    action_name = f'search r2 start {random_container_name} {target_object_name}'
    a2 = get_action_by_name(actions, action_name)
    sim.advance(a2)
    assert F("free r1") in sim.state.fluents
    assert F("free r2") not in sim.state.fluents
    assert F(f"revealed {target_container_name}") in sim.state.fluents
    assert F(f"found {target_object_name}") in sim.state.fluents
    assert pytest.approx(sim.state.time) == 50.0
    print(sim.state.fluents)

def test_simulator_with_procthor_map():
    args = get_args()
    env = environments.procthor.ProcTHOREnvironment(args)

    objects_by_type = {
        "robot": [f'r{i+1}' for i in range(args.num_robots)] ,
        "location": env.objects_at_locations.keys(),
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

    object_likelihood_fn = get_object_likelihood_fn(env)
    move_time_fn = env.get_move_cost_fn()
    search_op = environments.actions.construct_search_operator(object_likelihood_fn, move_time_fn)
    all_actions = search_op.instantiate(objects_by_type)

    mcts = MCTSPlanner(all_actions)
    sim = environments.simulator.Simulator(init_state, objects_by_type, [search_op], env)

    goal_fluents = {F(f"found {env.target_object}")}

    actions_taken = []
    for _ in range(5):
        action_name = mcts(sim.state, goal_fluents, max_iterations=1000, c=10)
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action)
            actions_taken.append(action_name)

        if sim.is_goal_reached(goal_fluents):
            print("Goal reached!")
            break

    print(f"Actions taken: {actions_taken}")

    robot_all_poses = [Pose(*env.locations['start'])]
    for action in actions_taken:
        _, _, _, to, _ = action.split()
        robot_all_poses.append(Pose(*env.locations[to]))
    print(robot_all_poses)


    cost, trajectory = utils.compute_cost_and_trajectory(env.grid, robot_all_poses, 1.0)

    plt.figure(figsize=(8, 8))
    known_locations = [env.graph.get_node_name_by_idx(idx) for idx in env.thor_interface.target_objs_info['container_idxs']]
    plt.suptitle(f"Seed: {args.current_seed} | Target object: {env.target_object}\n"
                 f"Known locations: {known_locations} ")

    ax = plt.subplot(221)
    plt.title('Whole scene graph')
    procthor.plotting.plot_graph(ax, env.graph.nodes, env.graph.edges)

    ax = plt.subplot(222)
    procthor.plotting.plot_graph_on_grid(ax, env.grid, env.graph)
    plt.text(env.robot_pose.x, env.robot_pose.y, '+', color='red', size=6, rotation=45)
    plt.title('Graph over occupancy grid')

    plt.subplot(223)
    top_down_image = env.thor_interface.get_top_down_image()
    plt.imshow(top_down_image)
    plt.title('Top-down view of the map')
    plt.axis('off')

    plt.subplot(224)
    ax = plt.subplot(224)
    plotting.plot_grid_with_robot_trajectory(ax, env.grid, robot_all_poses, trajectory, env.graph)
    plt.title(f"Cost: {cost:0.1f}")

    plt.savefig(Path(args.save_dir) / f'object_search_optimistic_{args.current_seed}.png', dpi=1000)

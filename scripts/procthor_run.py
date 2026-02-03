"""
ProcTHOR environment planning demonstration.

Uses the Environment/SymbolicEnvironment architecture for symbolic planning.
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
from pathlib import Path

from common import Pose
import environments
import environments.procthor
from railroad.planner import MCTSPlanner
from railroad.core import Fluent as F, State, get_action_by_name
from railroad._bindings import ff_heuristic
from railroad.environment import SymbolicEnvironment
from railroad._bindings import State
from railroad.dashboard import PlannerDashboard
from railroad import operators
from environments import plotting, utils


def main():
    args = SimpleNamespace(
        num_robots=2,
        current_seed=4001,
        resolution=0.05,
        save_dir='./data/test_logs',
    )
    robot_locations = {
        'robot1': 'start_loc',
        'robot2': 'start_loc',
    }
    procthor_env = environments.procthor.ProcTHOREnvironment(args, robot_locations=robot_locations)
    objects = ['teddybear_6', 'pencil_17']
    to_loc = 'garbagecan_5'

    objects_by_type = {
        "robot": set(robot_locations.keys()),
        "location": set(procthor_env.locations.keys()),
        "object": set(objects),
    }

    initial_fluents = {
        F("revealed start_loc"),
        F("at robot1 start_loc"), F("free robot1"),
        F("at robot2 start_loc"), F("free robot2"),
    }

    initial_state = State(0.0, initial_fluents)

    # Task: Place all objects at target location
    goal = F(f"at {objects[0]} {to_loc}") & F(f"at {objects[1]} {to_loc}")

    # Create operators with time functions from ProcTHOR
    move_time_fn = procthor_env.get_skills_time_fn(skill_name='move')
    search_time = procthor_env.get_skills_time_fn(skill_name='search')
    pick_time = procthor_env.get_skills_time_fn(skill_name='pick')
    place_time = procthor_env.get_skills_time_fn(skill_name='place')

    # Build mapping of object -> actual location from the known graph (ground truth)
    objects_at_locations: dict[str, set[str]] = {}
    for container_idx in procthor_env.known_graph.container_indices:
        location_name = f"{procthor_env.known_graph.get_node_name_by_idx(container_idx)}_{container_idx}"
        object_idxs = procthor_env.known_graph.get_adjacent_nodes_idx(container_idx, filter_by_type=3)
        for obj_idx in object_idxs:
            obj_name = f'{procthor_env.known_graph.get_node_name_by_idx(obj_idx)}_{obj_idx}'
            objects_at_locations.setdefault(location_name, set()).add(obj_name)

    # Create probability function based on ground truth
    object_locations = {obj: loc for loc, objs in objects_at_locations.items() for obj in objs}
    object_find_prob = lambda r, l, o: 0.8 if object_locations.get(o) == l else 0.1

    move_op = operators.construct_move_operator_blocking(move_time_fn)
    search_op = operators.construct_search_operator(object_find_prob, search_time)
    pick_op = operators.construct_pick_operator_blocking(pick_time)
    place_op = operators.construct_place_operator_blocking(place_time)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Create symbolic environment with ground truth from ProcTHOR
    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=[no_op, pick_op, place_op, move_op, search_op],
        true_object_locations=objects_at_locations,
    )

    # Planning loop
    actions_taken = []
    max_iterations = 60

    # Dashboard
    h_value = ff_heuristic(env.state, goal, env.get_actions())
    with PlannerDashboard(goal, initial_heuristic=h_value) as dashboard:
        dashboard.update(state=env.state)

        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(env.state, goal, max_iterations=10000, c=300, max_depth=20, heuristic_multiplier=2)

            if action_name == 'NONE':
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            action = get_action_by_name(all_actions, action_name)
            env.act(action)
            actions_taken.append(action_name)

            tree_trace = mcts.get_trace_from_last_mcts_tree()
            h_value = ff_heuristic(env.state, goal, env.get_actions())
            relevant_fluents = {
                f for f in env.state.fluents
                if any(keyword in f.name for keyword in ["at", "holding", "found", "searched"])
            }
            dashboard.update(
                state=env.state,
                relevant_fluents=relevant_fluents,
                tree_trace=tree_trace,
                step_index=iteration,
                last_action_name=action_name,
                heuristic_value=h_value,
            )

    dashboard.print_history(env.state, actions_taken)

    # Plotting
    robot_poses_dict = utils.extract_robot_poses(actions_taken, robot_locations, procthor_env.locations)

    robots_data = {}
    total_cost = 0

    for robot_name, poses in robot_poses_dict.items():
        cost, trajectory = utils.compute_cost_and_trajectory(
            procthor_env.grid, poses, 1.0, use_robot_model=True)
        robots_data[robot_name] = (poses, trajectory)
        total_cost += cost

    plt.figure(figsize=(16, 8))

    ax1 = plt.subplot(1, 2, 1)
    top_down_image = procthor_env.thor_interface.get_top_down_image(orthographic=True)
    ax1.imshow(top_down_image)
    ax1.axis('off')
    ax1.set_title("Top-down View")

    ax2 = plt.subplot(1, 2, 2)
    plotting.plot_multi_robot_trajectories(ax2, procthor_env.grid, robots_data, procthor_env.known_graph)
    ax2.set_title(f"Multi Robot Trajectory Cost: {total_cost:.1f}")

    figpath = Path(args.save_dir) / f'procthor_run_{args.current_seed}.png'
    figpath.parent.mkdir(parents=True, exist_ok=True)

    figpath_str = figpath if figpath.as_posix().startswith(("/", "./", "../")) else f"./{figpath}"
    plt.savefig(figpath, dpi=300)

    dashboard.console.print(f"\nSaved plot to [yellow]{figpath_str}[/yellow]")


if __name__ == "__main__":
    main()

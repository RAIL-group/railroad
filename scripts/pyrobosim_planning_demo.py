from environments.pyrobosim_v2 import PyRoboSimEnvironment
from railroad import operators
from railroad.planner import MCTSPlanner
from railroad.core import Fluent as F, get_action_by_name
from railroad.environment import EnvironmentInterfaceV2
from railroad._bindings import ff_heuristic
from railroad.dashboard import PlannerDashboard
import argparse
import logging


# Fixed operator times for non-move skills
SEARCH_TIME = 1.0
PICK_TIME = 1.0
PLACE_TIME = 1.0


def main(args):
    # Create the new self-contained environment
    env = PyRoboSimEnvironment(
        world_file=args.world_file,
        show_plot=args.show_plot,
        record_plots=not args.no_video,
    )

    # Goal: move apple and banana to counter
    goal = F("at apple0 counter0") & F("at banana0 counter0")

    # Create operators - move uses distance-based time from environment
    move_time_fn = env.get_move_cost_fn()
    object_find_prob = lambda r, l, o: 0.8 if l == 'my_desk' and o == 'apple0' else 0.2
    move_op = operators.construct_move_operator_blocking(move_time_fn)
    search_op = operators.construct_search_operator(object_find_prob, SEARCH_TIME)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Create interface with environment and operators
    env_interface = EnvironmentInterfaceV2(env, [no_op, pick_op, place_op, move_op, search_op])

    # Planning loop
    actions_taken = []
    max_iterations = 60  # Limit iterations to avoid infinite loops

    # Dashboard
    h_value = ff_heuristic(env_interface.state, goal, env_interface.get_actions())
    with PlannerDashboard(goal, initial_heuristic=h_value) as dashboard:
        # (Optional) initial dashboard update
        dashboard.update(sim_state=env_interface.state)

        for iteration in range(max_iterations):
            # Check if goal is reached
            if goal.evaluate(env_interface.state.fluents):
                break

            # Get available actions
            all_actions = env_interface.get_actions()
            # Plan next action
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(env_interface.state, goal, max_iterations=10000, c=300, max_depth=20)

            if action_name == 'NONE':
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            # Execute action
            action = get_action_by_name(all_actions, action_name)
            env_interface.advance(action, do_interrupt=False, loop_callback_fn=env.canvas.update)
            actions_taken.append(action_name)

            tree_trace = mcts.get_trace_from_last_mcts_tree()
            h_value = ff_heuristic(env_interface.state, goal, env_interface.get_actions())
            relevant_fluents = {
                f for f in env_interface.state.fluents
                if any(keyword in f.name for keyword in ["at", "holding", "found", "searched"])
            }
            dashboard.update(
                sim_state=env_interface.state,
                relevant_fluents=relevant_fluents,
                tree_trace=tree_trace,
                step_index=iteration,
                last_action_name=action_name,
                heuristic_value=h_value,
            )

    # Print the full dashboard history to the console (optional)
    dashboard.print_history(env_interface.state, actions_taken)
    if not args.no_video:
        env.canvas.save_animation(filepath='./data/pyrobosim_planning_demo.mp4')
    env.canvas.wait_for_close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--world-file", type=str, default='./resources/pyrobosim_worlds/test_world.yaml',
                      help="Path to the world YAML file.")
    args.add_argument("--show-plot", action='store_true', help="Whether to show the plot window.")
    args.add_argument("--no-video", action='store_true', help="Whether to disable generating video of the simulation.")
    args = args.parse_args()

    # Turn off all logging of level INFO and below (to suppress pyrobosim logs)
    logging.disable(logging.INFO)

    main(args)

import environments
from mrppddl.planner import MCTSPlanner
from mrppddl.core import Fluent as F, State, get_action_by_name
from environments.core import EnvironmentInterface
from mrppddl._bindings import ff_heuristic
from mrppddl.dashboard import PlannerDashboard
import argparse
import logging


def main(args):
    env = environments.pyrobosim.PyRoboSimEnv(args.world_file,
                                              show_plot=True,
                                              record_plots=True)
    objects = ['apple0']

    objects_by_type = {
        "robot": env.robots.keys(),
        "location": env.locations.keys(),
        "object": objects,
    }

    initial_state = State(
        time=0,
        fluents={
            F("revealed robot1_loc"),
            F("at robot1 robot1_loc"), F("free robot1"),
            F("revealed robot2_loc"),
            F("at robot2 robot1_loc"), F("free robot2"),
        },
    )
    goal = F("at apple0 counter0") & F("at robot1 counter0")

    # Create operators
    move_time_fn = env.get_skills_cost_fn(skill_name='move')
    search_time = env.get_skills_cost_fn(skill_name='search')
    pick_time = env.get_skills_cost_fn(skill_name='pick')
    place_time = env.get_skills_cost_fn(skill_name='place')
    object_find_prob = lambda r, l, o: 0.8 if l == 'my_desk' and o == 'apple0' else 0.2
    move_op = environments.operators.construct_move_operator(move_time_fn)
    search_op = environments.operators.construct_search_operator(object_find_prob, search_time)
    pick_op = environments.operators.construct_pick_operator(pick_time)
    place_op = environments.operators.construct_place_operator(place_time)
    no_op = environments.operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Create simulator
    env_interface = EnvironmentInterface(
        initial_state,
        objects_by_type,
        [no_op, pick_op, place_op, move_op, search_op],
        # [pick_op, place_op, move_op, search_op],
        env
    )

    # Planning loop
    actions_taken = []
    max_iterations = 60  # Limit iterations to avoid infinite loops

    # Dashboard
    h_value = ff_heuristic(initial_state, goal, env_interface.get_actions())
    with PlannerDashboard(goal, initial_heuristic=h_value) as dashboard:
        # (Optional) initial dashboard update
        dashboard.update(sim_state=env_interface.state)

        for iteration in range(max_iterations):
            # Check if goal is reached
            if goal.evaluate(env_interface.state.fluents):
                break

            # Get available actions
            all_actions = env_interface.get_actions()
            # # Plan next action
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

    env.canvas.save_animation(filepath='./data/pyrobosim_planning_demo.mp4')
    env.canvas.wait_for_close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--world_file", type=str, default='./resources/pyrobosim_worlds/test_world.yaml',
                      help="Path to the world YAML file.")
    args = args.parse_args()

    # Turn off all logging of level INFO and below (to suppress pyrobosim logs)
    logging.disable(logging.INFO)

    main(args)

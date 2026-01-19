import environments
from mrppddl.planner import MCTSPlanner
from mrppddl.core import Fluent as F, State, get_action_by_name
from pathlib import Path
from environments.core import EnvironmentInterface
from mrppddl._bindings import ff_heuristic
from mrppddl.dashboard import PlannerDashboard
import argparse


def main(args):
    robot_locations = {
        'robot': 'kitchen',
    }
    env = environments.pyrobosim.PyRoboSimEnv(args.world_file)
    objects = ['apple0']

    objects_by_type = {
        "robot": robot_locations.keys(),
        "location": env.locations.keys(),
        "object": objects,
    }
    print(env.locations)

    initial_state = State(
        time=0,
        fluents={
            F("revealed kitchen"),
            F("at robot kitchen"), F("free robot"),
        },
    )
    goal_fluents = {F(f"at {obj} counter0") for obj in objects}

    # Create operators
    move_time_fn = env.get_skills_cost_fn(skill_name='move')
    search_time = env.get_skills_cost_fn(skill_name='search')
    pick_time = env.get_skills_cost_fn(skill_name='pick')
    place_time = env.get_skills_cost_fn(skill_name='place')
    object_find_prob = lambda r, l, o: 1.0
    move_op = environments.operators.construct_move_operator_nonblocking(move_time_fn)
    search_op = environments.operators.construct_search_operator(object_find_prob, search_time)
    pick_op = environments.operators.construct_pick_operator_nonblocking(pick_time)
    place_op = environments.operators.construct_place_operator_nonblocking(place_time)
    # no_op = environments.operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    print(move_time_fn('robot', 'kitchen', 'my_desk'))
    exit()

    # Create simulator
    env_interface = EnvironmentInterface(
        initial_state,
        objects_by_type,
        # [no_op, pick_op, place_op, move_op, search_op],
        [pick_op, place_op, move_op, search_op],
        env
    )
    action_list = [
        ('move robot kitchen my_desk'),
        ('search robot my_desk apple0'),
        ('pick robot my_desk apple0'),
        ('move robot my_desk counter0'),
        ('place robot counter0 apple0'),
        # ('no_op robot'),
        ('move robot counter0 kitchen'),
        ('NONE'),
    ]
    # Planning loop
    actions_taken = []
    max_iterations = 60  # Limit iterations to avoid infinite loops

    # Dashboard
    h_value = ff_heuristic(initial_state, goal_fluents, env_interface.get_actions())
    # with PlannerDashboard(goal_fluents, initial_heuristic=h_value) as dashboard:
    if True:
        # (Optional) initial dashboard update
        # dashboard.update(sim_state=env_interface.state)

        for iteration in range(max_iterations):
            # Check if goal is reached
            if env_interface.is_goal_reached(goal_fluents):
                break

            # Get available actions
            all_actions = env_interface.get_actions()
            # # Plan next action
            # mcts = MCTSPlanner(all_actions)
            # action_name = 'NONE'
            action_name = action_list.pop(0)

            if action_name == 'NONE':
                # dashboard.console.print("No more actions available. Goal may not be achievable.")
                break
            print("========================================")
            print(f"Iteration {iteration}: Selected action: {action_name}, press Enter to execute...")
            # Execute action
            action = get_action_by_name(all_actions, action_name)
            print(action)
            env_interface.advance(action, do_interrupt=False, loop_callback=env.canvas.update)
            actions_taken.append(action_name)
            print(f'Fluents: {env_interface.state.fluents}')
            print("========================================")
            # tree_trace = mcts.get_trace_from_last_mcts_tree()
            # h_value = ff_heuristic(env_interface.state, goal_fluents, env_interface.get_actions())
            relevant_fluents = {
                f for f in env_interface.state.fluents
                if any(keyword in f.name for keyword in ["at", "holding", "found", "searched"])
            }
            # dashboard.update(
            #     sim_state=env_interface.state,
            #     relevant_fluents=relevant_fluents,
            #     tree_trace=tree_trace,
            #     step_index=iteration,
            #     last_action_name=action_name,
            #     heuristic_value=h_value,
            # )

    # Print the full dashboard history to the console (optional)
    # dashboard.print_history(env_interface.state, actions_taken)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--world_file", type=str, default='./pyrobosim_worlds/test_world.yaml',
                      help="Path to the world YAML file.")
    args = args.parse_args()

    main(args)

"""
Multi-Object Search and Place Task

This script demonstrates a more complex planning scenario where a robot must:
1. Search for multiple objects scattered across different locations
2. Handle missing objects (some searches will fail)
3. Pick up found objects and transport them to target locations
4. Place objects at their designated spots

The environment simulates a household with multiple rooms where items are
disorganized and some items are missing entirely.

Uses the new Goal API for defining planning objectives.
"""

from functools import reduce
from operator import and_

import numpy as np
from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl._bindings import ff_heuristic_goal
from mrppddl.planner import MCTSPlanner
from mrppddl.dashboard import PlannerDashboard
import environments
from environments import SimpleEnvironment
from environments.core import EnvironmentInterface


# Define locations with coordinates (for move cost calculation)
LOCATIONS = {
    "start_loc": np.array([-5, -5]),
    "living_room": np.array([0, 0]),
    "kitchen": np.array([10, 0]),
    "bedroom": np.array([0, 12]),
    "office": np.array([10, 12]),
}

# Define where objects actually are (ground truth)
# Note: Mug is missing - it's not at any location!
OBJECTS_AT_LOCATIONS = {
    "living_room": {"object": {"Notebook", "Pillow"}},
    "kitchen": {"object": {"Clock", "Mug"}},
    "bedroom": {"object": {"Knife"}},
    "office": {"object": set()},  # Empty
    "start_loc": {"object": set()},  # Empty
}


def main():

    # Initialize environment
    robot_locations = {'robot1': 'start_loc', 'robot2': 'start_loc'}
    env = SimpleEnvironment(LOCATIONS, OBJECTS_AT_LOCATIONS, robot_locations)

    # Define the objects we're looking for
    objects_of_interest = ["Knife", "Notebook", "Clock", "Mug", "Pillow"]

    # Define initial state
    initial_state = State(
        time=0,
        fluents={
            # Robot starts in living room
            F("at", "robot1", "start_loc"),
            F("at", "robot2", "start_loc"),
            F("free", "robot1"),
            F("free", "robot2"),
        },
    )

    # Define goal: all items at their proper locations
    # Using Goal API: reduce(and_, [...]) creates an AndGoal
    goal = reduce(and_, [
        F("at Knife kitchen"),
        F("at Mug kitchen"),
        F("at Clock bedroom"),
        F("at Pillow bedroom"),
        F("at Notebook office"),
    ])

    # Initial objects by type (robot only knows about some objects initially)
    objects_by_type = {
        "robot": ["robot1", "robot2"],
        "location": list(LOCATIONS.keys()),
        "object": objects_of_interest,  # Robot knows these objects exist
    }


    # Create operators
    move_time_fn = env.get_skills_cost_fn(skill_name='move')
    search_time = env.get_skills_cost_fn(skill_name='search')
    pick_time = env.get_skills_cost_fn(skill_name='pick')
    place_time = env.get_skills_cost_fn(skill_name='place')
    object_find_prob=lambda r, l, o: 0.6 if 'kitchen' in l else 0.4
    move_op = environments.operators.construct_move_operator(move_time_fn)
    search_op = environments.operators.construct_search_operator(object_find_prob, search_time)
    pick_op = environments.operators.construct_pick_operator(pick_time)
    place_op = environments.operators.construct_place_operator(place_time)
    no_op = environments.operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Create simulator
    sim = EnvironmentInterface(
        initial_state,
        objects_by_type,
        [no_op, move_op, search_op, pick_op, place_op],
        env
    )

    # Planning loop
    print("Starting planning and execution...\n")
    actions_taken = []
    max_iterations = 60  # Limit iterations to avoid infinite loops

    # Dashboard
    h_value = ff_heuristic_goal(initial_state, goal, sim.get_actions())
    with PlannerDashboard(goal, initial_heuristic=h_value) as dashboard:
        # (Optional) initial dashboard update
        dashboard.update(sim_state=sim.state)

        for iteration in range(max_iterations):
            # Check if goal is reached
            if goal.evaluate(sim.state.fluents):
                break

            # Get available actions
            all_actions = sim.get_actions()

            # Plan next action
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(sim.state, goal, max_iterations=4000, c=300, max_depth=20)

            if action_name == 'NONE':
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            # Execute action
            action = get_action_by_name(all_actions, action_name)

            sim.advance(action, do_interrupt=False)
            actions_taken.append(action_name)

            # Print relevant state information
            tree_trace = mcts.get_trace_from_last_mcts_tree()
            relevant_fluents = {
                f for f in sim.state.fluents
                if any(keyword in f.name for keyword in ["at", "holding", "found", "searched"])
            }
            dashboard.update(
                sim_state=sim.state,
                relevant_fluents=relevant_fluents,
                tree_trace=tree_trace,
                step_index=iteration,
                last_action_name=action_name,
                heuristic_value=h_value,
            )

    # Print the full dashboard history to the console (optional)
    dashboard.print_history(sim.state, actions_taken)


if __name__ == "__main__":
    main()

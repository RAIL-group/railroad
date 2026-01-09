"""
Multi-Object Search and Place Task

This script demonstrates a more complex planning scenario where a robot must:
1. Search for multiple objects scattered across different locations
2. Handle missing objects (some searches will fail)
3. Pick up found objects and transport them to target locations
4. Place objects at their designated spots

The environment simulates a household with multiple rooms where items are
disorganized and some items are missing entirely.

Updated to use new Goal objects (AndGoal, LiteralGoal) for complex goal support.
The goal is defined as an AND of literals, which is equivalent to the previous
fluent set but demonstrates the new Goal API and enables future OR goal usage.
"""

import numpy as np
from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl.planner import MCTSPlanner
from mrppddl.dashboard import PlannerDashboard
import environments
from environments.core import EnvironmentInterface
from environments import SimpleEnvironment
from mrppddl._bindings import ff_heuristic, ff_heuristic_goal, AndGoal, LiteralGoal, OrGoal

# Fancy error handling; shows local vars
from rich.traceback import install
install(show_locals=True)


# Define locations with coordinates (for move cost calculation)
LOCATIONS = {
    "living_room": np.array([0, 0]),
    "kitchen": np.array([10, 0]),
    "bedroom": np.array([0, 12]),
    "office": np.array([10, 12]),
    "den": np.array([15, 5]),
}

# Define where objects actually are (ground truth)
# Note: Mug is missing - it's not at any location!
OBJECTS_AT_LOCATIONS = {
    "living_room": {"object": {"Remote"}},
    "kitchen": {"object": {"Cookie", "Plate"}},
    "bedroom": {"object": set()},
    "office": {"object": {"Couch"}},
    "den": {"object": set()},
}


def main():

    # Initialize environment
    robot_locations = {"robot1": "living_room",
                                 "robot2": "living_room"}
    env = SimpleEnvironment(LOCATIONS, OBJECTS_AT_LOCATIONS,
                            robot_locations=robot_locations)
    # Define the objects we're looking for
    objects_of_interest = ["Remote", "Cookie", "Plate", "Couch"]

    # Define initial state
    initial_state = State(
        time=0,
        fluents={
            # Robots free and start in (revealed) living room
            F("free robot1"),
            F("free robot2"),
            F("at robot1 living_room"),
            F("at robot2 living_room"),
            F("revealed living_room"),
            F("at Remote living_room"),
            F("found Remote"),
            F("revealed den"),
        },
    )

    # Define goal: all items at their proper locations
    # Using new Goal objects (complex goal support)
    goal = AndGoal([
        OrGoal([
            LiteralGoal(F("at Remote den")),
            LiteralGoal(F("at Plate den")),
        ]),
        OrGoal([
            LiteralGoal(F("at Cookie den")),
            LiteralGoal(F("at Couch den")),
        ]),
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
    object_find_prob = lambda r, loc, o: 0.8 if o in OBJECTS_AT_LOCATIONS.get(loc, dict()).get("object", dict()) else 0.2
    move_op = environments.operators.construct_move_operator(move_time_fn)
    search_op = environments.operators.construct_search_operator(object_find_prob, search_time)
    pick_op = environments.operators.construct_pick_operator(pick_time)
    place_op = environments.operators.construct_place_operator(place_time)
    no_op = environments.operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Create simulator
    sim = EnvironmentInterface(
        initial_state,
        objects_by_type,
        [no_op, pick_op, place_op, move_op, search_op],
        env
    )

    # Planning loop
    actions_taken = []
    max_iterations = 60  # Limit iterations to avoid infinite loops

    # Dashboard
    # Use new ff_heuristic_goal for efficient Goal object support
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

            # Plan next action using Goal object
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(sim.state, goal, max_iterations=4000, c=300, max_depth=20)

            if action_name == 'NONE':
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            # Execute action
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action, do_interrupt=False)
            actions_taken.append(action_name)

            tree_trace = mcts.get_trace_from_last_mcts_tree()
            h_value = ff_heuristic_goal(sim.state, goal, sim.get_actions())
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

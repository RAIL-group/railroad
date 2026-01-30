"""Multi-Object Search and Place Task.

This example demonstrates a complex planning scenario where multiple robots must:
1. Search for objects scattered across different locations
2. Handle probabilistic search outcomes
3. Pick up found objects and transport them to target locations
4. Coordinate using no-op (wait) actions

The environment simulates a household with multiple rooms where items need
to be reorganized.
"""

from functools import reduce
from operator import and_

import numpy as np

from railroad.core import Fluent as F, State, get_action_by_name, ff_heuristic
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.environment import EnvironmentInterface, SimpleEnvironment


# Define locations with coordinates (for move cost calculation)
LOCATIONS = {
    "start_loc": np.array([-5, -5]),
    "living_room": np.array([0, 0]),
    "kitchen": np.array([10, 0]),
    "bedroom": np.array([0, 12]),
    "office": np.array([10, 12]),
}

# Define where objects actually are (ground truth)
OBJECTS_AT_LOCATIONS = {
    "living_room": {"object": {"Notebook", "Pillow"}},
    "kitchen": {"object": {"Clock", "Mug"}},
    "bedroom": {"object": {"Knife"}},
    "office": {"object": set()},
    "start_loc": {"object": set()},
}


def main() -> None:
    """Run the multi-object search example."""
    # Initialize environment
    robot_locations = {"robot1": "start_loc", "robot2": "start_loc"}
    env = SimpleEnvironment(LOCATIONS, OBJECTS_AT_LOCATIONS, robot_locations)

    # Define the objects we're looking for
    objects_of_interest = ["Knife", "Notebook", "Clock", "Mug", "Pillow"]

    # Define initial state
    initial_state = State(
        time=0,
        fluents={
            F("at", "robot1", "start_loc"),
            F("at", "robot2", "start_loc"),
            F("free", "robot1"),
            F("free", "robot2"),
        },
    )

    # Define goal: all items at their proper locations
    goal = (
        F("at Knife kitchen")
        & F("at Mug kitchen")
        & F("at Clock bedroom")
        & F("at Pillow bedroom")
        & F("at Notebook office")
    )

    # Objects by type
    objects_by_type = {
        "robot": ["robot1", "robot2"],
        "location": list(LOCATIONS.keys()),
        "object": objects_of_interest,
    }

    # Create operators
    move_time_fn = env.get_skills_time_fn(skill_name="move")
    search_time = env.get_skills_time_fn(skill_name="search")
    pick_time = env.get_skills_time_fn(skill_name="pick")
    place_time = env.get_skills_time_fn(skill_name="place")

    # Probabilistic search - higher success rate in kitchen
    object_find_prob = lambda r, loc, o: 0.6 if "kitchen" in loc else 0.4

    move_op = operators.construct_move_operator_blocking(move_time_fn)
    search_op = operators.construct_search_operator(object_find_prob, search_time)
    pick_op = operators.construct_pick_operator_blocking(pick_time)
    place_op = operators.construct_place_operator_blocking(place_time)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Create simulator
    sim = EnvironmentInterface(
        initial_state, objects_by_type, [no_op, move_op, search_op, pick_op, place_op], env
    )

    # Planning loop
    actions_taken = []
    max_iterations = 60

    h_value = ff_heuristic(initial_state, goal, sim.get_actions())
    with PlannerDashboard(goal, initial_heuristic=h_value) as dashboard:
        dashboard.update(sim_state=sim.state)

        for iteration in range(max_iterations):
            if goal.evaluate(sim.state.fluents):
                dashboard.console.print("[green]Goal achieved![/green]")
                break

            all_actions = sim.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(sim.state, goal, max_iterations=4000, c=300, max_depth=20)

            if action_name == "NONE":
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            action = get_action_by_name(all_actions, action_name)
            sim.advance(action, do_interrupt=False)
            actions_taken.append(action_name)

            tree_trace = mcts.get_trace_from_last_mcts_tree()
            h_value = ff_heuristic(sim.state, goal, sim.get_actions())
            relevant_fluents = {
                f
                for f in sim.state.fluents
                if any(kw in f.name for kw in ["at", "holding", "found", "searched"])
            }
            dashboard.update(
                sim_state=sim.state,
                relevant_fluents=relevant_fluents,
                tree_trace=tree_trace,
                step_index=iteration,
                last_action_name=action_name,
                heuristic_value=h_value,
            )

    dashboard.print_history(sim.state, actions_taken)


if __name__ == "__main__":
    main()

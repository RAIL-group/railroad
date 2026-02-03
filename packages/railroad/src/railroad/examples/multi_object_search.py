"""Multi-Object Search and Place Task.

This example demonstrates a complex planning scenario where multiple robots must:
1. Search for objects scattered across different locations
2. Handle probabilistic search outcomes
3. Pick up found objects and transport them to target locations
4. Coordinate using no-op (wait) actions

The environment simulates a household with multiple rooms where items need
to be reorganized.
"""

import numpy as np

from railroad.core import Fluent as F, get_action_by_name, ff_heuristic
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.environment import SymbolicEnvironment
from railroad._bindings import State


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
    "living_room": {"Notebook", "Pillow"},
    "kitchen": {"Clock", "Mug"},
    "bedroom": {"Knife"},
    "office": set(),
    "start_loc": set(),
}

# Fixed operator times for symbolic planning
ROBOT_VELOCITY = 1.0
SEARCH_TIME = 5.0
PICK_TIME = 5.0
PLACE_TIME = 5.0


def main() -> None:
    """Run the multi-object search example."""
    # Define the objects we're looking for
    objects_of_interest = ["Knife", "Notebook", "Clock", "Mug", "Pillow"]

    # Define initial fluents
    initial_fluents = {
        F("at", "robot1", "start_loc"),
        F("at", "robot2", "start_loc"),
        F("free", "robot1"),
        F("free", "robot2"),
    }

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
        "robot": {"robot1", "robot2"},
        "location": set(LOCATIONS.keys()),
        "object": set(objects_of_interest),
    }

    # # Probabilistic search - higher success rate in kitchen
    # object_find_prob = lambda r, loc, o: 0.6 if "kitchen" in loc else 0.4
    # Higher find probability if object is actually at the location
    def object_find_prob(robot: str, loc: str, obj: str) -> float:
        objects_here = OBJECTS_AT_LOCATIONS.get(loc, set())
        return 0.8 if obj in objects_here else 0.2


    # Distance-based move time function
    def move_time(robot: str, loc_from: str, loc_to: str) -> float:
        distance = float(np.linalg.norm(LOCATIONS[loc_from] - LOCATIONS[loc_to]))
        return distance / ROBOT_VELOCITY

    # Create operators - move uses distance-based time
    move_op = operators.construct_move_operator_blocking(move_time)
    search_op = operators.construct_search_operator(object_find_prob, SEARCH_TIME)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Initialize symbolic environment with initial state
    initial_state = State(0.0, initial_fluents, [])
    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=[no_op, move_op, search_op, pick_op, place_op],
        true_object_locations=OBJECTS_AT_LOCATIONS,
    )

    # Planning loop
    actions_taken = []
    max_iterations = 60

    h_value = ff_heuristic(env.state, goal, env.get_actions())
    with PlannerDashboard(goal, initial_heuristic=h_value) as dashboard:
        dashboard.update(state=env.state)

        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Goal achieved![/green]")
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(env.state, goal, max_iterations=4000, c=300, max_depth=20)

            if action_name == "NONE":
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            action = get_action_by_name(all_actions, action_name)
            env.act(action, do_interrupt=False)
            actions_taken.append(action_name)

            tree_trace = mcts.get_trace_from_last_mcts_tree()
            h_value = ff_heuristic(env.state, goal, env.get_actions())
            relevant_fluents = {
                f
                for f in env.state.fluents
                if any(kw in f.name for kw in ["at", "holding", "found", "searched"])
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


if __name__ == "__main__":
    main()

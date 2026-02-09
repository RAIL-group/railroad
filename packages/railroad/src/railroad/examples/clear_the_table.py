"""Clear the Table Task.

This example demonstrates a "none" goal type where the objective is to ensure
NO objects remain on a table. The goal is expressed as an AND of negated literals:
for each object that starts on the table, the goal requires NOT(at object table).

This tests the goal system's handling of negated fluents in goals, which is useful
for scenarios where the desired state is the absence of certain predicates rather
than their presence.
"""

from functools import reduce
from operator import and_

import numpy as np

from railroad.core import Fluent as F, get_action_by_name
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.environment import SymbolicEnvironment
from railroad._bindings import State


# Define locations
LOCATIONS = {
    "living_room": np.array([0, 0]),
    "kitchen": np.array([5, 0]),
    "table": np.array([2, 3]),  # The table location to clear
    "shelf": np.array([8, 3]),  # Destination for cleared items
}

# Define where objects actually are (ground truth)
OBJECTS_AT_LOCATIONS = {
    "living_room": set(),
    "kitchen": set(),
    "table": {"Book", "Mug", "Vase"},  # Objects to clear
    "shelf": set(),  # Destination
}

# Fixed operator times for symbolic planning
ROBOT_VELOCITY = 1.0
PICK_TIME = 5.0
PLACE_TIME = 5.0


def main(
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    video_fps: int | None = None,
    video_dpi: int | None = None,
) -> None:
    """Run the clear-the-table example."""
    # Objects that need to be cleared from the table
    objects_on_table = ["Book", "Mug", "Vase"]

    # Define initial fluents
    initial_fluents = {
        # Robot free and starts in living room
        F("free robot1"),
        F("at robot1 living_room"),
        # Objects are on the table
        F("at Book table"),
        F("at Mug table"),
        F("at Vase table"),
    }

    # Define goal: NO objects on the table
    # This is a "none" goal - we want the absence of all objects at the table
    # Expressed as: forall obj in objects_on_table: NOT(at obj table)
    goal = reduce(and_, [~F(f"at {obj} table") for obj in objects_on_table])

    # Objects by type
    objects_by_type = {
        "robot": {"robot1"},
        "location": set(LOCATIONS.keys()),
        "object": set(objects_on_table),
    }

    # Create operators with fixed times
    # Distance-based move time function
    def move_time(robot: str, loc_from: str, loc_to: str) -> float:
        distance = float(np.linalg.norm(LOCATIONS[loc_from] - LOCATIONS[loc_to]))
        return distance / ROBOT_VELOCITY

    move_op = operators.construct_move_operator_blocking(move_time)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)

    # Initialize symbolic environment with initial state
    initial_state = State(0.0, initial_fluents, [])
    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=[pick_op, place_op, move_op],
        true_object_locations=OBJECTS_AT_LOCATIONS,
    )

    # Planning loop
    max_iterations = 40

    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "holding", "found"])
    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        for iteration in range(max_iterations):
            # Check if goal is reached
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Goal achieved! Table is clear.[/green]")
                break

            # Get available actions
            all_actions = env.get_actions()

            # Plan next action
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(env.state, goal, max_iterations=4000, c=300, max_depth=20)

            if action_name == "NONE":
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            # Execute action
            action = get_action_by_name(all_actions, action_name)
            env.act(action)
            dashboard.update(mcts, action_name)

    location_coords = {name: (float(c[0]), float(c[1])) for name, c in LOCATIONS.items()}
    dashboard.show_plots(
        save_plot=save_plot, show_plot=show_plot, save_video=save_video,
        video_fps=video_fps, video_dpi=video_dpi,
        location_coords=location_coords,
    )


if __name__ == "__main__":
    main()

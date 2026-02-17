"""Find and Move Couch Task.

This example demonstrates:
1. OR goals - satisfying one of several possible conditions
2. Multi-robot coordination with search and transport
3. Complex goal expressions using the fluent operator syntax

The goal is to move either a Remote OR Plate to the den, AND either a Cookie
OR Couch to the den. This shows how OR goals allow flexibility in achieving
objectives.
"""

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
    "kitchen": np.array([10, 0]),
    "bedroom": np.array([0, 12]),
    "office": np.array([10, 12]),
    "den": np.array([15, 5]),
}

# Define where objects actually are (ground truth)
OBJECTS_AT_LOCATIONS = {
    "living_room": {"Remote"},
    "kitchen": {"Cookie", "Plate"},
    "bedroom": set(),
    "office": {"Couch"},
    "den": set(),
}

# Fixed operator times for symbolic planning
ROBOT_VELOCITY = 1.0
SEARCH_TIME = 5.0
PICK_TIME = 5.0
PLACE_TIME = 5.0


def main(
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    video_fps: int = 60,
    video_dpi: int = 150,
) -> None:
    """Run the find-and-move-couch example."""
    # Define the objects we're looking for
    objects_of_interest = ["Remote", "Cookie", "Plate", "Couch"]

    # Define initial fluents - robots start with some knowledge
    initial_fluents = {
        F("free robot1"),
        F("free robot2"),
        F("at robot1 living_room"),
        F("at robot2 living_room"),
        # Living room is revealed (searched), so we know Remote is there
        F("revealed living_room"),
        F("at Remote living_room"),
        F("found Remote"),
        # Den is revealed but empty
        F("revealed den"),
    }

    # Define goal using OR expressions:
    # (Remote at den OR Plate at den) AND (Cookie at den OR Couch at den)
    # This allows flexibility - the planner can choose which objects to move
    goal = (F("at Remote den") | F("at Plate den")) & (F("at Cookie den") | F("at Couch den"))

    # Objects by type
    objects_by_type = {
        "robot": {"robot1", "robot2"},
        "location": set(LOCATIONS.keys()),
        "object": set(objects_of_interest),
    }

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
        operators=[no_op, pick_op, place_op, move_op, search_op],
        true_object_locations=OBJECTS_AT_LOCATIONS,
    )

    # Planning loop
    max_iterations = 60

    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "holding", "found", "searched"])
    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Goal achieved![/green]")
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(env.state, goal, max_iterations=4000, c=300, max_depth=20, heuristic_multiplier=1)

            if action_name == "NONE":
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

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

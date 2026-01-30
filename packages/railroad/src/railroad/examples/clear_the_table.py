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

from railroad.core import Fluent as F, State, get_action_by_name, ff_heuristic
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.environment import EnvironmentInterface, SimpleEnvironment


# Define locations with coordinates (for move cost calculation)
LOCATIONS = {
    "living_room": np.array([0, 0]),
    "kitchen": np.array([5, 0]),
    "table": np.array([2, 3]),  # The table location to clear
    "shelf": np.array([8, 3]),  # Destination for cleared items
}

# Define where objects actually are (ground truth)
OBJECTS_AT_LOCATIONS = {
    "living_room": {"object": set()},
    "kitchen": {"object": set()},
    "table": {"object": {"Book", "Mug", "Vase"}},  # Objects to clear
    "shelf": {"object": set()},  # Destination
}


def main() -> None:
    """Run the clear-the-table example."""
    # Initialize environment
    robot_locations = {"robot1": "living_room"}
    env = SimpleEnvironment(LOCATIONS, OBJECTS_AT_LOCATIONS, robot_locations=robot_locations)

    # Objects that need to be cleared from the table
    objects_on_table = ["Book", "Mug", "Vase"]

    # Define initial state
    initial_state = State(
        time=0,
        fluents={
            # Robot free and starts in living room
            F("free robot1"),
            F("at robot1 living_room"),
            # Objects are on the table
            F("at Book table"),
            F("at Mug table"),
            F("at Vase table"),
        },
    )

    # Define goal: NO objects on the table
    # This is a "none" goal - we want the absence of all objects at the table
    # Expressed as: forall obj in objects_on_table: NOT(at obj table)
    goal = reduce(and_, [~F(f"at {obj} table") for obj in objects_on_table])

    # Objects by type
    objects_by_type = {
        "robot": ["robot1"],
        "location": list(LOCATIONS.keys()),
        "object": objects_on_table,
    }

    # Create operators
    move_time_fn = env.get_skills_time_fn(skill_name="move")
    pick_time = env.get_skills_time_fn(skill_name="pick")
    place_time = env.get_skills_time_fn(skill_name="place")

    move_op = operators.construct_move_operator_blocking(move_time_fn)
    pick_op = operators.construct_pick_operator_blocking(pick_time)
    place_op = operators.construct_place_operator_blocking(place_time)

    # Create simulator
    sim = EnvironmentInterface(initial_state, objects_by_type, [pick_op, place_op, move_op], env)

    # Planning loop
    actions_taken = []
    max_iterations = 40

    # Dashboard
    h_value = ff_heuristic(initial_state, goal, sim.get_actions())
    with PlannerDashboard(goal, initial_heuristic=h_value) as dashboard:
        dashboard.update(sim_state=sim.state)

        for iteration in range(max_iterations):
            # Check if goal is reached
            if goal.evaluate(sim.state.fluents):
                dashboard.console.print("[green]Goal achieved! Table is clear.[/green]")
                break

            # Get available actions
            all_actions = sim.get_actions()

            # Plan next action
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(sim.state, goal, max_iterations=4000, c=300, max_depth=20)

            if action_name == "NONE":
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            # Execute action
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action, do_interrupt=False)
            actions_taken.append(action_name)

            tree_trace = mcts.get_trace_from_last_mcts_tree()
            h_value = ff_heuristic(sim.state, goal, sim.get_actions())
            relevant_fluents = {
                f for f in sim.state.fluents if any(kw in f.name for kw in ["at", "holding", "found"])
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

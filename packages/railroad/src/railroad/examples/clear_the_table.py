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

from railroad.core import Fluent as F, get_action_by_name
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.environment import EnvironmentInterfaceV2, SimpleSymbolicEnvironment


# Define locations
LOCATIONS = ["living_room", "kitchen", "table", "shelf"]

# Define where objects actually are (ground truth)
OBJECTS_AT_LOCATIONS = {
    "living_room": set(),
    "kitchen": set(),
    "table": {"Book", "Mug", "Vase"},  # Objects to clear
    "shelf": set(),  # Destination
}

# Fixed operator times for symbolic planning
MOVE_TIME = 5.0
PICK_TIME = 5.0
PLACE_TIME = 5.0


def main() -> None:
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
        "location": set(LOCATIONS),
        "object": set(objects_on_table),
    }

    # Create operators with fixed times
    move_op = operators.construct_move_operator_blocking(MOVE_TIME)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)

    # Initialize symbolic environment
    env = SimpleSymbolicEnvironment(initial_fluents, objects_by_type, OBJECTS_AT_LOCATIONS)

    # Create interface
    sim = EnvironmentInterfaceV2(env, [pick_op, place_op, move_op])

    # Planning loop
    actions_taken = []
    max_iterations = 40

    # Dashboard
    all_actions = sim.get_actions()
    mcts = MCTSPlanner(all_actions)
    h_value = mcts.heuristic(sim.state, goal)
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
            h_value = mcts.heuristic(sim.state, goal)
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

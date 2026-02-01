"""Multi-Object Search and Place Task.

This example demonstrates a complex planning scenario where multiple robots must:
1. Search for objects scattered across different locations
2. Handle probabilistic search outcomes
3. Pick up found objects and transport them to target locations
4. Coordinate using no-op (wait) actions

The environment simulates a household with multiple rooms where items need
to be reorganized.
"""

from railroad.core import Fluent as F, State, get_action_by_name, ff_heuristic
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.environment import EnvironmentInterface, SimpleOperatorEnvironment


# Define locations
LOCATIONS = ["start_loc", "living_room", "kitchen", "bedroom", "office"]

# Define where objects actually are (ground truth)
OBJECTS_AT_LOCATIONS = {
    "living_room": {"Notebook", "Pillow"},
    "kitchen": {"Clock", "Mug"},
    "bedroom": {"Knife"},
    "office": set(),
    "start_loc": set(),
}


def main() -> None:
    """Run the multi-object search example."""
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
        "location": LOCATIONS,
        "object": objects_of_interest,
    }

    # Define operators with timing baked in
    object_find_prob = lambda r, loc, o: 0.6 if "kitchen" in loc else 0.4

    move_op = operators.construct_move_operator_blocking(move_time=10.0)
    search_op = operators.construct_search_operator(object_find_prob, search_time=5.0)
    pick_op = operators.construct_pick_operator_blocking(pick_time=5.0)
    place_op = operators.construct_place_operator_blocking(place_time=5.0)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    all_operators = [no_op, move_op, search_op, pick_op, place_op]

    # Create environment with ground truth object locations
    env = SimpleOperatorEnvironment(
        operators=all_operators,
        objects_at_locations=OBJECTS_AT_LOCATIONS,
    )

    # Create simulator
    sim = EnvironmentInterface(initial_state, objects_by_type, all_operators, env)

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

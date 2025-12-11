"""
Multi-Object Search and Place Task

This script demonstrates a more complex planning scenario where a robot must:
1. Search for multiple objects scattered across different locations
2. Handle missing objects (some searches will fail)
3. Pick up found objects and transport them to target locations
4. Place objects at their designated spots

The environment simulates a household with multiple rooms where items are
disorganized and some items are missing entirely.
"""

import numpy as np
from rich import print
from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl.planner import MCTSPlanner
from mrppddl.dashboard import PlannerDashboard
import environments
from environments import Simulator, SimpleEnvironment

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
    env = SimpleEnvironment(LOCATIONS, OBJECTS_AT_LOCATIONS)

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
        },
    )

    # Define goal: all items at their proper locations
    goal_fluents = {
        F("at Remote den"),
        F("at Plate den"),
        F("at Cookie den"),
        F("at Couch den"),
    }


    # Initial objects by type (robot only knows about some objects initially)
    objects_by_type = {
        "robot": ["robot1", "robot2"],
        "location": list(LOCATIONS.keys()),
        "object": objects_of_interest,  # Robot knows these objects exist
    }

    # Create operators
    move_op = environments.actions.construct_move_operator(
        move_time=env.get_move_cost_fn()
    )

    # Search operator with 80% success rate when object is actually present
    search_op = environments.actions.construct_search_operator(
        object_find_prob=lambda r, l, o: 0.8 if o in OBJECTS_AT_LOCATIONS.get(l, dict()).get("objects", dict()) else 0.2,
        search_time=lambda r, l: 5.0
    )

    from mrppddl.core import Operator, Effect
    no_op = Operator(
        name="no-op",
        parameters=[("?r", "robot")],
        preconditions=[F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(time=100, resulting_fluents={F("free ?r")}),
        ],
    )
    pick_op = environments.actions.construct_pick_operator(
        pick_time=lambda r, l, o: 5.0
    )

    place_op = environments.actions.construct_place_operator(
        place_time=lambda r, l, o: 5.0
    )

    # Create simulator
    sim = Simulator(
        initial_state,
        objects_by_type,
        [no_op, pick_op, place_op, move_op, search_op],
        env
    )

    # Planning loop
    print("Starting planning and execution...\n")
    actions_taken = []
    max_iterations = 60  # Limit iterations to avoid infinite loops

    # Dashboard
    from rich.live import Live
    from rich.console import Console
    console = Console()
    from mrppddl._bindings import ff_heuristic
    h_value = ff_heuristic(initial_state, goal_fluents, sim.get_actions())
    dashboard = PlannerDashboard(goal_fluents, initial_heuristic=h_value)

    with Live(dashboard.renderable, refresh_per_second=100, screen=True):
        relevant_fluents = {
            f for f in sim.state.fluents
                if any(keyword in f.name for keyword in ["at", "holding", "found", "searched"])
        }

        dashboard.update(
            sim_state=sim.state,
            relevant_fluents=set(),
            step_index="---",
            last_action_name="---"
        )

        for iteration in range(max_iterations):
            # Check if goal is reached
            if sim.is_goal_reached(goal_fluents):
                console.print("\n" + "=" * 70)
                console.print("GOAL REACHED!")
                console.print("=" * 70)
                break

            # Get available actions
            all_actions = sim.get_actions()

            # Plan next action
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(sim.state, goal_fluents, max_iterations=2000, c=1000, max_depth=40)
            tree_trace = mcts.get_trace_from_last_mcts_tree()

            if action_name == 'NONE':
                console.print("No more actions available. Goal may not be achievable.")
                break

            try:
                # Execute action
                action = get_action_by_name(all_actions, action_name)
                sim.advance(action, do_interrupt=False)
                actions_taken.append(action_name)
            except ValueError as e:
                print(f"  ERROR: {e}")
                break

            h_value = ff_heuristic(sim.state, goal_fluents, sim.get_actions())
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

    # Summary
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total actions executed: {len(actions_taken)}")
    print(f"Total time: {sim.state.time:.1f} seconds")
    print()
    print("Actions taken:")
    for i, action in enumerate(actions_taken, 1):
        print(f"  {i}. {action}")
    print()

    # Check which goals were achieved
    print("Goal status:")
    for goal in goal_fluents:
        achieved = goal in sim.state.fluents
        status = "✓" if achieved else "✗"
        print(f"  {status} {goal}")
    print()

    print("=" * 70)


if __name__ == "__main__":
    main()

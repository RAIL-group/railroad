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
import environments
from environments import Simulator, SimpleEnvironment


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
            Effect(time=1000, resulting_fluents={F("free ?r")}),
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

    for iteration in range(max_iterations):
        # Check if goal is reached
        if sim.is_goal_reached(goal_fluents):
            print("\n" + "=" * 70)
            print("GOAL REACHED!")
            print("=" * 70)
            break

        # Get available actions
        all_actions = sim.get_actions()
        print(sim.state)

        # Plan next action
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(sim.state, goal_fluents, max_iterations=10000, c=100, max_depth=40)

        if action_name == 'NONE':
            print("\n" + "=" * 70)
            print("No more actions available. Goal may not be achievable.")
            print("(This could be because some objects are missing!)")
            print("=" * 70)
            break

        # Execute action
        action = get_action_by_name(all_actions, action_name)
        print(f"[Step {iteration + 1}] Executing: {action_name}")

        try:
            sim.advance(action, do_interrupt=False)
            actions_taken.append(action_name)

            # Print relevant state information
            relevant_fluents = {
                f for f in sim.state.fluents
                if any(keyword in f.name for keyword in ["at", "holding", "found", "searched"])
            }
            print(f"  Time: {sim.state.time:.1f}s")
            print(f"  Relevant state changes:")
            for f in sorted(relevant_fluents, key=lambda x: x.name):
                if "robot1" not in f.name or "at robot1" in f.name or "holding" in f.name:
                    print(f"    {f}")
            print()

        except ValueError as e:
            print(f"  ERROR: {e}")
            break

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

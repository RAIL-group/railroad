"""
Test MCTS looping behavior when objects are already found.

This test reproduces the looping behavior observed in the full simulator,
but uses only MCTS with a predefined state where objects have already been found.
"""

import numpy as np
from mrppddl.core import Fluent as F, State
from mrppddl.planner import MCTSPlanner
from environments.simulator.actions import (
    construct_move_operator,
    construct_pick_operator,
    construct_place_operator,
)
from mrppddl._bindings import ff_heuristic


def mcts_looping_with_found_objects():
    # Define locations with coordinates (for move cost calculation)
    LOCATIONS = {
        "start_loc": np.array([-5, -5]),
        "living_room": np.array([0, 0]),
        "kitchen": np.array([10, 0]),
        "bedroom": np.array([0, 12]),
        "office": np.array([10, 12]),
    }

    def get_move_time(robot, loc_from, loc_to):
        distance = np.linalg.norm(LOCATIONS[loc_from] - LOCATIONS[loc_to])
        return distance

    # Initial state from the looping scenario
    initial_state = State(
        time=0,
        fluents={
            # Robot positions
            F("free robot1"),
            F("at robot1 living_room"),
            F("at Notebook living_room"),
            F("at Pillow living_room"),
            F("at Mug kitchen"),
            F("at Clock kitchen"),
            F("at Knife living_room"),
        },
    )

    # Goal: all items at their proper locations
    goal_fluents = {
        F("at Knife kitchen"),
        F("at Mug kitchen"),
        F("at Clock bedroom"),
        F("at Pillow bedroom"),
        F("at Notebook office"),
    }

    # Define objects
    objects_by_type = {
        "robot": ["robot1"],
        "location": list(LOCATIONS.keys()),
        "object": ["Knife", "Notebook", "Clock", "Mug", "Pillow"],
    }

    # Create operators
    from mrppddl.core import Operator, Effect
    no_op = Operator(
        name="no-op",
        parameters=[("?r", "robot")],
        preconditions=[F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(time=200, resulting_fluents={F("free ?r")}),
        ],
    )
    move_op = construct_move_operator(move_time=get_move_time)
    pick_op = construct_pick_operator(pick_time=3.0)
    place_op = construct_place_operator(place_time=3.0)

    # Ground all actions
    all_actions = []
    for op in [no_op, move_op, pick_op, place_op]:
        all_actions.extend(op.instantiate(objects_by_type))

    # Create planner
    planner = MCTSPlanner(all_actions)

    # Simulate execution
    current_state = initial_state
    actions_taken = []
    max_iterations = 30

    print(f"\nRunning MCTS planner...")
    for iteration in range(max_iterations):
        h_value = ff_heuristic(current_state,
                               goal_fluents,
                               all_actions)
        print(f"\nH = {h_value}")
        # Check if goal reached
        if all(g in current_state.fluents for g in goal_fluents):
            print(f"\n✓ GOAL REACHED after {iteration} steps!")
            break

        # Plan next action
        action_name = planner(current_state, goal_fluents,
                              max_iterations=40000, max_depth=20, c=1000)

        if action_name == 'NONE':
            print("\nNo action returned by planner.")
            break

        actions_taken.append(action_name)
        print(f"\n[Step {iteration + 1}] Action selected: {action_name}")

        # Find the action and apply it
        action = next((a for a in all_actions if a.name == action_name), None)

        # Get successor state (take first outcome for deterministic execution)
        from mrppddl.core import transition
        successors = transition(current_state, action, relax=False)
        current_state = successors[0][0]

        # Print relevant state changes
        relevant_fluents = {
            f for f in current_state.fluents
            if any(keyword in f.name for keyword in ["at", "holding", "hand-full"])
            and any(obj in f.name for obj in ["Knife", "Notebook", "Clock", "Mug", "Pillow", "robot1"])
        }
        print(f"  Time: {current_state.time:.1f}s")
        print(f"  Relevant state:")
        for f in sorted(relevant_fluents, key=lambda x: str(x)):
            print(f"    {f}")

        # Detect looping: same action repeated
        if len(actions_taken) >= 4:
            last_four = actions_taken[-4:]
            if last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                print(f"\n⚠ LOOPING DETECTED!")
                print(f"  Repeating pattern: [{last_four[0]}] → [{last_four[1]}]")
                break

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total actions taken: {len(actions_taken)}")
    print(f"Actions sequence:")
    for i, action in enumerate(actions_taken, 1):
        print(f"  {i}. {action}")

    # Check for looping pattern
    pick_place_pairs = 0
    for i in range(len(actions_taken) - 1):
        if "pick" in actions_taken[i] and "place" in actions_taken[i + 1]:
            # Check if picking and placing same object at same location
            pick_parts = actions_taken[i].split()
            place_parts = actions_taken[i + 1].split()
            if len(pick_parts) >= 4 and len(place_parts) >= 4:
                if (pick_parts[2] == place_parts[2] and  # same location
                    pick_parts[3] == place_parts[3]):      # same object
                    pick_place_pairs += 1
                    print(f"\n⚠ Detected unproductive pick/place pair at step {i+1}-{i+2}:")
                    print(f"  {actions_taken[i]}")
                    print(f"  {actions_taken[i+1]}")

    if pick_place_pairs > 0:
        print(f"\n⚠ Total unproductive pick/place pairs: {pick_place_pairs}")
        print("This indicates looping behavior!")

    # Goal status
    print(f"\nGoal status:")
    for goal in goal_fluents:
        achieved = goal in current_state.fluents
        status = "✓" if achieved else "✗"
        print(f"  {status} {goal}")

    print("=" * 70)

    # Assertions for test
    assert len(actions_taken) < max_iterations, "Should not hit iteration limit"

    # The test passes if we can identify the looping behavior
    # (We're documenting the bug, not asserting it doesn't exist)
    if pick_place_pairs > 0:
        print("\n⚠ WARNING: Looping behavior confirmed in this test case!")

    return actions_taken, current_state


if __name__ == "__main__":
    mcts_looping_with_found_objects()

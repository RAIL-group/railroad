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
from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl._bindings import ff_heuristic
from mrppddl.planner import MCTSPlanner
import environments
from environments import Simulator


# Define locations with coordinates (for move cost calculation)
LOCATIONS = {
    "start_loc": np.array([-5, -5]),
    "living_room": np.array([0, 0]),
    "kitchen": np.array([10, 0]),
    "bedroom": np.array([0, 12]),
    "office": np.array([10, 12]),
}

# Define where objects actually are (ground truth)
# Note: Mug is missing - it's not at any location!
OBJECTS_AT_LOCATIONS = {
    "living_room": {"object": {"Notebook", "Pillow"}},
    "kitchen": {"object": {"Clock", "Mug"}},
    "bedroom": {"object": {"Knife"}},
    "office": {"object": set()},  # Empty
    "start_loc": {"object": set()},  # Empty
}


class HouseholdEnvironment(environments.BaseEnvironment):
    """Simple household environment for testing multi-object manipulation."""

    def __init__(self, locations, objects_at_locations):
        super().__init__()
        self.locations = locations.copy()
        self._ground_truth = objects_at_locations
        self._objects_at_locations = {loc: {"object": set()} for loc in locations}

    def get_objects_at_location(self, location):
        """Return objects at a location (simulates perception)."""
        objects_found = self._ground_truth.get(location, {}).copy()
        # Update internal knowledge
        if "object" in objects_found:
            for obj in objects_found["object"]:
                self.add_object_at_location(obj, location)
        return objects_found

    def get_move_cost_fn(self):
        """Return a function that computes movement time between locations."""
        def get_move_time(robot, loc_from, loc_to):
            distance = np.linalg.norm(
                self.locations[loc_from] - self.locations[loc_to]
            )
            return distance  # 1 unit of distance = 1 second
        return get_move_time

    def get_intermediate_coordinates(self, time, loc_from, loc_to):
        """Compute intermediate position during movement (for visualization)."""
        coord_from = self.locations[loc_from]
        coord_to = self.locations[loc_to]
        dist = np.linalg.norm(coord_to - coord_from)
        if dist < 0.01:
            return coord_to
        elif time > dist:
            return coord_to
        direction = (coord_to - coord_from) / dist
        new_coord = coord_from + direction * time
        return new_coord

    def remove_object_from_location(self, obj, location, object_type="object"):
        """Remove an object from a location (e.g., when picked up)."""
        self._objects_at_locations[location][object_type].discard(obj)
        # Also update ground truth
        if location in self._ground_truth and object_type in self._ground_truth[location]:
            self._ground_truth[location][object_type].discard(obj)

    def add_object_at_location(self, obj, location, object_type="object"):
        """Add an object to a location (e.g., when placed down)."""
        self._objects_at_locations[location][object_type].add(obj)
        # Also update ground truth
        if location not in self._ground_truth:
            self._ground_truth[location] = {}
        if object_type not in self._ground_truth[location]:
            self._ground_truth[location][object_type] = set()
        self._ground_truth[location][object_type].add(obj)


def main():
    print("=" * 70)
    print("Multi-Object Search and Place Task")
    print("=" * 70)
    print()
    print("Goal: Organize household items to their proper locations")
    print("  - Kitchen items (Knife, Mug) -> kitchen")
    print("  - Bedroom items (Clock, Pillow) -> bedroom")
    print("  - Office items (Notebook) -> office")
    print()
    print("=" * 70)
    print()

    # Initialize environment
    env = HouseholdEnvironment(LOCATIONS, OBJECTS_AT_LOCATIONS)

    # Define the objects we're looking for
    objects_of_interest = ["Knife", "Notebook", "Clock", "Mug", "Pillow"]

    # Define initial state
    initial_state = State(
        time=0,
        fluents={
            # Robot starts in living room
            F("at", "robot1", "start_loc"),
            F("at", "robot2", "start_loc"),
            F("free", "robot1"),
            F("free", "robot2"),
        },
    )

    # Define goal: all items at their proper locations
    goal_fluents = {
        F("at", "Knife", "kitchen"),
        F("at", "Mug", "kitchen"),
        F("at", "Clock", "bedroom"),
        F("at", "Pillow", "bedroom"),
        F("at", "Notebook", "office"),
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
        object_find_prob=lambda r, l, o: 0.6 if 'kitchen' in l else 0.4,
        search_time=lambda r, l: 5.0
    )

    pick_op = environments.actions.construct_pick_operator(
        pick_time=lambda r, l, o: 5.0
    )

    place_op = environments.actions.construct_place_operator(
        place_time=lambda r, l, o: 5.0
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
        extra_cost=100,
    )

    # Create simulator
    sim = Simulator(
        initial_state,
        objects_by_type,
        [no_op, move_op, search_op, pick_op, place_op],
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
        action_name = mcts(sim.state, goal_fluents, max_iterations=10000, c=400, max_depth=20)

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
            sim.advance(action)
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

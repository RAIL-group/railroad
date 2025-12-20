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

import time
import numpy as np
from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl._bindings import ff_heuristic
from mrppddl.planner import MCTSPlanner
from mrppddl.dashboard import PlannerDashboard
import environments
from environments.core import EnvironmentInterface
from environments import SimpleEnvironment
from rich.console import Console

from bench import benchmark, BenchmarkCase


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


@benchmark(
    name="multi_object_search",
    description="Find 5 objects and bring them to where they belong.",
    tags=["multi-agent", "search"],
    timeout=120.0,
)
def bench_multi_object_search(case: BenchmarkCase):
    # Define locations with coordinates (for move cost calculation)
    locations = {
        "start_loc": np.array([-5, -5]),
        "living_room": np.array([0, 0]),
        "kitchen": np.array([10, 0]),
        "bedroom": np.array([0, 12]),
        "office": np.array([10, 12]),
    }

    # Define where objects actually are (ground truth)
    objects_at_locations = {
        "living_room": {"object": {"Notebook", "Pillow"}},
        "kitchen": {"object": {"Clock", "Mug"}},
        "bedroom": {"object": {"Knife"}},
        "office": {"object": set()},  # Empty
        "start_loc": {"object": set()},  # Empty
    }

    # Initialize environment
    env = SimpleEnvironment(locations, objects_at_locations, num_robots=case.num_robots)

    # Define the objects we're looking for
    objects_of_interest = ["Knife", "Notebook", "Clock", "Mug", "Pillow"]

    # Define initial state
    initial_fluents = set()
    robot_names = []
    for ii in range(case.num_robots):
        # Free all robots and put in the living room
        robot_name = f"robot{ii+1}"
        robot_names.append(robot_name)
        initial_fluents.add(F(f"free {robot_name}"))
        initial_fluents.add(F(f"at {robot_name} start_loc"))

    initial_state = State(time=0, fluents=initial_fluents)

    # Define goal: all items at their proper locations
    goal_fluents = {
        F("at Knife kitchen"),
        F("at Mug kitchen"),
        F("at Clock bedroom"),
        F("at Pillow bedroom"),
        F("at Notebook office"),
    }

    # Initial objects by type (robot only knows about some objects initially)
    objects_by_type = {
        "robot": robot_names,
        "location": list(locations.keys()),
        "object": objects_of_interest,  # Robot knows these objects exist
    }

    # Create operators
    move_op = environments.operators.construct_move_operator(
        move_time=env.get_move_cost_fn()
    )

    # Search operator with 80% success rate when object is actually present
    search_op = environments.operators.construct_search_operator(
        object_find_prob=lambda r, l, o: 0.6 if 'kitchen' in l else 0.4,
        search_time=lambda r, l: 5.0
    )

    pick_op = environments.operators.construct_pick_operator(
        pick_time=lambda r, l, o: 5.0
    )

    place_op = environments.operators.construct_place_operator(
        place_time=lambda r, l, o: 5.0
    )

    from mrppddl.core import Operator, Effect
    no_op = Operator(
        name="no-op",
        parameters=[("?r", "robot")],
        preconditions=[F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(time=5, resulting_fluents={F("free ?r")}),
        ],
        extra_cost=10,
    )

    # Create simulator
    sim = EnvironmentInterface(
        initial_state,
        objects_by_type,
        [no_op, move_op, search_op, pick_op, place_op],
        env
    )

    # Planning loop
    actions_taken = []
    max_iterations = 60  # Limit iterations to avoid infinite loops

    # Run planning loop
    start_time = time.perf_counter()

    # Dashboard with recording console
    recording_console = Console(record=True, force_terminal=True, width=120)
    h_value = ff_heuristic(initial_state, goal_fluents, sim.get_actions())
    dashboard = PlannerDashboard(goal_fluents, initial_heuristic=h_value, console=recording_console)

    for iteration in range(max_iterations):
        # Check if goal is reached
        if sim.goal_reached(goal_fluents):
            break

        # Get available actions
        all_actions = sim.get_actions()

        # Plan next action
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(sim.state, goal_fluents,
                           max_iterations=case.mcts.iterations,
                           c=case.mcts.c,
                           max_depth=20)

        if action_name == 'NONE':
            dashboard.console.print("No more actions available. Goal may not be achievable.")
            break

        # Execute action
        action = get_action_by_name(all_actions, action_name)

        sim.advance(action, do_interrupt=False)
        actions_taken.append(action_name)

        # Print relevant state information
        tree_trace = mcts.get_trace_from_last_mcts_tree()
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

    # Export the recorded console output as HTML
    dashboard.print_history(sim.state, actions_taken)
    html_output = recording_console.export_html(inline_styles=True)

    return {
        "success": sim.goal_reached(goal_fluents),
        "wall_time": time.perf_counter() - start_time,
        "plan_cost": float(sim.state.time),
        "actions_count": len(actions_taken),
        "actions": actions_taken,
        "log_html": html_output,  # Will be logged as HTML artifact
    }


# Register parameter combinations
bench_multi_object_search.add_cases([
    {"mcts.iterations": 100, "mcts.c": 300, "num_robots": 1},
    {"mcts.iterations": 400, "mcts.c": 300, "num_robots": 1},
    {"mcts.iterations": 1000, "mcts.c": 300, "num_robots": 1},
    {"mcts.iterations": 4000, "mcts.c": 300, "num_robots": 1},
    {"mcts.iterations": 10000, "mcts.c": 300, "num_robots": 1},
    {"mcts.iterations": 400, "mcts.c": 300, "num_robots": 2},
    {"mcts.iterations": 1000, "mcts.c": 300, "num_robots": 2},
    {"mcts.iterations": 4000, "mcts.c": 300, "num_robots": 2},
    {"mcts.iterations": 10000, "mcts.c": 300, "num_robots": 2},
    {"mcts.iterations": 400, "mcts.c": 300, "num_robots": 3},
    {"mcts.iterations": 1000, "mcts.c": 300, "num_robots": 3},
    {"mcts.iterations": 4000, "mcts.c": 300, "num_robots": 3},
    {"mcts.iterations": 10000, "mcts.c": 300, "num_robots": 3},
])

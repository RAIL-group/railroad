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
import copy
from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl.planner import MCTSPlanner
from mrppddl.dashboard import PlannerDashboard
import environments
from environments.core import EnvironmentInterface as Simulator
from environments import SimpleEnvironment
from mrppddl._bindings import ff_heuristic
from rich.console import Console



# Define where objects actually are (ground truth)
# Note: Mug is missing - it's not at any location!


from bench import benchmark, BenchmarkCase
@benchmark(
    name="movie_night",
    description="N Robots find some objects to bring to the den",
    tags=["multi-agent", "search"],
    timeout=120.0,
)
def bench_movie_night(case: BenchmarkCase):
    mcts_iterations = case.params["mcts_iterations"]
    mcts_search_weight = case.params["mcts_search_weight"]
    num_robots = case.params["num_robots"]
    print(num_robots)

    # Define locations with coordinates (for move cost calculation)
    locations = {
        "living_room": np.array([0, 0]),
        "kitchen": np.array([10, 0]),
        "bedroom": np.array([0, 12]),
        "office": np.array([10, 12]),
        "den": np.array([15, 5]),
    }
    objects_at_locations = {
        "living_room": {"object": {"Remote"}},
        "kitchen": {"object": {"Cookie", "Plate"}},
        "bedroom": {"object": set()},
        "office": {"object": {"Couch"}},
        "den": {"object": set()},
    }

    # Initialize environment
    env = SimpleEnvironment(locations, objects_at_locations, num_robots=num_robots)

    # Define the objects we're looking for
    objects_of_interest = ["Remote", "Cookie", "Plate", "Couch"]

    # Define initial state
    initial_fluents = {
            # Robots free and start in (revealed) living room
            F("free robot1"),
            F("at robot1 living_room"),
            F("revealed living_room"),
            F("at Remote living_room"),
            F("found Remote"),
            F("revealed den"),
        }
    if num_robots >= 2:
        initial_fluents.add(F("free robot2"))
        initial_fluents.add(F("at robot2 living_room"))
    if num_robots >= 3:
        initial_fluents.add(F("free robot3"))
        initial_fluents.add(F("at robot3 living_room"))

    initial_state = State(
        time=0,
        fluents=initial_fluents,
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
        "robot": ["robot1", "robot2", "robot3"],
        "location": list(locations.keys()),
        "object": objects_of_interest,  # Robot knows these objects exist
    }

    # Create operators
    move_op = environments.operators.construct_move_operator(
        move_time=env.get_move_cost_fn()
    )

    # Search operator with 80% success rate when object is actually present
    object_find_prob = lambda r, loc, o: 0.8 if o in objects_at_locations.get(loc, dict()).get("object", dict()) else 0.2
    search_op = environments.operators.construct_search_operator(
        object_find_prob=object_find_prob,
        search_time=lambda r, loc: 5.0
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
        extra_cost=100,
    )
    pick_op = environments.operators.construct_pick_operator(
        pick_time=lambda r, l, o: 5.0
    )

    place_op = environments.operators.construct_place_operator(
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
    actions_taken = []
    max_iterations = 60  # Limit iterations to avoid infinite loops

    # Run planning loop
    start_time = time.perf_counter()

    # Dashboard with recording console
    recording_console = Console(record=True, force_terminal=True, width=120)
    h_value = ff_heuristic(initial_state, goal_fluents, sim.get_actions())
    dashboard = PlannerDashboard(goal_fluents, initial_heuristic=h_value, console=recording_console)
    if True:
        # (Optional) initial dashboard update
        dashboard.update(sim_state=sim.state)

        for iteration in range(max_iterations):
            # Check if goal is reached
            if sim.goal_reached(goal_fluents):
                break

            # Get available actions
            all_actions = sim.get_actions()

            # Plan next action
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(sim.state, goal_fluents, 
                               max_iterations=mcts_iterations,
                               c=mcts_search_weight,
                               max_depth=20)

            if action_name == 'NONE':
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            # Execute action
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action, do_interrupt=False)
            actions_taken.append(action_name)

            tree_trace = mcts.get_trace_from_last_mcts_tree()
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

    # Print the full dashboard history to the recording console
    dashboard.print_history(sim.state, actions_taken)

    wall_time = time.perf_counter() - start_time

    # Export the recorded console output as HTML
    html_output = recording_console.export_html(inline_styles=True)

    return {
        "success": sim.goal_reached(goal_fluents),
        "wall_time": wall_time,
        "plan_cost": float(sim.state.time),
        "actions_count": len(actions_taken),
        "actions": actions_taken,
        "log_html": html_output,  # Will be logged as HTML artifact
    }

# Register parameter combinations
bench_movie_night.add_cases([
    {"mcts_iterations": 400, "mcts_search_weight": 300, "num_robots": 1},
    {"mcts_iterations": 1000, "mcts_search_weight": 300, "num_robots": 1},
    {"mcts_iterations": 4000, "mcts_search_weight": 300, "num_robots": 1},
    {"mcts_iterations": 10000, "mcts_search_weight": 300, "num_robots": 1},
    {"mcts_iterations": 400, "mcts_search_weight": 300, "num_robots": 2},
    {"mcts_iterations": 1000, "mcts_search_weight": 300, "num_robots": 2},
    {"mcts_iterations": 4000, "mcts_search_weight": 300, "num_robots": 2},
    {"mcts_iterations": 10000, "mcts_search_weight": 300, "num_robots": 2},
    {"mcts_iterations": 400, "mcts_search_weight": 300, "num_robots": 3},
    {"mcts_iterations": 1000, "mcts_search_weight": 300, "num_robots": 3},
    {"mcts_iterations": 4000, "mcts_search_weight": 300, "num_robots": 3},
    {"mcts_iterations": 10000, "mcts_search_weight": 300, "num_robots": 3},
])

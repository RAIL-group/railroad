"""
Example benchmark: Basic single-robot planning tasks.

Demonstrates benchmark registration and parametrization.
"""

from bench import benchmark, BenchmarkCase
from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl.planner import MCTSPlanner
from environments import SimpleEnvironment
from environments.core import EnvironmentInterface
import environments.operators
import time


@benchmark(
    name="single_robot_at_location",
    description="Simple case showing that a robot can reach a destination via move.",
    repeat=1,
    tags=["simple", "navigation"],
)
def bench_single_robot_nav(case: BenchmarkCase):
    """
    Benchmark single robot navigation.

    Measures planning performance for a robot moving between locations.
    """
    # Extract parameters
    num_locations = case.num_locations

    # Create simple test environment
    import numpy as np
    locations = {f"loc{i}": np.array([i * 2.0, 0.0]) for i in range(num_locations)}
    locations["living_room"] = np.array([0.0, 0.0])  # Required for SimpleEnvironment
    locations["start"] = np.array([0.0, 0.0])

    env = SimpleEnvironment(locations=locations, objects_at_locations={})

    # Define initial state
    initial_state = State(
        time=0,
        fluents={
            F("at robot1 living_room"),
            F("free robot1"),
        },
    )

    # Define goal: visit the last location
    goal_fluents = {F(f"at robot1 loc{num_locations - 1}")}

    # Create operators
    move_time_fn = env.get_move_cost_fn()
    move_op = environments.operators.construct_move_operator(move_time_fn)

    objects_by_type = {
        "robot": ["robot1"],
        "location": list(locations.keys()),
    }

    # Create simulator
    sim = EnvironmentInterface(
        initial_state,
        objects_by_type,
        [move_op],
        env
    )

    # Run planning loop with timing
    start_time = time.perf_counter()

    actions_taken = []
    planner = MCTSPlanner(sim.get_actions())

    max_steps = 100
    for iteration in range(max_steps):
        if sim.goal_reached(goal_fluents):
            break

        action_name = planner(
            sim.state,
            goal_fluents,
            max_iterations=case.mcts.iterations,
            c=100
        )

        if action_name == 'NONE':
            break

        action = get_action_by_name(sim.get_actions(), action_name)
        sim.advance(action)
        actions_taken.append(action_name)

    wall_time = time.perf_counter() - start_time

    # Return metrics
    return {
        "success": sim.goal_reached(goal_fluents),
        "wall_time": wall_time,
        "plan_cost": float(sim.state.time),
        "actions_count": len(actions_taken),
        "actions": actions_taken,  # Will be logged as artifact
    }


# Register parameter combinations
bench_single_robot_nav.add_cases([
    {"num_locations": 3, "mcts.iterations": 100,},
])

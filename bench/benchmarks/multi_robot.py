"""
Example benchmark: Multi-robot coordination tasks.

Demonstrates multi-agent planning benchmarks.
"""

from bench import benchmark, BenchmarkCase
from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl.planner import MCTSPlanner
from environments import SimpleEnvironment
from environments.core import EnvironmentInterface
import environments.operators
import random
import time


@benchmark(
    name="two_robot_coordination",
    description="Two robots coordinating to reach separate goals",
    tags=["planning", "multi-agent", "coordination"],
    timeout=120.0,
)
def bench_two_robot_coord(case: BenchmarkCase):
    """
    Benchmark two-robot coordination.

    Two robots must navigate to different goal locations.
    """
    # Extract parameters
    num_locations = case.params["num_locations"]
    mcts_iterations = case.params["mcts_iterations"]
    seed = case.params.get("seed", case.repeat_idx)

    # Set random seed
    random.seed(seed)
    # Create environment
    import numpy as np
    locations = {f"loc{i}": np.array([i * 2.0, 0.0]) for i in range(num_locations)}
    locations["living_room"] = np.array([0.0, 0.0])  # Required for SimpleEnvironment
    locations["start"] = np.array([0.0, 0.0])

    env = SimpleEnvironment(locations=locations, objects_at_locations={}, num_robots=2)

    # Initial state: both robots at living_room
    initial_state = State(
        time=0,
        fluents={
            F("at robot1 living_room"),
            F("free robot1"),
            F("at robot2 living_room"),
            F("free robot2"),
        },
    )

    # Goals: robot1 to loc0, robot2 to last location
    goal_fluents = {
        F("at robot1 loc0"),
        F(f"at robot2 loc{num_locations - 1}"),
    }

    # Create operators
    move_time_fn = env.get_move_cost_fn()
    move_op = environments.operators.construct_move_operator(move_time_fn)

    objects_by_type = {
        "robot": ["robot1", "robot2"],
        "location": list(locations.keys()),
    }

    # Create simulator
    sim = EnvironmentInterface(
        initial_state,
        objects_by_type,
        [move_op],
        env
    )

    # Run planning loop
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
            max_iterations=mcts_iterations,
            c=100
        )

        if action_name == 'NONE':
            break

        action = get_action_by_name(sim.get_actions(), action_name)
        sim.advance(action)
        actions_taken.append(action_name)

    wall_time = time.perf_counter() - start_time

    # Count goals achieved
    goals_achieved = sum(1 for g in goal_fluents if g in sim.state.fluents)

    return {
        "success": sim.goal_reached(goal_fluents),
        "wall_time": wall_time,
        "plan_cost": float(sim.state.time),
        "actions_count": len(actions_taken),
        "planning_steps": iteration + 1,
        "goals_achieved": goals_achieved,
        "actions": actions_taken,
    }


# Register parameter combinations
bench_two_robot_coord.add_cases([
    {"num_locations": 3, "mcts_iterations": 200, "seed": 42},
    {"num_locations": 5, "mcts_iterations": 500, "seed": 42},
    {"num_locations": 8, "mcts_iterations": 1000, "seed": 42},
])

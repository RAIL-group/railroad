"""
Benchmark: OR vs AND Goals Performance

This benchmark demonstrates the efficiency gains from using OR goals with the
new efficient heuristic implementation. It compares:

1. OR goal: Visit ANY ONE of several distant locations
2. AND goal: Visit ALL of the distant locations

The OR goal should:
- Have a lower heuristic estimate (only needs to reach nearest location)
- Complete faster (only visits one location)
- Make smarter decisions (chooses nearest location)

Usage:
    python scripts/benchmark_or_vs_and_goals.py
"""

import time
import numpy as np
from mrppddl.core import Fluent as F, State, get_action_by_name, transition
from mrppddl.planner import MCTSPlanner, get_usable_actions
from mrppddl._bindings import (
    AndGoal,
    OrGoal,
    LiteralGoal,
    goal_from_fluent_set,
    ff_heuristic,
    ff_heuristic_goal,
)
from mrppddl.helper import construct_move_visited_operator


def create_grid_scenario(num_targets=4):
    """
    Create a scenario with a robot at the origin and multiple target locations
    arranged in a grid at varying distances.

    Returns:
        objects_by_type: Dict of object types
        all_actions: List of grounded actions
        initial_state: Initial state with robot at origin
        target_locations: List of target location names
    """
    # Robot starts at origin, targets are at increasing distances
    locations = ["origin"]
    for i in range(num_targets):
        locations.append(f"target_{i}")

    objects_by_type = {
        "robot": ["r1"],
        "location": locations,
    }

    # Define move costs based on distance
    # target_0 is nearest (cost 5), target_3 is farthest (cost 20)
    def move_cost_fn(robot, from_loc, to_loc):
        if to_loc == "origin":
            return 5.0
        elif to_loc.startswith("target_"):
            target_num = int(to_loc.split("_")[1])
            return 5.0 + target_num * 5.0  # Cost increases with target number
        return 10.0

    move_op = construct_move_visited_operator(move_cost_fn)
    all_actions = move_op.instantiate(objects_by_type)

    initial_state = State(
        time=0,
        fluents={
            F("at r1 origin"),
            F("free r1"),
            F("visited origin"),
        }
    )

    target_locations = [f"target_{i}" for i in range(num_targets)]

    return objects_by_type, all_actions, initial_state, target_locations


def benchmark_goal_type(goal, goal_name, all_actions, initial_state, max_steps=20):
    """
    Benchmark planning and execution with a given goal.

    Returns:
        dict with timing and execution metrics
    """
    # Measure heuristic computation time
    heuristic_start = time.perf_counter()
    h_initial = ff_heuristic_goal(initial_state, goal, all_actions)
    heuristic_time = time.perf_counter() - heuristic_start

    # Get usable actions
    usable_actions = get_usable_actions(initial_state, goal.get_all_literals(), all_actions)

    # Create planner
    mcts = MCTSPlanner(usable_actions)

    # Execute planning loop
    state = initial_state
    actions_taken = []
    total_planning_time = 0.0

    for step in range(max_steps):
        if goal.evaluate(state.fluents):
            break

        # Time the planning step
        plan_start = time.perf_counter()
        action_name = mcts(state, goal, max_iterations=500, c=10, max_depth=15)
        plan_time = time.perf_counter() - plan_start
        total_planning_time += plan_time

        if action_name == "NONE":
            break

        # Execute action
        action = get_action_by_name(usable_actions, action_name)
        result = transition(state, action)
        if not result:
            break
        state = result[0][0]
        actions_taken.append(action_name)

    return {
        "goal_name": goal_name,
        "h_initial": h_initial,
        "heuristic_time_ms": heuristic_time * 1000,
        "total_planning_time_ms": total_planning_time * 1000,
        "num_actions": len(actions_taken),
        "final_time": state.time(),
        "goal_achieved": goal.evaluate(state.fluents),
        "actions": actions_taken,
    }


def run_benchmark(num_targets=4):
    """Run the OR vs AND benchmark."""
    print("=" * 80)
    print(f"Benchmark: OR vs AND Goals ({num_targets} targets)")
    print("=" * 80)

    # Create scenario
    objects_by_type, all_actions, initial_state, target_locations = create_grid_scenario(num_targets)

    print(f"\nScenario:")
    print(f"  - Robot starts at: origin")
    print(f"  - Target locations: {', '.join(target_locations)}")
    print(f"  - Move costs: target_0=5, target_1=10, target_2=15, target_3=20")

    # Create goals
    target_literals = [LiteralGoal(F(f"visited {loc}")) for loc in target_locations]

    or_goal = OrGoal(target_literals)
    and_goal = AndGoal(target_literals)

    print(f"\n{'Goal Type':<20} {'Initial h':<12} {'Heuristic (ms)':<15} {'Planning (ms)':<15} {'Actions':<10} {'Final Time':<12} {'Success':<10}")
    print("-" * 110)

    # Benchmark OR goal
    or_results = benchmark_goal_type(or_goal, "OR (any one)", all_actions, initial_state)
    print(f"{or_results['goal_name']:<20} {or_results['h_initial']:<12.2f} "
          f"{or_results['heuristic_time_ms']:<15.3f} {or_results['total_planning_time_ms']:<15.1f} "
          f"{or_results['num_actions']:<10} {or_results['final_time']:<12.1f} "
          f"{'✓' if or_results['goal_achieved'] else '✗':<10}")

    # Benchmark AND goal
    and_results = benchmark_goal_type(and_goal, "AND (all)", all_actions, initial_state)
    print(f"{and_results['goal_name']:<20} {and_results['h_initial']:<12.2f} "
          f"{and_results['heuristic_time_ms']:<15.3f} {and_results['total_planning_time_ms']:<15.1f} "
          f"{and_results['num_actions']:<10} {and_results['final_time']:<12.1f} "
          f"{'✓' if and_results['goal_achieved'] else '✗':<10}")

    # Print insights
    print("\n" + "=" * 80)
    print("Analysis:")
    print("=" * 80)

    speedup = and_results['final_time'] / or_results['final_time']
    actions_ratio = and_results['num_actions'] / max(or_results['num_actions'], 1)
    h_ratio = and_results['h_initial'] / max(or_results['h_initial'], 0.1)

    print(f"\nEfficiency Gains with OR Goals:")
    print(f"  • Heuristic correctly estimates minimum: {or_results['h_initial']:.1f} vs {and_results['h_initial']:.1f}")
    print(f"  • Execution time: {speedup:.1f}x faster (OR: {or_results['final_time']:.1f}, AND: {and_results['final_time']:.1f})")
    print(f"  • Actions required: {actions_ratio:.1f}x fewer (OR: {or_results['num_actions']}, AND: {and_results['num_actions']})")
    print(f"  • Heuristic ratio: {h_ratio:.1f}x (reflects that AND needs all {num_targets} locations)")

    print(f"\nOR Goal Actions: {' → '.join(or_results['actions'][:5])}")
    if len(or_results['actions']) > 5:
        print(f"                 ... ({len(or_results['actions']) - 5} more)")

    print(f"\nAND Goal Actions: {' → '.join(and_results['actions'][:5])}")
    if len(and_results['actions']) > 5:
        print(f"                  ... ({len(and_results['actions']) - 5} more)")

    print("\n" + "=" * 80)
    print("Key Insight:")
    print("=" * 80)
    print("The OR goal efficiently identifies and executes the shortest path to ANY target,")
    print("while the AND goal must visit ALL targets. The efficient heuristic implementation")
    print("runs the forward phase once and evaluates each OR branch, selecting the minimum.")
    print("=" * 80)


if __name__ == "__main__":
    # Run with different numbers of targets
    for num_targets in [3, 5, 8]:
        run_benchmark(num_targets)
        print("\n" * 2)

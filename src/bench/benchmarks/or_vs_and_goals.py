"""
Benchmark: OR vs AND Goals Performance

Compares planning performance and efficiency between OR goals (visit ANY location)
and AND goals (visit ALL locations) using the efficient heuristic implementation.

Key findings:
- OR goals select the nearest/cheapest option automatically
- Heuristic correctly estimates minimum cost over branches
- Execution time and action count are significantly lower for OR goals
"""

from bench import benchmark, BenchmarkCase
from mrppddl.core import Fluent as F, State, get_action_by_name, transition
from mrppddl.planner import MCTSPlanner, get_usable_actions
from mrppddl._bindings import AndGoal, OrGoal, LiteralGoal, ff_heuristic_goal
from mrppddl.helper import construct_move_visited_operator
import time


@benchmark(
    name="or_vs_and_goal_comparison",
    description="Compare OR vs AND goals with varying number of targets",
    tags=["goals", "heuristic", "comparison"],
    repeat=5,
    timeout=60.0,
)
def bench_or_vs_and_goals(case: BenchmarkCase):
    """
    Benchmark OR vs AND goal performance.

    Scenario:
    - Robot starts at origin
    - Multiple target locations at varying distances
    - OR goal: Visit ANY ONE target
    - AND goal: Visit ALL targets

    Measures:
    - Initial heuristic estimate
    - Planning time
    - Execution time (final state time)
    - Number of actions
    """
    num_targets = case.num_targets
    goal_type = case.goal_type  # "or" or "and"

    # Create scenario with targets at increasing distances
    locations = ["origin"]
    for i in range(num_targets):
        locations.append(f"target_{i}")

    objects_by_type = {
        "robot": ["r1"],
        "location": locations,
    }

    # Define move costs: target_0 is nearest (cost 5), costs increase by 5 for each target
    def move_cost_fn(robot, from_loc, to_loc):
        if to_loc == "origin":
            return 5.0
        elif to_loc.startswith("target_"):
            target_num = int(to_loc.split("_")[1])
            return 5.0 + target_num * 5.0
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

    # Create goal based on type
    target_literals = [LiteralGoal(F(f"visited target_{i}")) for i in range(num_targets)]

    if goal_type == "or":
        goal = OrGoal(target_literals)
    elif goal_type == "and":
        goal = AndGoal(target_literals)
    else:
        raise ValueError(f"Unknown goal_type: {goal_type}")

    # Measure heuristic computation
    heuristic_start = time.perf_counter()
    h_initial = ff_heuristic_goal(initial_state, goal, all_actions)
    heuristic_time = time.perf_counter() - heuristic_start

    # Get usable actions
    usable_actions = get_usable_actions(initial_state, goal.get_all_literals(), all_actions)

    # Create planner
    mcts = MCTSPlanner(usable_actions)

    # Execute planning loop
    planning_start = time.perf_counter()
    state = initial_state
    actions_taken = []

    max_steps = 50
    for step in range(max_steps):
        if goal.evaluate(state.fluents):
            break

        action_name = mcts(
            state,
            goal,
            max_iterations=case.mcts.iterations,
            c=case.mcts.c,
            max_depth=20
        )

        if action_name == "NONE":
            break

        action = get_action_by_name(usable_actions, action_name)
        result = transition(state, action)
        if not result:
            break
        state = result[0][0]
        actions_taken.append(action_name)

    planning_time = time.perf_counter() - planning_start

    # Return metrics
    return {
        "success": goal.evaluate(state.fluents),
        "h_initial": h_initial,
        "heuristic_time_ms": heuristic_time * 1000,
        "wall_time": planning_time,
        "plan_cost": float(state.time),
        "actions_count": len(actions_taken),
        "goal_type": goal_type,
        "num_targets": num_targets,
        "actions": actions_taken,  # Will be logged as artifact
    }


# Register parameter combinations
# Compare OR vs AND with different numbers of targets
bench_or_vs_and_goals.add_cases([
    # 3 targets
    {"num_targets": 3, "goal_type": "or", "mcts.iterations": 500, "mcts.c": 10},
    {"num_targets": 3, "goal_type": "and", "mcts.iterations": 500, "mcts.c": 10},
    # 5 targets
    {"num_targets": 5, "goal_type": "or", "mcts.iterations": 500, "mcts.c": 10},
    {"num_targets": 5, "goal_type": "and", "mcts.iterations": 500, "mcts.c": 10},
    # 8 targets
    {"num_targets": 8, "goal_type": "or", "mcts.iterations": 500, "mcts.c": 10},
    {"num_targets": 8, "goal_type": "and", "mcts.iterations": 500, "mcts.c": 10},
])


@benchmark(
    name="or_branch_selection",
    description="Verify OR goal selects the nearest/cheapest branch",
    tags=["goals", "heuristic", "verification"],
    repeat=3,
    timeout=30.0,
)
def bench_or_branch_selection(case: BenchmarkCase):
    """
    Verify that OR goals correctly select the nearest/cheapest option.

    Tests heuristic accuracy by comparing OR heuristic to individual branch heuristics.
    """
    num_targets = case.num_targets

    # Create scenario
    locations = ["origin"] + [f"target_{i}" for i in range(num_targets)]

    objects_by_type = {
        "robot": ["r1"],
        "location": locations,
    }

    def move_cost_fn(robot, from_loc, to_loc):
        if to_loc == "origin":
            return 5.0
        elif to_loc.startswith("target_"):
            target_num = int(to_loc.split("_")[1])
            return 5.0 + target_num * 5.0
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

    # Create OR goal
    target_literals = [LiteralGoal(F(f"visited target_{i}")) for i in range(num_targets)]
    or_goal = OrGoal(target_literals)

    # Compute OR heuristic
    h_or = ff_heuristic_goal(initial_state, or_goal, all_actions)

    # Compute heuristic for each individual branch
    individual_heuristics = []
    for i, literal in enumerate(target_literals):
        h_branch = ff_heuristic_goal(initial_state, literal, all_actions)
        individual_heuristics.append(h_branch)

    # OR heuristic should equal the minimum branch heuristic
    h_min = min(individual_heuristics)
    heuristic_correct = abs(h_or - h_min) < 0.01

    return {
        "success": heuristic_correct,
        "h_or": h_or,
        "h_min": h_min,
        "h_max": max(individual_heuristics),
        "h_individual": individual_heuristics,
        "num_targets": num_targets,
    }


# Register parameter combinations for branch selection verification
bench_or_branch_selection.add_cases([
    {"num_targets": 3},
    {"num_targets": 5},
    {"num_targets": 10},
])

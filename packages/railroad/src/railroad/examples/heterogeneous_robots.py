"""Heterogeneous Robots Search and Deliver Task.

This example demonstrates multi-robot planning with heterogeneous robot capabilities:
1. Different robot types (rover, crawler, drone) with different abilities
2. Drone can only search (and move), while rover and crawler can also pick/place
3. Drone moves faster than ground robots
4. Robots must search locations to find supplies and deliver them to base

The heterogeneous capabilities require the planner to reason about which robot
can perform which actions, leading to interesting coordination strategies.
"""

import numpy as np

from railroad.core import Fluent as F, get_action_by_name, ff_heuristic
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.environment import SymbolicEnvironment
from railroad._bindings import State


# Define locations with coordinates (for move cost calculation)
LOCATIONS = {
    "start": np.array([0, 0]),
    "location1": np.array([10, 9]),
    "location2": np.array([9, 0]),
    "location3": np.array([1, 2]),
    "location4": np.array([1, 10]),
}

# Ground truth: where objects actually are
OBJECTS_AT_LOCATIONS = {
    "start": set(),
    "location1": set(),
    "location2": {"supplies"},
    "location3": set(),
    "location4": set(),
}

# Skills available to each robot type and their execution times
SKILLS_TIME = {
    "rover": {"pick": 10.0, "place": 10.0, "search": 10.0},
    "crawler": {"pick": 10.0, "place": 10.0, "search": 10.0},
    "drone": {"search": 10.0},  # Drone can only search (plus move)
}

# Movement speed multiplier (drone is faster)
SPEED_MULTIPLIER = {
    "rover": 1.0,
    "crawler": 1.0,
    "drone": 2.0,  # Drone moves twice as fast
}


def get_skill_time(skill_name: str):
    """Return a function that computes skill time based on robot type."""
    def skill_time_fn(robot: str, *args, **kwargs) -> float:
        robot_type = robot  # In this example, robot name == robot type
        return SKILLS_TIME.get(robot_type, {}).get(skill_name, float("inf"))
    return skill_time_fn


def get_move_time(robot: str, loc_from: str, loc_to: str) -> float:
    """Compute move time based on distance and robot speed."""
    distance = float(np.linalg.norm(LOCATIONS[loc_from] - LOCATIONS[loc_to]))
    speed = SPEED_MULTIPLIER.get(robot, 1.0)
    return distance / speed


def main() -> None:
    """Run the heterogeneous robots example."""
    # Available robots - each is a different type with different capabilities
    available_robots = ["rover", "drone", "crawler"]

    # Define initial fluents - all robots start at base
    initial_fluents = set()
    for robot in available_robots:
        initial_fluents.add(F("at", robot, "start"))
        initial_fluents.add(F("free", robot))
    initial_fluents.add(F("revealed", "start"))

    # Goal: find supplies and bring them to the start location
    goal = F("found supplies") & F("at supplies start")

    # Objects by type
    objects_by_type = {
        "robot": set(available_robots),
        "location": set(LOCATIONS.keys()),
        "object": {"supplies"},
    }

    # Search probability - higher if object is actually there
    def object_find_prob(robot: str, loc: str, obj: str) -> float:
        objects_here = OBJECTS_AT_LOCATIONS.get(loc, set())
        return 0.9 if obj in objects_here else 0.1

    # Create operators with robot-type-specific times
    move_op = operators.construct_move_operator_blocking(get_move_time)
    search_op = operators.construct_search_operator(
        object_find_prob, get_skill_time("search")
    )
    pick_op = operators.construct_pick_operator_blocking(get_skill_time("pick"))
    place_op = operators.construct_place_operator_blocking(get_skill_time("place"))
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Initialize symbolic environment
    initial_state = State(0.0, initial_fluents, [])
    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=[no_op, move_op, search_op, pick_op, place_op],
        true_object_locations=OBJECTS_AT_LOCATIONS,
    )

    # Planning loop
    actions_taken = []
    max_iterations = 60

    all_actions = env.get_actions()
    mcts = MCTSPlanner(all_actions)
    h_value = mcts.heuristic(env.state, goal)

    with PlannerDashboard(goal, initial_heuristic=h_value) as dashboard:
        dashboard.update(state=env.state)

        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Goal achieved![/green]")
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(
                env.state, goal, max_iterations=10000, c=300, max_depth=20
            )

            if action_name == "NONE":
                dashboard.console.print(
                    "No more actions available. Goal may not be achievable."
                )
                break

            action = get_action_by_name(all_actions, action_name)
            env.act(action)
            actions_taken.append(action_name)

            tree_trace = mcts.get_trace_from_last_mcts_tree()
            h_value = mcts.heuristic(env.state, goal)
            relevant_fluents = {
                f
                for f in env.state.fluents
                if any(
                    kw in f.name for kw in ["at", "holding", "found", "searched", "free"]
                )
            }
            dashboard.update(
                state=env.state,
                relevant_fluents=relevant_fluents,
                tree_trace=tree_trace,
                step_index=iteration,
                last_action_name=action_name,
                heuristic_value=h_value,
            )

    dashboard.print_history(env.state, actions_taken)


if __name__ == "__main__":
    main()

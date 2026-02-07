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

from railroad.core import Fluent as F, get_action_by_name
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.environment import SymbolicEnvironment
from railroad.environment.symbolic import (
    InterruptableMoveSymbolicSkill,
    LocationRegistry,
)
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


def make_move_time_fn(registry: LocationRegistry | None = None):
    """Create a move time function, optionally using a LocationRegistry.

    When a registry is provided, it supports dynamically-created intermediate
    locations from interrupted moves.
    """
    def get_move_time(robot: str, loc_from: str, loc_to: str) -> float:
        """Compute move time based on distance and robot speed."""
        if registry is not None:
            start = registry.get(loc_from)
            end = registry.get(loc_to)
            if start is None or end is None:
                return float("inf")
            diff = end - start
            distance = float((diff @ diff) ** 0.5)
        else:
            if loc_from not in LOCATIONS or loc_to not in LOCATIONS:
                return float("inf")
            distance = float(np.linalg.norm(LOCATIONS[loc_from] - LOCATIONS[loc_to]))
        speed = SPEED_MULTIPLIER.get(robot, 1.0)
        return distance / speed
    return get_move_time


def main(
    use_interruptible_moves: bool = False,
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
) -> None:
    """Run the heterogeneous robots example.

    Args:
        use_interruptible_moves: If True, move actions can be interrupted when
            another robot becomes free. This creates intermediate locations
            and allows for more flexible replanning.
        save_plot: Save trajectory plot to file.
        show_plot: Show trajectory plot interactively.
        save_video: Save trajectory animation to file.
    """
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

    # Set up location registry for interruptible moves (if enabled)
    location_registry: LocationRegistry | None = None
    skill_overrides: dict | None = None
    if use_interruptible_moves:
        location_registry = LocationRegistry(LOCATIONS)
        skill_overrides = {"move": InterruptableMoveSymbolicSkill}

    # Create operators with robot-type-specific times
    move_time_fn = make_move_time_fn(location_registry)
    move_op = operators.construct_move_operator_blocking(move_time_fn)
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
        skill_overrides=skill_overrides,
        location_registry=location_registry,
    )

    # Planning loop
    max_iterations = 60

    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "holding", "found", "searched", "free"])
    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
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
            dashboard.update(mcts, action_name)

    location_coords = {name: (float(c[0]), float(c[1])) for name, c in LOCATIONS.items()}
    dashboard.show_plots(
        save_plot=save_plot, show_plot=show_plot, save_video=save_video,
        location_coords=location_coords,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Heterogeneous robots example")
    parser.add_argument(
        "--interruptible-moves",
        action="store_true",
        help="Enable interruptible move actions",
    )
    parser.add_argument("--save-plot", default=None, help="Save trajectory plot to file")
    parser.add_argument("--show-plot", action="store_true", help="Show trajectory plot")
    parser.add_argument("--save-video", default=None, help="Save trajectory animation to file")
    args = parser.parse_args()
    main(
        use_interruptible_moves=args.interruptible_moves,
        save_plot=args.save_plot,
        show_plot=args.show_plot,
        save_video=args.save_video,
    )

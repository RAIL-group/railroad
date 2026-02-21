import numpy as np

from railroad.core import Fluent as F, get_action_by_name, Operator, Effect
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.environment import SymbolicEnvironment
from railroad.environment.symbolic import (
    InterruptableMoveSymbolicSkill,
    LocationRegistry,
)
from railroad._bindings import State


# Define locations with coordinates
LOCATIONS = {
    "start": np.array([0, 0]),
    "location1": np.array([10, 9]),
    "location2": np.array([9, 0]),
    "location3": np.array([1, 6]),
    "location4": np.array([1, 10]),
    # "location5": np.array([10, 1]),
    "location6": np.array([5, 5]),
    # "location7": np.array([3, 8]),
}

MONITOR_TIME = 15.0
# WAIT_TIME = 5.0

# Ground truth: where objects actually are
OBJECTS_AT_LOCATIONS = {
    "start": set(),
    "location1": set(),
    "location2": {"supplies"},
    "location3": set(),
    "location4": set(),
    # "location5": set(),
    "location6": set(),
    # "location7": set(),
}

# Skills available to each robot type and their execution times
SKILLS_TIME = {
    # "vertiwheeler": {"wait": WAIT_TIME, "pick": 1.0, "place": 1.0, "search": 2.0},
    "vertiwheeler": {"pick": 1.0, "place": 1.0, "search": 2.0},
    "drone": {"monitor": MONITOR_TIME, "search": 2.0},
}

# Movement speed multiplier (drone is faster)
SPEED_MULTIPLIER = {
    "vertiwheeler": 1.0,
    "drone": 3.0,
}


def get_skill_time(skill_name: str):
    """Return a function that computes skill time based on robot type."""
    def skill_time_fn(robot: str, *args, **kwargs) -> float:
        robot_type = robot  # In this example, robot name == robot type
        return SKILLS_TIME.get(robot_type, {}).get(skill_name, float("inf"))
    return skill_time_fn


def construct_monitor_operator(monitor_time: float):
    return Operator(
        name="monitor",
        parameters=[("?r", "robot"), ("?loc", "location")],
        preconditions=[F("at", "?r", "?loc"), F("free", "?r"), F("is_drone", "?r")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free", "?r"), F("monitoring", "?loc")}),
            Effect(
                time=monitor_time,
                resulting_fluents={F("free", "?r"), ~F("monitoring", "?loc")}
            )
        ]
    )


def construct_wait_operator(wait_time: float):
    return Operator(
        name="wait",
        parameters=[("?r", "robot"), ("?loc", "location")],
        preconditions=[F("at", "?r", "?loc"), F("free", "?r"), F("is_vertiwheeler", "?r")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free", "?r")}),
            Effect(time=wait_time, resulting_fluents={F("free", "?r")})
        ]
    )


def make_move_time_fn(registry: LocationRegistry | None = None):
    """Create a move time function, optionally using a LocationRegistry."""
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


def make_move_and_deliver_time_fn(registry: LocationRegistry | None = None):
    move_time_fn = make_move_time_fn(registry)
    deliver_time_fn = get_skill_time("place")

    def get_move_and_deliver_time(robot: str, loc_from: str, obj: str) -> float:
        loc_to = "location4"  # Delivery in location 4
        move_time = move_time_fn(robot, loc_from, loc_to)
        deliver_time = deliver_time_fn(robot)
        return move_time + deliver_time
    return get_move_and_deliver_time


def main(
    use_interruptible_moves: bool = False,
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = "./data/vertiwheeler_delivery.mp4",
    video_fps: int = 60,
    video_dpi: int = 150,
) -> None:
    """Run the vertiwheeler delivery example."""
    # Available robots
    available_robots = ["vertiwheeler", "drone"]

    # Define initial fluents
    initial_fluents = {
        F("at", "drone", "start"),
        F("at", "vertiwheeler", "start"),
        F("free", "drone"),
        F("free", "vertiwheeler"),
        F("is_drone", "drone"),
        F("is_vertiwheeler", "vertiwheeler"),
        F("revealed", "start"),
    }

    # Goal: delivery supply to location 4
    goal = (
        F("at", "supplies", "location4")
        & F("found", "supplies")
        & F("monitoring", "location4")
    )

    # Objects by type
    objects_by_type = {
        "robot": set(available_robots),
        "location": set(LOCATIONS.keys()),
        "object": {"supplies"},
    }

    # Search probability - higher if object is actually there
    def object_find_prob(robot: str, loc: str, obj: str) -> float:
        objects_here = OBJECTS_AT_LOCATIONS.get(loc, set())
        return 0.9 if loc in objects_here | {"location1", "location3"} else 0.1
        return 0.8 if obj in objects_here else 0.2
        return 0.0

    # Set up location registry for interruptible moves (if enabled)
    location_registry: LocationRegistry | None = None
    skill_overrides: dict | None = None
    if use_interruptible_moves:
        location_registry = LocationRegistry(LOCATIONS)
        skill_overrides = {"move": InterruptableMoveSymbolicSkill}

    # Create operators
    move_time_fn = make_move_time_fn(location_registry)
    # move_and_deliver_time_fn = make_move_and_deliver_time_fn(location_registry)
    monitor_op = construct_monitor_operator(MONITOR_TIME)
    # wait_op = construct_wait_operator(WAIT_TIME)
    # no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    move_drone_op = Operator(
        name="move_drone",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?r", "?from"), F("free", "?r"), F("is_drone", "?r")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free", "?r"), ~F("at", "?r", "?from")}),
            Effect(
                time=(move_time_fn, ["?r", "?from", "?to"]),
                resulting_fluents={F("free", "?r"), F("at", "?r", "?to")},
            ),
        ],
        extra_cost=0.0
    )

    # Vertiwheeler can only move to monitored locations
    move_vertiwheeler_op = Operator(
        name="move_vertiwheeler",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[
            F("at", "?r", "?from"),
            F("free", "?r"),
            F("is_vertiwheeler", "?r"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free", "?r"), ~F("at", "?r", "?from")}),
            Effect(
                time=(move_time_fn, ["?r", "?from", "?to"]),
                resulting_fluents={F("free", "?r"), F("at", "?r", "?to")},
            ),
        ],
        extra_cost=1.0
    )

    # move_and_deliver_op = construct_move_and_deliver_operator(move_and_deliver_time_fn, delivery_to_loc="location4")
    search_op = operators.construct_search_operator(
        object_find_prob, get_skill_time("search")
    )

    # Only vertiwheeler picks/places
    pick_op_base = operators.construct_pick_operator_blocking(get_skill_time("pick"))
    pick_op = Operator(
        name=pick_op_base.name,
        parameters=pick_op_base.parameters,
        preconditions=pick_op_base.preconditions + [F("is_vertiwheeler", "?r")],
        effects=pick_op_base.effects
    )

    # Vertiwheeler can only deliver (place) if location is monitored
    place_op_base = operators.construct_place_operator_blocking(get_skill_time("place"))
    place_op = Operator(
        name=place_op_base.name,
        parameters=place_op_base.parameters,
        preconditions=place_op_base.preconditions + [F("is_vertiwheeler", "?r")],
        effects=place_op_base.effects
    )

    # Initialize symbolic environment
    initial_state = State(0.0, initial_fluents, [])
    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=[move_drone_op, move_vertiwheeler_op, search_op, pick_op, monitor_op, place_op],
        # operators=[move_drone_op, move_vertiwheeler_op, search_op, pick_op, place_op],
        true_object_locations=OBJECTS_AT_LOCATIONS,
        skill_overrides=skill_overrides,
        location_registry=location_registry,
    )

    # Planning loop
    max_iterations = 200

    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "holding", "found", "searched", "free", "monitoring", "visited"])
    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Goal achieved![/green]")
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(
                env.state, goal, max_iterations=100000, c=500, max_depth=60)

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
        video_fps=video_fps, video_dpi=video_dpi,
        location_coords=location_coords,
    )


if __name__ == "__main__":
    main()

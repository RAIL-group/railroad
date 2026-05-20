import os
import random
import sys
import time
from functools import reduce
from operator import and_

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tutorial_brown_base import BenchmarkCase, LABEL_ENV_VAR, tutorial


def fluent_filter(f):
    return any(kw in f.name for kw in ["at", "found", "searched"])


def _setup_procthor(seed: int, num_objects: int):
    """Load a ProcTHOR scene and extract grid, sites, and target objects."""
    from railroad.environment.procthor import ProcTHORScene

    scene = ProcTHORScene(seed=seed)
    true_grid = scene.grid

    hidden_sites: dict[str, tuple[int, int]] = {}
    for name, loc in scene.locations.items():
        if name != "start_loc":
            hidden_sites[name] = (int(loc[0]), int(loc[1]))

    true_object_locations = scene.object_locations

    all_objects = sorted({
        obj for objs in true_object_locations.values() for obj in objs
    })
    rng = random.Random(seed)
    target_objects = rng.sample(all_objects, k=min(num_objects, len(all_objects)))

    start_loc = scene.locations.get("start_loc")
    return scene, true_grid, hidden_sites, true_object_locations, start_loc, target_objects


@tutorial(description="Frontier-based exploration and object search in ProcTHOR",
          repeat=8,
          timeout=600.0)
def tutorial_main(bcase: BenchmarkCase) -> dict:
    import numpy as np

    from railroad._bindings import State
    from railroad.core import Fluent as F, get_action_by_name
    from railroad.environment.symbolic import LocationRegistry
    from railroad.experimental.unknown_search import (
        NavigationConfig,
        Pose,
        UnknownSpaceEnvironment,
    )
    from railroad.experimental.unknown_search.operators import (
        construct_move_navigable_operator,
        construct_search_at_site_operator,
        construct_search_frontier_operator,
    )
    from railroad.navigation.constants import FREE_VAL, OBSTACLE_THRESHOLD
    from railroad.operators import construct_no_op_operator
    from railroad.planner import MCTSPlanner

    num_robots = bcase.num_robots
    num_objects = bcase.num_objects
    scene_seed = bcase.scene_seed
    allow_move_interruptions = bool(getattr(bcase, "allow_move_interruptions", False))

    if num_robots < 1:
        raise ValueError("num_robots must be >= 1")

    scene, true_grid, hidden_sites, true_object_locations, start_coord, target_objects = (
        _setup_procthor(seed=scene_seed, num_objects=num_objects)
    )

    # --- Operators ------------------------------------------------------
    # Use a ref so the move-time closure can access env after construction.
    env_ref: list[UnknownSpaceEnvironment | None] = [None]
    unreachable_move_penalty = 1_000_000.0

    def snap_to_known_free_cell(row: int, col: int) -> tuple[int, int]:
        if env_ref[0] is None:
            return row, col
        grid = env_ref[0].observed_grid
        r = max(0, min(int(row), grid.shape[0] - 1))
        c = max(0, min(int(col), grid.shape[1] - 1))
        if FREE_VAL <= float(grid[r, c]) < OBSTACLE_THRESHOLD:
            return r, c
        free_coords = np.argwhere(
            (grid >= FREE_VAL) & (grid < OBSTACLE_THRESHOLD)
        )
        if free_coords.size == 0:
            return r, c
        deltas = free_coords - np.array([r, c], dtype=int)
        nearest_idx = int(np.argmin(np.sum(deltas * deltas, axis=1)))
        nearest = free_coords[nearest_idx]
        return int(nearest[0]), int(nearest[1])

    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        if env_ref[0] is None:
            return 5.0
        move_time = env_ref[0].estimate_move_time(robot, loc_from, loc_to)
        if np.isinf(move_time):
            if F("at", robot, loc_from) in env_ref[0].fluents:
                return unreachable_move_penalty
            registry = env_ref[0].location_registry
            speed = env_ref[0].config.speed_cells_per_sec
            if registry is not None:
                c_from = registry.get(loc_from)
                c_to = registry.get(loc_to)
                if c_from is not None and c_to is not None:
                    return float(np.linalg.norm(c_to - c_from)) / max(speed, 1e-6)
            return 5.0
        return move_time

    def search_frontier_prob_fn(robot, frontier, obj):
        return 0.5

    def search_container_prob_fn(robot, location, obj):
        del robot
        return 0.85 if obj in true_object_locations.get(location, set()) else 0.15

    ops = [
        construct_move_navigable_operator(move_time_fn),
        construct_search_frontier_operator(
            object_find_prob=search_frontier_prob_fn,
            search_time=20.0,
        ),
        construct_search_at_site_operator(
            search_container_prob_fn,
            search_time=20.0,
            container_type="container",
        ),
        construct_no_op_operator(no_op_time=300.0, extra_cost=100.0),
    ]

    # --- Environment ----------------------------------------------------
    config = NavigationConfig(
        sensor_range=120.0,
        max_move_action_time=10_000.0,
        interrupt_min_new_cells=30000,
        interrupt_min_dt=30000.0,
    )

    robots = [f"robot{i + 1}" for i in range(num_robots)]
    start_name = "start_loc"

    location_registry = LocationRegistry({
        start_name: np.array(start_coord, dtype=float)
    })

    fluents: set = set()
    robot_initial_poses: dict[str, Pose] = {}
    for robot in robots:
        fluents |= {
            F(f"at {robot} {start_name}"),
            F(f"free {robot}"),
            F(f"revealed {start_name}"),
        }
        robot_initial_poses[robot] = Pose(
            float(start_coord[0]), float(start_coord[1]), 0.0
        )

    if allow_move_interruptions:
        from railroad.environment.skill import InterruptibleNavigationMoveSkill
        move_skill = InterruptibleNavigationMoveSkill
    else:
        from railroad.environment.skill import NavigationMoveSkill
        move_skill = NavigationMoveSkill

    env = UnknownSpaceEnvironment(
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": {start_name},
            "container": set(),
            "frontier": set(),
            "object": set(target_objects),
        },
        operators=ops,
        skill_overrides={'move': move_skill},
        true_grid=true_grid,
        robot_initial_poses=robot_initial_poses,
        location_registry=location_registry,
        hidden_sites=hidden_sites,
        true_object_locations=true_object_locations,
        config=config,
    )
    env.scene = scene  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    env_ref[0] = env

    def sync_known_hidden_sites() -> None:
        for site, (row, col) in hidden_sites.items():
            if env.is_cell_observed(row, col):
                env.register_discovered_location(site, snap_to_known_free_cell(row, col))
                env.objects_by_type.setdefault("container", set()).add(site)

    sync_known_hidden_sites()

    # --- Planning loop --------------------------------------------------
    goal = reduce(and_, [F(f"found {obj}") for obj in target_objects])

    location_coords = {
        name: (float(coord[0]), float(coord[1]))
        for name, coord in scene.locations.items()
    }

    start_time = time.perf_counter()
    with bcase.make_dashboard(goal, env, fluent_filter=fluent_filter,
                              location_coords=location_coords) as dashboard:
        act_callback = dashboard.make_act_callback()
        for _ in range(80):
            if goal.evaluate(env.state.fluents):
                break
            actions = env.get_actions()
            if not actions:
                break
            planner = MCTSPlanner(actions)
            action_name = planner(
                env.state, goal,
                max_iterations=bcase.mcts.iterations,
                c=bcase.mcts.c, max_depth=20,
                heuristic_multiplier=bcase.mcts.h_mult,
            )
            if action_name == "NONE":
                break
            env.act(get_action_by_name(actions, action_name), loop_callback_fn=act_callback)
            sync_known_hidden_sites()
            dashboard.update(planner, action_name)

    return {"success": goal.evaluate(env.state.fluents),
            "wall_time": time.perf_counter() - start_time,
            "plan_cost": float(env.state.time),
            "actions": [name for name, _ in dashboard.actions_taken],}


# Parameter sweep for --bench mode (single mode uses index --case, default 0).
tutorial_main.add_cases([
    {"mcts.iterations": 4000, "mcts.c": 300, "mcts.h_mult": 2,
     "scene_seed": 8612, "num_robots": 2, "num_objects": 2},
])


# Needed for the tutorial to make sure the name is static for the dashboard
if os.environ.get(LABEL_ENV_VAR):
    tutorial_main._register(os.environ[LABEL_ENV_VAR])


if __name__ == "__main__":
    tutorial_main.run_cli()

"""Frontier-based exploration and object-search example.

Demonstrates end-to-end planning: one or more robots explore unknown space by
moving to frontiers, then searching for target objects from those frontiers
with symbolic ``(at object frontier)`` assignments used for planning.

Two modes:
- **Synthetic** (default): built-in 80x80 corridor grid with hidden sites.
- **ProcTHOR** (``--procthor``): loads a ProcTHOR scene for the grid and sites.

Usage:
    uv run railroad example frontier-search
    uv run railroad example frontier-search --num-robots 2
    uv run railroad example frontier-search --procthor --seed 4001
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def main(
    procthor: bool = False,
    seed: int | None = None,
    num_objects: int = 2,
    num_robots: int = 1,
    disable_move_interruptions: bool = False,
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    video_fps: int = 60,
    video_dpi: int = 150,
) -> None:
    """Run frontier-based exploration and object search."""
    from functools import reduce
    from operator import and_

    import numpy as np

    from railroad._bindings import State
    from railroad.core import Fluent as F, get_action_by_name
    from railroad.dashboard import PlannerDashboard
    from railroad.environment.unknown_space import NavigationConfig, Pose, UnknownSpaceEnvironment
    from railroad.navigation.constants import FREE_VAL, OBSTACLE_THRESHOLD
    from railroad.environment.symbolic import LocationRegistry
    from railroad.operators import (
        construct_move_navigable_operator,
        construct_no_op_operator,
        construct_search_at_site_operator,
        construct_search_frontier_operator,
    )
    from railroad.planner import MCTSPlanner

    # ------------------------------------------------------------------
    # Setup: grid, hidden sites, target objects
    # ------------------------------------------------------------------

    if num_robots < 1:
        raise ValueError("num_robots must be >= 1")

    if procthor:
        true_grid, hidden_sites, true_object_locations, start_coords_list, target_objects = (
            _setup_procthor(seed=seed, num_objects=num_objects, num_robots=num_robots)
        )
    else:
        true_grid, hidden_sites, true_object_locations, start_coords_list, target_objects = (
            _setup_synthetic(num_robots=num_robots)
        )

    print(f"Grid: {true_grid.shape[0]}x{true_grid.shape[1]}")
    print(f"Hidden sites: {list(hidden_sites.keys())}")
    print(f"Target objects: {target_objects}")
    print(f"Starts: {start_coords_list}")

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    # Use a ref so the move-time closure can access env after construction
    env_ref: list[UnknownSpaceEnvironment | None] = [None]
    unreachable_move_penalty = 1_000_000.0

    def snap_to_known_free_cell(row: int, col: int) -> tuple[int, int]:
        """Snap a map coordinate to the nearest observed free cell."""
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
        if env_ref[0] is not None:
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
        return 5.0

    def search_frontier_prob_fn(robot: str, frontier: str, obj: str) -> float:
        return 0.5
        # del robot
        # if env_ref[0] is None or not hidden_sites:
        #     return 0.10
        # registry = env_ref[0].location_registry
        # if registry is None:
        #     return 0.10
        # frontier_xy = registry.get(frontier)
        # if frontier_xy is None:
        #     return 0.10
        # best_site = min(
        #     hidden_sites.items(),
        #     key=lambda kv: float(np.linalg.norm(frontier_xy - np.asarray(kv[1], dtype=float))),
        # )[0]
        # return 0.85 if obj in true_object_locations.get(best_site, set()) else 0.25

    def search_container_prob_fn(robot: str, location: str, obj: str) -> float:
        del robot
        return 0.85 if obj in true_object_locations.get(location, set()) else 0.15

    operators = [
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

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    config = NavigationConfig(
        sensor_range=120.0,
        sensor_fov_rad=2 * np.pi,
        sensor_num_rays=361,
        move_execution_use_theta_star=True,
        move_execution_interruptible=not disable_move_interruptions,
        trajectory_use_soft_cost=False,
        trajectory_soft_cost_scale=12.0,
        max_move_action_time=10_000.0,
        interrupt_min_new_cells=30000,
        interrupt_min_dt=30000.0,
    )

    robots = [f"robot{i + 1}" for i in range(num_robots)]
    start_names = ["start" if i == 0 else f"start{i + 1}" for i in range(num_robots)]

    location_registry = LocationRegistry(
        {
            start_name: np.array(start_coords_list[i], dtype=float)
            for i, start_name in enumerate(start_names)
        }
    )

    fluents: set = set()
    robot_initial_poses: dict[str, Pose] = {}
    for i, robot in enumerate(robots):
        start_name = start_names[i]
        start_coords = start_coords_list[i]
        fluents |= {
            F(f"at {robot} {start_name}"),
            F(f"free {robot}"),
            F(f"revealed {start_name}"),
        }
        robot_initial_poses[robot] = Pose(
            float(start_coords[0]), float(start_coords[1]), 0.0
        )

    env = UnknownSpaceEnvironment(
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": set(start_names),
            "container": set(),
            "frontier": set(),
            "object": set(target_objects),
        },
        operators=operators,
        true_grid=true_grid,
        robot_initial_poses=robot_initial_poses,
        location_registry=location_registry,
        hidden_sites=hidden_sites,
        true_object_locations=true_object_locations,
        config=config,
    )
    env_ref[0] = env

    def sync_known_hidden_sites() -> None:
        """Expose hidden containers only after their map cells are observed."""
        for site, (row, col) in hidden_sites.items():
            if env.is_cell_observed(row, col):
                env.register_discovered_location(site, snap_to_known_free_cell(row, col))
                env.objects_by_type.setdefault("container", set()).add(site)

    sync_known_hidden_sites()

    # ------------------------------------------------------------------
    # Planning loop
    # ------------------------------------------------------------------

    goal = reduce(and_, [F(f"found {obj}") for obj in target_objects])

    def fluent_filter(f):  # noqa: ANN001
        return any(kw in f.name for kw in ["at", "found", "searched"])

    max_iterations = 80

    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        act_callback = dashboard.make_act_callback()
        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]All objects found![/green]")
                break

            actions = env.get_actions()
            if not actions:
                dashboard.console.print("[red]No actions available — stuck.[/red]")
                break

            mcts = MCTSPlanner(actions)
            action_name = mcts(
                env.state,
                goal,
                max_iterations=4000,
                c=300,
                max_depth=20,
                heuristic_multiplier=2,
            )

            if action_name == "NONE":
                dashboard.console.print("[yellow]Planner returned NONE — stopping.[/yellow]")
                break

            action = get_action_by_name(actions, action_name)
            env.act(action, loop_callback_fn=act_callback)
            sync_known_hidden_sites()
            dashboard.update(mcts, action_name)

    dashboard.show_plots(
        save_plot=save_plot,
        show_plot=show_plot,
        save_video=save_video,
        video_fps=video_fps,
        video_dpi=video_dpi,
    )


# ======================================================================
# Setup helpers
# ======================================================================


def _setup_synthetic(num_robots: int = 1) -> tuple[
    "np.ndarray",
    dict[str, tuple[int, int]],
    dict[str, set[str]],
    list[tuple[int, int]],
    list[str],
]:
    """Build a synthetic corridor grid with hidden container sites."""
    import numpy as np

    from railroad.navigation.constants import COLLISION_VAL, FREE_VAL

    size = 80
    mid = size // 2

    grid = np.full((size, size), COLLISION_VAL)

    # Central hub (16x16)
    hub_half = 8
    grid[mid - hub_half : mid + hub_half, mid - hub_half : mid + hub_half] = FREE_VAL

    # Four corridors
    cor_half = 3
    grid[1 : mid - hub_half + 1, mid - cor_half : mid + cor_half + 1] = FREE_VAL  # North
    grid[mid + hub_half - 1 : size - 1, mid - cor_half : mid + cor_half + 1] = FREE_VAL  # South
    grid[mid - cor_half : mid + cor_half + 1, 1 : mid - hub_half + 1] = FREE_VAL  # West
    grid[mid - cor_half : mid + cor_half + 1, mid + hub_half - 1 : size - 1] = FREE_VAL  # East

    # Side rooms off each corridor
    grid[4:12, mid - cor_half - 8 : mid - cor_half] = FREE_VAL
    grid[4:12, mid + cor_half + 1 : mid + cor_half + 9] = FREE_VAL
    grid[size - 12 : size - 4, mid - cor_half - 8 : mid - cor_half] = FREE_VAL
    grid[size - 12 : size - 4, mid + cor_half + 1 : mid + cor_half + 9] = FREE_VAL
    grid[mid - cor_half - 8 : mid - cor_half, 4:12] = FREE_VAL
    grid[mid + cor_half + 1 : mid + cor_half + 9, 4:12] = FREE_VAL
    grid[mid - cor_half - 8 : mid - cor_half, size - 12 : size - 4] = FREE_VAL
    grid[mid + cor_half + 1 : mid + cor_half + 9, size - 12 : size - 4] = FREE_VAL

    # Small obstacles in the hub
    for dr, dc in [(-3, -3), (-3, 3), (3, -3), (3, 3)]:
        r, c = mid + dr, mid + dc
        grid[r, c] = COLLISION_VAL
        grid[r - 1, c] = COLLISION_VAL
        grid[r, c - 1] = COLLISION_VAL

    hidden_sites: dict[str, tuple[int, int]] = {
        "container_north": (8, mid),
        "container_south": (size - 8, mid),
        "container_east": (mid, size - 8),
    }

    true_object_locations: dict[str, set[str]] = {
        "container_north": {"Mug"},
        "container_east": {"Knife"},
    }

    target_objects = ["Mug", "Knife"]
    start_coords = [(mid, mid)] * num_robots

    return grid, hidden_sites, true_object_locations, start_coords, target_objects


def _setup_procthor(
    seed: int | None = None,
    num_objects: int = 2,
    num_robots: int = 1,
) -> tuple[
    "np.ndarray",
    dict[str, tuple[int, int]],
    dict[str, set[str]],
    list[tuple[int, int]],
    list[str],
]:
    """Load a ProcTHOR scene and extract grid, sites, and objects."""
    import random

    try:
        from railroad.environment.procthor import ProcTHORScene
    except ImportError as e:
        raise ImportError(
            "ProcTHOR dependencies not installed. "
            "Install with: pip install railroad[procthor]"
        ) from e

    scene_seed = seed if seed is not None else 4001
    print(f"Loading ProcTHOR scene (seed={scene_seed})...")
    scene = ProcTHORScene(seed=scene_seed)

    true_grid = scene.grid

    # All locations except start_loc become hidden sites
    hidden_sites: dict[str, tuple[int, int]] = {}
    for name, loc in scene.locations.items():
        if name != "start_loc":
            hidden_sites[name] = (int(loc[0]), int(loc[1]))

    true_object_locations = scene.object_locations

    # Select target objects
    all_objects = sorted({
        obj for objs in true_object_locations.values() for obj in objs
    })
    if seed is not None:
        random.seed(seed)
    target_objects = random.sample(all_objects, k=min(num_objects, len(all_objects)))

    # Shared start position for all robots
    start_loc = scene.locations.get("start_loc")
    if start_loc is not None:
        shared_start = (int(start_loc[0]), int(start_loc[1]))
    else:
        # Fallback to grid center
        shared_start = (true_grid.shape[0] // 2, true_grid.shape[1] // 2)

    return (
        true_grid,
        hidden_sites,
        true_object_locations,
        [shared_start] * num_robots,
        target_objects,
    )


if __name__ == "__main__":
    main()

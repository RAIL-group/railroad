"""Frontier-based exploration and object search example.

Demonstrates end-to-end planning: a single robot explores unknown space by
moving to frontiers, discovers hidden container sites as map cells are
revealed, and searches those sites for target objects using the MCTS planner.

Two modes:
- **Synthetic** (default): built-in 80x80 corridor grid with hidden sites.
- **ProcTHOR** (``--procthor``): loads a ProcTHOR scene for the grid and sites.

Usage:
    uv run railroad example frontier-search
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
    from railroad.environment.navigation import NavigationConfig, Pose, UnknownSpaceEnvironment
    from railroad.environment.symbolic import LocationRegistry
    from railroad.operators import (
        construct_move_navigable_operator,
        construct_observe_site_operator,
        construct_search_at_site_operator,
        construct_wait_operator,
    )
    from railroad.planner import MCTSPlanner

    # ------------------------------------------------------------------
    # Setup: grid, hidden sites, target objects
    # ------------------------------------------------------------------

    if procthor:
        true_grid, hidden_sites, true_object_locations, start_coords, target_objects = (
            _setup_procthor(seed=seed, num_objects=num_objects)
        )
    else:
        true_grid, hidden_sites, true_object_locations, start_coords, target_objects = (
            _setup_synthetic()
        )

    print(f"Grid: {true_grid.shape[0]}x{true_grid.shape[1]}")
    print(f"Hidden sites: {list(hidden_sites.keys())}")
    print(f"Target objects: {target_objects}")
    print(f"Start: {start_coords}")

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    # Use a ref so the move-time closure can access env after construction
    env_ref: list[UnknownSpaceEnvironment | None] = [None]

    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        if env_ref[0] is not None:
            return env_ref[0].estimate_move_time(robot, loc_from, loc_to)
        return 5.0

    def object_find_prob_fn(robot: str, location: str, obj: str) -> float:
        for loc, objs in true_object_locations.items():
            if obj in objs:
                return 0.8 if loc == location else 0.1
        return 0.1

    operators = [
        construct_move_navigable_operator(move_time_fn),
        construct_observe_site_operator(observe_success_prob=0.8, observe_time=1.0, container_type="container"),
        construct_search_at_site_operator(object_find_prob_fn, search_time=2.0, container_type="container"),
        construct_wait_operator(),
    ]

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    config = NavigationConfig(
        interrupt_min_new_cells=100000,   # effectively disable interrupt
        interrupt_min_dt=100000.0,
    )

    start_loc = np.array(start_coords, dtype=float)
    location_registry = LocationRegistry(
        {"start": start_loc}
        | {k: np.array(v, dtype=float) for k, v in hidden_sites.items()}
    )

    fluents: set = {
        F("at robot1 start"),
        F("free robot1"),
        F("revealed start"),
    }

    env = UnknownSpaceEnvironment(
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": {"robot1"},
            "location": {"start"} | set(hidden_sites.keys()),
            "container": set(hidden_sites.keys()),
            "frontier": set(),
            "object": set(target_objects),
        },
        operators=operators,
        true_grid=true_grid,
        robot_initial_poses={"robot1": Pose(float(start_coords[0]), float(start_coords[1]), 0.0)},
        location_registry=location_registry,
        hidden_sites=hidden_sites,
        true_object_locations=true_object_locations,
        config=config,
    )
    env_ref[0] = env

    # ------------------------------------------------------------------
    # Planning loop
    # ------------------------------------------------------------------

    goal = reduce(and_, [F(f"found {obj}") for obj in target_objects])

    def fluent_filter(f):  # noqa: ANN001
        return any(kw in f.name for kw in ["at", "found", "searched", "navigable", "candidate"])

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
            )

            if action_name == "NONE":
                dashboard.console.print("[yellow]Planner returned NONE — stopping.[/yellow]")
                break

            action = get_action_by_name(actions, action_name)
            env.act(action, loop_callback_fn=act_callback)
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


def _setup_synthetic() -> tuple[
    "np.ndarray",
    dict[str, tuple[int, int]],
    dict[str, set[str]],
    tuple[int, int],
    list[str],
]:
    """Build a synthetic corridor grid with hidden container sites."""
    import numpy as np

    from railroad.environment.navigation.constants import COLLISION_VAL, FREE_VAL

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
    start_coords = (mid, mid)

    return grid, hidden_sites, true_object_locations, start_coords, target_objects


def _setup_procthor(
    seed: int | None = None, num_objects: int = 2
) -> tuple[
    "np.ndarray",
    dict[str, tuple[int, int]],
    dict[str, set[str]],
    tuple[int, int],
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

    # Start position
    start_loc = scene.locations.get("start_loc")
    if start_loc is not None:
        start_coords = (int(start_loc[0]), int(start_loc[1]))
    else:
        # Fallback to grid center
        start_coords = (true_grid.shape[0] // 2, true_grid.shape[1] // 2)

    return true_grid, hidden_sites, true_object_locations, start_coords, target_objects


if __name__ == "__main__":
    main()

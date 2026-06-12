"""Point-goal navigation under uncertainty with LSP training-data generation.

A robot must reach the scene's goal location through unknown space. The
goal is treated like an object whose spatial location is known in
advance: exploring a frontier may *reveal* it (with probability and
durations from a property provider), and the robot then drives to it by
following a real path on the observed map. The ``oracle`` provider
computes frontier properties from the true map (per frontier, mask all
other frontiers and check whether a path through it reaches the goal),
while the ``optimistic`` provider uses fixed constants. Execution always
resolves explore outcomes from the true map.

As the robot explores, every frontier is labeled against the true map
and, whenever a frontier's label or best panoramic vantage changes, a
training datum is written: the panorama rotated to look at the frontier,
egocentric frontier/goal locations, the feasibility label, and the
success or exploration costs.

Rendering needs a working OpenGL context (CGL on macOS, GLX/EGL on
Linux). Set ``RAILSIM_GL_BACKEND`` to pin a backend.

Usage:
    uv run railroad example lsp-point-goal-nav
    uv run railroad example lsp-point-goal-nav --env office --provider optimistic
    uv run railroad example lsp-point-goal-nav --save-data-dir data/lsp
"""

from __future__ import annotations


def main(
    env_name: str = "maze",
    seed: int | None = None,
    provider_name: str = "oracle",
    optimistic_prob: float = 0.8,
    save_data_dir: str | None = None,
    allow_move_interruptions: bool = False,
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    video_fps: int = 60,
    video_dpi: int = 150,
) -> None:
    """Run point-goal navigation with LSP frontier actions."""
    import numpy as np

    from railroad._bindings import State
    from railroad.core import Fluent as F, get_action_by_name
    from railroad.dashboard import PlannerDashboard
    from railroad.environment.symbolic import LocationRegistry
    from railroad.experimental.unknown_search import (
        NavigationConfig,
        Pose,
    )
    from railroad.experimental.unknown_search.operators import (
        construct_move_navigable_operator,
    )
    from railroad.lsp import (
        OptimisticFrontierPropertyProvider,
        OracleFrontierPropertyProvider,
        TrainingDataWriter,
        construct_lsp_explore_operator,
        construct_move_to_goal_operator,
    )
    from railroad.operators import construct_no_op_operator
    from railroad.planner import MCTSPlanner

    try:
        from railroad.environment.railsim import RailsimScene
        from railroad.lsp.environment import LSPVisualEnvironment
    except ImportError as e:
        raise ImportError(
            "railsim dependencies not installed. "
            "Install with: pip install railroad[railsim]"
        ) from e

    # ------------------------------------------------------------------
    # Setup: scene (grid + visual simulator)
    # ------------------------------------------------------------------

    if env_name not in ("maze", "office"):
        raise ValueError(f"Unknown --env {env_name!r}; expected 'maze' or 'office'")
    if provider_name not in ("oracle", "optimistic"):
        raise ValueError(
            f"Unknown --provider {provider_name!r}; expected 'oracle' or 'optimistic'"
        )

    scene_seed = seed if seed is not None else 2024
    print(f"Generating {env_name} scene (seed={scene_seed})...")
    if env_name == "maze":
        scene = RailsimScene.maze(seed=scene_seed)
    else:
        from railroad.environment.railsim import OfficeConfig

        scene = RailsimScene.office(
            seed=scene_seed,
            config=OfficeConfig(grid_size=(300, 200), num_hallways=4),
        )

    start_coord = scene.locations["start_loc"]
    goal_coord = scene.locations["goal_loc"]
    print(f"Grid: {scene.grid.shape[0]}x{scene.grid.shape[1]} "
          f"({scene.resolution} m/cell)")
    print(f"Start: {start_coord}  Goal: {goal_coord}")

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    config = NavigationConfig(
        sensor_range=60.0,
        max_move_action_time=10_000.0,
        interrupt_min_new_cells=30000,
        interrupt_min_dt=30000.0,
    )

    # Both the move-time function and the oracle provider need the env,
    # which doesn't exist yet; defer the binding through env_ref.
    env_ref: list[LSPVisualEnvironment | None] = [None]

    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        if env_ref[0] is None:
            return 5.0
        return env_ref[0].estimate_move_time_safe(robot, loc_from, loc_to)

    if provider_name == "oracle":
        provider = OracleFrontierPropertyProvider(
            lambda: env_ref[0].oracle_labels if env_ref[0] is not None else {}
        )
    else:
        provider = OptimisticFrontierPropertyProvider(
            prob_feasible=optimistic_prob,
            delta_success_cost=0.0,
            exploration_cost=10.0,
        )

    operators = [
        construct_move_navigable_operator(move_time_fn),
        construct_move_to_goal_operator(move_time_fn),
        construct_lsp_explore_operator(
            provider, speed_cells_per_sec=config.speed_cells_per_sec
        ),
        construct_no_op_operator(no_op_time=300.0, extra_cost=100.0),
    ]

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    robot = "robot1"
    start_name = "start_loc"

    # The goal's coordinates are known in advance (the environment
    # registers them), but it only becomes a movable destination once
    # revealed — by observing its cell, or symbolically through the
    # lsp-explore success branch during planning rollouts.
    location_registry = LocationRegistry({
        start_name: np.array(start_coord, dtype=float)
    })

    fluents = {
        F(f"at {robot} {start_name}"),
        F(f"free {robot}"),
        F(f"revealed {start_name}"),
    }
    robot_initial_poses = {
        robot: Pose(float(start_coord[0]), float(start_coord[1]), 0.0)
    }

    if allow_move_interruptions:
        from railroad.environment.skill import InterruptibleNavigationMoveSkill
        move_skill = InterruptibleNavigationMoveSkill
    else:
        from railroad.environment.skill import NavigationMoveSkill
        move_skill = NavigationMoveSkill

    data_writer = None
    if save_data_dir is not None:
        data_writer = TrainingDataWriter(
            save_data_dir,
            run_metadata={
                "env": env_name,
                "seed": scene_seed,
                "provider": provider_name,
                "goal_cell": [int(goal_coord[0]), int(goal_coord[1])],
            },
        )

    env = LSPVisualEnvironment(
        scene=scene,
        goal_cell=(int(goal_coord[0]), int(goal_coord[1])),
        data_writer=data_writer,
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": {robot},
            "location": {start_name},
            "frontier": set(),
            "object": set(),
            "goal": set(),  # populated by the environment with the goal name
        },
        operators=operators,
        skill_overrides={'move': move_skill},
        robot_initial_poses=robot_initial_poses,
        location_registry=location_registry,
        config=config,
    )
    env_ref[0] = env

    # ------------------------------------------------------------------
    # Planning loop
    # ------------------------------------------------------------------

    goal = F(f"at {robot} goal")
    max_iterations = 200

    def fluent_filter(f):  # noqa: ANN001
        return any(kw in f.name for kw in ["at", "explored", "revealed"])

    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        act_callback = dashboard.make_act_callback()
        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Goal reached![/green]")
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
            dashboard.update(mcts, action_name)

        dashboard.console.print(
            f"Finished at t={env.state.time:.1f}s with "
            f"{len(env.pano_records)} panoramas and "
            f"{env.num_data_written} training data written."
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    if data_writer is not None:
        data_writer.close()
        print(f"Wrote {data_writer.num_written} training data to {save_data_dir}")

    dashboard.show_plots(
        save_plot=save_plot,
        show_plot=show_plot,
        save_video=save_video,
        video_fps=video_fps,
        video_dpi=video_dpi,
    )

    scene.release()


if __name__ == "__main__":
    main()

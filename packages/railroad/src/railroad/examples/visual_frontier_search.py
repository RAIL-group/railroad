"""Frontier exploration with visual (panoramic) sensing in railsim worlds.

One or more robots explore an unknown procedurally generated maze or office
until no reachable frontiers remain. As each robot moves, the environment
senses at a fixed cadence and renders a panoramic image at the robot's pose;
the collected panoramas are the visual record of what the robot saw while
revealing space and can optionally be written to disk.

Rendering needs a working OpenGL context (CGL on macOS, GLX/EGL on Linux).
Set ``RAILSIM_GL_BACKEND`` (``cgl``, ``egl``, ``glx``, or ``cpu``) to pin a
backend, e.g. ``cpu`` for Mesa software rendering on headless Linux.

Usage:
    uv run railroad example visual-frontier-search
    uv run railroad example visual-frontier-search --env office --seed 7
    uv run railroad example visual-frontier-search --save-pano-dir data/panos
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from railroad.environment.railsim import PanoRecord


def _plot_onboard_images(
    records: "list[PanoRecord]",
    save_path: str | None = None,
    show: bool = False,
    max_images: int = 8,
) -> None:
    """Plot a time-sampled grid of panoramas captured onboard the robot(s)."""
    import math

    import matplotlib.pyplot as plt
    import numpy as np

    if not records or (save_path is None and not show):
        return

    indices = sorted(set(
        np.linspace(0, len(records) - 1, min(max_images, len(records)))
        .round().astype(int)
    ))
    sampled = [records[i] for i in indices]

    ncols = 2 if len(sampled) > 1 else 1
    nrows = math.ceil(len(sampled) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 1.9 * nrows), squeeze=False
    )
    fig.suptitle("Onboard panoramas during exploration", fontsize=11)
    for ax in axes.flat:
        ax.axis("off")
    for ax, rec in zip(axes.flat, sampled):
        ax.imshow(rec.image)
        ax.set_title(f"{rec.robot}  t={rec.time:.1f}s", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved onboard image plot to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(
    env_name: str = "maze",
    seed: int | None = None,
    num_robots: int = 1,
    save_pano_dir: str | None = None,
    allow_move_interruptions: bool = False,
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    video_fps: int = 60,
    video_dpi: int = 150,
) -> None:
    """Run frontier exploration with panoramic image collection."""
    from functools import reduce
    from operator import and_

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
        construct_explore_frontier_operator,
        construct_move_navigable_operator,
    )
    from railroad.operators import construct_no_op_operator
    from railroad.planner import MCTSPlanner

    try:
        from railroad.environment.railsim import (
            RailsimScene,
            VisualUnknownSpaceEnvironment,
        )
    except ImportError as e:
        raise ImportError(
            "railsim dependencies not installed. "
            "Install with: pip install railroad[railsim]"
        ) from e

    # ------------------------------------------------------------------
    # Setup: scene (grid + visual simulator)
    # ------------------------------------------------------------------

    if num_robots < 1:
        raise ValueError("num_robots must be >= 1")
    if env_name not in ("maze", "office"):
        raise ValueError(f"Unknown --env {env_name!r}; expected 'maze' or 'office'")

    scene_seed = seed if seed is not None else 2024
    print(f"Generating {env_name} scene (seed={scene_seed})...")
    if env_name == "maze":
        scene = RailsimScene.maze(seed=scene_seed)
    else:
        # Smaller than the OfficeConfig default (500x300) so the example
        # finishes in a few minutes; pass a custom RailsimScene for full size.
        from railroad.environment.railsim import OfficeConfig

        scene = RailsimScene.office(
            seed=scene_seed,
            config=OfficeConfig(grid_size=(300, 200), num_hallways=4),
        )

    start_coord = scene.locations["start_loc"]
    print(f"Grid: {scene.grid.shape[0]}x{scene.grid.shape[1]} "
          f"({scene.resolution} m/cell)")
    print(f"Start: {start_coord}")

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    # The move operator's time function needs the env, which doesn't exist
    # yet. Defer the binding through env_ref and use the env's safe
    # estimator (Euclidean fallback for unreachable hypotheticals).
    env_ref: list[VisualUnknownSpaceEnvironment | None] = [None]

    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        if env_ref[0] is None:
            return 5.0
        return env_ref[0].estimate_move_time_safe(robot, loc_from, loc_to)

    operators = [
        construct_move_navigable_operator(move_time_fn),
        construct_explore_frontier_operator(explore_time=10.0, completion_prob=0.25),
        construct_no_op_operator(no_op_time=300.0, extra_cost=100.0),
    ]

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    config = NavigationConfig(
        sensor_range=60.0,
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

    env = VisualUnknownSpaceEnvironment(
        scene=scene,
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": {start_name},
            "frontier": set(),
            "object": set(),
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

    # Termination is judged against the ground-truth ``exploration-complete``
    # fluent, which the environment maintains from the true frontier set. The
    # *planning* goal is recomputed each iteration as "explored every
    # currently known frontier": planning directly for exploration-complete
    # makes the 'failed explore' branch a symbolic dead end (frontiers can't
    # be re-explored), which biases MCTS toward no_op.
    goal = F("exploration-complete")

    def make_planning_goal():  # noqa: ANN202
        frontier_ids = sorted(env.objects_by_type.get("frontier", set()))
        if not frontier_ids:
            return goal
        return reduce(and_, [F(f"explored {f}") for f in frontier_ids])

    def all_frontiers_explored() -> bool:
        """True when every remaining frontier has already been explored.

        A frontier can persist after the robot visits and explores it when
        the unknown space behind it is occluded from all reachable poses
        (e.g. behind inflated clutter); exploration then cannot progress
        further and the run should stop.
        """
        frontier_ids = set(env.objects_by_type.get("frontier", set()))
        explored = {
            f.args[0] for f in env.state.fluents
            if f.name == "explored" and f.args
        }
        return bool(frontier_ids) and frontier_ids <= explored

    def fluent_filter(f):  # noqa: ANN001
        return any(kw in f.name for kw in ["at", "explored", "exploration-complete"])

    max_iterations = 200

    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        act_callback = dashboard.make_act_callback()
        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Exploration complete![/green]")
                break
            if all_frontiers_explored():
                dashboard.console.print(
                    "[yellow]All reachable frontiers explored — remaining "
                    "unknown space is occluded. Stopping.[/yellow]"
                )
                break

            actions = env.get_actions()
            if not actions:
                dashboard.console.print("[red]No actions available — stuck.[/red]")
                break

            mcts = MCTSPlanner(actions)
            action_name = mcts(
                env.state,
                make_planning_goal(),
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
            f"Collected {len(env.pano_records)} panoramic images "
            f"across {env.state.time:.1f}s of exploration."
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    if save_pano_dir is not None:
        import os

        import matplotlib.pyplot as plt

        os.makedirs(save_pano_dir, exist_ok=True)
        for i, rec in enumerate(env.pano_records):
            filename = f"pano_{i:04d}_{rec.robot}_t{rec.time:07.2f}.png"
            plt.imsave(os.path.join(save_pano_dir, filename), rec.image)
        print(f"Saved {len(env.pano_records)} panos to {save_pano_dir}")

    dashboard.show_plots(
        save_plot=save_plot,
        show_plot=show_plot,
        save_video=save_video,
        video_fps=video_fps,
        video_dpi=video_dpi,
    )

    pano_plot_path = None
    if save_plot is not None:
        import os

        stem, ext = os.path.splitext(save_plot)
        pano_plot_path = f"{stem}_panos{ext or '.png'}"
    _plot_onboard_images(env.pano_records, save_path=pano_plot_path, show=show_plot)

    scene.release()


if __name__ == "__main__":
    main()

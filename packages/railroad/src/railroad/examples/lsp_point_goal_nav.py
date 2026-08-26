"""Point-goal navigation under uncertainty with LSP training-data generation.

A robot must reach the scene's goal location through unknown space. The
goal is treated like an object whose spatial location is known in
advance: exploring a frontier may *reveal* it, and the robot then drives
to it by following a real path on the observed map. With ``--num-robots``
> 1, all robots start co-located and the goal is reached as soon as any
one of them arrives.

How promising each frontier looks to the planner is set by a
frontier-statistics estimator (``--frontier-statistics``):

- ``oracle``: exact statistics from the true map (per frontier, mask all
  other frontiers and check whether a path through it reaches the goal),
- ``fixed-prior``: the same fixed constants for every frontier — no
  oracle needed, as in a deployment,
- ``learned``: an LSPFrontierNet (trained with ``railroad lsp
  train-network``) predicts the statistics from each frontier's best
  panoramic vantage + egocentric frontier/goal locations — exactly what
  the training data stores. Pass the weights via ``--network-file``.

Regardless of estimator, *execution* always resolves explore outcomes
from the true map.

As the robot explores, every frontier is labeled against the true map
and, whenever a frontier's label or best panoramic vantage changes, a
training datum is written (``--save-data-dir``); inspect the result with
``railroad lsp inspect-data``.

Rendering needs a working OpenGL context (CGL on macOS, GLX/EGL on
Linux). Set ``RAILSIM_GL_BACKEND`` to pin a backend.

Usage:
    uv run railroad example lsp-point-goal-nav
    uv run railroad example lsp-point-goal-nav --env office --frontier-statistics fixed-prior
    uv run railroad example lsp-point-goal-nav --save-data-dir data/lsp
    uv run railroad example lsp-point-goal-nav --num-robots 3
    uv run railroad example lsp-point-goal-nav --frontier-statistics learned --network-file data/maze/training/LSPFrontierNet.pt
"""

from __future__ import annotations


def main(
    env_name: str = "maze",
    seed: int | None = None,
    frontier_statistics_name: str = "oracle",
    prior_prob: float = 0.8,
    network_file: str | None = None,
    save_data_dir: str | None = None,
    num_robots: int = 1,
    allow_move_interruptions: bool = False,
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    video_fps: int = 60,
    video_dpi: int = 150,
    video_time: float | str | None = None,
) -> None:
    """Run point-goal navigation with LSP frontier actions."""
    from railroad.core import get_action_by_name
    from railroad.dashboard import PlannerDashboard
    from railroad.planner import MCTSPlanner

    try:
        from railroad.lsp.rollout import build_point_goal_setup
    except ImportError as e:
        raise ImportError(
            "railsim dependencies not installed. "
            "Install with: pip install railroad[railsim]"
        ) from e

    # ------------------------------------------------------------------
    # Setup: scene, frontier statistics, environment, data writer
    # ------------------------------------------------------------------

    scene_seed = seed if seed is not None else 2024
    print(f"Generating {env_name} scene (seed={scene_seed})...")
    setup = build_point_goal_setup(
        env_name,
        scene_seed,
        frontier_statistics_name=frontier_statistics_name,
        prior_prob=prior_prob,
        network_file=network_file,
        save_data_dir=save_data_dir,
        allow_move_interruptions=allow_move_interruptions,
        num_robots=num_robots,
    )
    scene, env, goal = setup.scene, setup.env, setup.goal
    data_writer = setup.data_writer

    print(f"Grid: {scene.grid.shape[0]}x{scene.grid.shape[1]} "
          f"({scene.resolution} m/cell)")
    print(f"Start: {scene.locations['start_loc']}  Goal: {setup.goal_cell}  "
          f"Robots: {num_robots}")

    # ------------------------------------------------------------------
    # Planning loop
    # ------------------------------------------------------------------

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

            # Large scenes expose many frontiers, so prune the explore actions:
            # per robot keep only the most-probable / cheapest few achievers of
            # `revealed goal`, then drop any frontier left with no reason to be
            # visited (and the moves routing to it), so MCTS branches over a
            # bounded set of exploration subgoals.
            mcts = MCTSPlanner(
                actions,
                prune_top_n=4,
                prune_cheapest_m=2,
                frontier_objects=set(env.frontiers),
            )
            # Point-goal navigation is value-driven: the FF heuristic on the
            # post-move state is ~D_optimistic(frontier, goal), so a small
            # exploration constant and a strong heuristic weight make MCTS
            # follow min_f [D(robot, f) + D_opt(f, goal)] instead of
            # wandering to whichever frontier is nearest (c=300 — tuned for
            # the explore-everything examples — drowns the value signal).
            action_name = mcts(
                env.state,
                goal,
                max_iterations=5000,
                c=300,
                max_depth=10,
                heuristic_multiplier=1.5,
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
        video_time=video_time,
    )

    scene.release()


if __name__ == "__main__":
    main()

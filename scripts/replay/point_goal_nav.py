"""Learned-policy offline replay (point-goal), end to end, with videos.

Demonstrates the real served-vantage panorama pipeline with the *model* faked:

1. Deploy an oracle point-goal policy in a railsim maze with a VISUAL env
   (records panoramas; needs OpenGL — run with RAILSIM_GL_BACKEND=egl or cpu).
   Renders ``deployment.mp4`` via the standard PlannerDashboard.
2. Record a RolloutLog (carries the panoramas), save+reload it (panos persist).
3. Build a policy-agnostic arena (ReplayEnvironment.from_log) and replay candidate
   "learned" policies over it. Renders ``replay.mp4`` for one candidate and prints
   a per-policy bound comparison for all. Each policy is
   ``LearnedFrontierStatistics(model)`` fed the recorded panoramas; only the
   model's numeric output is faked (preset_model) — a trained net is a drop-in.

Both videos use the same dashboard, which colours each frontier by the policy's
predicted probability and shows the running cost.

Usage:  RAILSIM_GL_BACKEND=egl uv run python scripts/replay/point_goal_nav.py \
            [--seed S] [--num-robots N] [--env maze|office]
"""

from __future__ import annotations

import argparse
from pathlib import Path

OUT_DIR = Path("data/replay/point_goal_nav")
FPS, DPI = 10, 130


def _fluent_filter(f):  # noqa: ANN001
    return any(kw in f.name for kw in ["at", "explored", "revealed"])


# ----------------------------------------------------------------------
# Planning loop (standard MCTS loop with PlannerDashboard) — example pattern
# ----------------------------------------------------------------------


def run_planning(env, goal, label: str, save_video: str, *, max_iterations: int = 200) -> float:
    """Drive one MCTS plan->act loop with a dashboard; return the makespan.

    Deployment and replay share this loop *and* its MCTS configuration — the
    counterfactual must hold the planner fixed and vary only the policy (the
    frontier-statistics estimator baked into the env).
    """
    from railroad.core import get_action_by_name
    from railroad.dashboard import PlannerDashboard
    from railroad.planner import MCTSPlanner

    with PlannerDashboard(goal, env, fluent_filter=_fluent_filter) as dashboard:
        act_callback = dashboard.make_act_callback()
        dashboard.console.print(f"[bold]{label}[/bold]")
        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                break
            actions = env.get_actions()
            if not actions:
                break
            mcts = MCTSPlanner(actions)
            action_name = mcts(
                env.state, goal,
                max_iterations=2000, c=10, max_depth=20, heuristic_multiplier=5,
            )
            if action_name in ("NONE", ""):
                break
            action = get_action_by_name(actions, action_name)
            env.act(action, loop_callback_fn=act_callback)
            dashboard.update(mcts, action_name)

    dashboard.show_plots(save_video=save_video, video_fps=FPS, video_dpi=DPI)
    return float(env.state.time)


def main(seed: int = 1, num_robots: int = 2, env_name: str = "maze") -> None:
    """Deploy + record, then replay candidate learned policies and rank by bound."""
    from railroad.environment.types import Pose
    from railroad.lsp.frontier_statistics import (
        FixedPriorFrontierStatistics,
        LearnedFrontierStatistics,
    )
    from railroad.replay import (
        ReplayEnvironment,
        accumulate_bounds,
        build_rollout_log,
        goal_fluent,
        load_rollout_log,
        preset_model,
        run_replay,
        save_rollout_log,
    )

    try:
        from railroad.lsp.rollout import build_point_goal_setup
    except ImportError as e:
        raise ImportError(
            "railsim dependencies not installed. "
            "Install with: pip install railroad[railsim]"
        ) from e

    # ------------------------------------------------------------------
    # Deployment: oracle policy records panoramas + the rollout
    # ------------------------------------------------------------------

    if num_robots < 1:
        raise ValueError("num_robots must be >= 1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup = build_point_goal_setup(
        env_name, seed, frontier_statistics_name="oracle", num_robots=num_robots
    )
    dep_env, goal = setup.env, setup.goal
    dep_cost = run_planning(
        dep_env, goal, "Deployment (oracle planner)",
        str(OUT_DIR / "deployment.mp4"), max_iterations=200,
    )

    # ------------------------------------------------------------------
    # Record the rollout (carries the panoramas)
    # ------------------------------------------------------------------

    start = setup.scene.locations["start_loc"]
    start_pose = Pose(float(start[0]), float(start[1]), 0.0)
    robots = sorted(dep_env.objects_by_type["robot"])
    log = build_rollout_log(
        dep_env,
        goal_cell=setup.goal_cell,
        robot_starts={robot: start_pose for robot in robots},
        env_name=env_name,
        seed=seed,
    )
    print(f"deployment: reached={goal.evaluate(dep_env.state.fluents)} cost={dep_cost:.1f}s "
          f"panos={len(log.pano_records)} grid={log.recorded_grid.shape}")

    try:
        # Persist + reload the log (panoramas survive the round-trip).
        save_rollout_log(log, OUT_DIR / f"seed_{seed}")
        log = load_rollout_log(OUT_DIR / f"seed_{seed}")
        print(f"saved + reloaded log (panos={len(log.pano_records)} "
              f"robots={log.robots})")

        # --------------------------------------------------------------
        # Replay candidate policies + rank by bound (selection precursor)
        # --------------------------------------------------------------

        # Candidate "learned" policies; only the model output is faked.
        # SWAP: LearnedFrontierStatistics(load_frontier_statistics_model("LSPFrontierNet.pt"))
        policies = {
            "learned[optimistic]": LearnedFrontierStatistics(preset_model("optimistic")),
            "learned[cautious]": LearnedFrontierStatistics(preset_model("cautious")),
            "fixed-prior": FixedPriorFrontierStatistics(prob_feasible=0.5),
        }

        # Text comparison across all candidates over one recording.
        arena = ReplayEnvironment.from_log(log)
        results = {
            name: run_replay(arena, est, max_planning_iterations=80, mcts_iterations=2000)
            for name, est in policies.items()
        }
        ranked = sorted(results.items(), key=lambda kv: kv[1].bounds.simply_connected_lb)

        # --------------------------------------------------------------
        # Render a replay video for the top candidate + report
        # --------------------------------------------------------------

        # The replay env serves each onboard observation from the robot's pose out
        # of the recorded panoramas, so the dashboard's onboard pane tracks it.
        best_name = ranked[0][0]
        rep_env = ReplayEnvironment.from_log(log, policies[best_name])
        rep_env.scene = setup.scene  # type: ignore[attr-defined]  # expose to dashboard for overhead map
        rep_goal = goal_fluent(log.robots)
        rep_cost = run_planning(
            rep_env, rep_goal, f"Replay — {best_name}",
            str(OUT_DIR / "replay.mp4"), max_iterations=120,
        )
        bounds = accumulate_bounds(rep_env.replay_commits, rep_cost)

        print("\n========== POLICY COMPARISON (replayed over one deployment) ==========")
        for name, result in ranked:
            print(f"  {name:22s}  C_sc={result.total_cost:7.1f}  "
                  f"C_opt={result.bounds.optimistic_lb:7.1f}  reached={result.goal_reached}")
        print(f"\nvideo: deployment.mp4 (oracle) + replay.mp4 ({best_name}, "
              f"C_sc={rep_cost:.1f} C_opt={bounds.optimistic_lb:.1f})")
        print(f"saved to {OUT_DIR}/")
    finally:
        setup.scene.release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learned-policy offline replay (point-goal) with videos."
    )
    parser.add_argument("--seed", type=int, default=1, help="scene seed")
    parser.add_argument(
        "--num-robots", type=int, default=2,
        help="robots deployed (co-located at start; any reaching goal wins)",
    )
    parser.add_argument(
        "--env", dest="env_name", choices=("maze", "office"), default="maze",
        help="railsim world to deploy in",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(seed=args.seed, num_robots=args.num_robots, env_name=args.env_name)

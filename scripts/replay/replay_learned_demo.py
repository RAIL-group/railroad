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

Usage:  RAILSIM_GL_BACKEND=egl uv run python scripts/replay/replay_learned_demo.py [seed]
"""

from __future__ import annotations

import sys
from pathlib import Path

from railroad.core import get_action_by_name
from railroad.dashboard import PlannerDashboard
from railroad.environment.types import Pose
from railroad.lsp.frontier_statistics import (
    FixedPriorFrontierStatistics,
    LearnedFrontierStatistics,
)
from railroad.lsp.rollout import build_point_goal_setup
from railroad.planner import MCTSPlanner
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

OUT_DIR = Path("data/replay_learned")
FPS, DPI = 10, 130


def _fluent_filter(f):  # noqa: ANN001
    return any(kw in f.name for kw in ["at", "explored", "revealed"])


def deploy_and_record(env_name: str, seed: int, video: str, num_robots: int = 1):
    """Oracle deployment (records panos) → deployment video + RolloutLog + setup."""
    setup = build_point_goal_setup(
        env_name, seed, frontier_statistics_name="oracle", num_robots=num_robots
    )
    env, goal = setup.env, setup.goal
    with PlannerDashboard(goal, env, fluent_filter=_fluent_filter) as dash:
        cb = dash.make_act_callback()
        dash.console.print("[bold]Deployment (oracle planner)[/bold]")
        for _ in range(200):
            if goal.evaluate(env.state.fluents):
                break
            actions = env.get_actions()
            if not actions:
                break
            mcts = MCTSPlanner(
                actions, prune_top_n=4, prune_cheapest_m=2,
                frontier_objects=set(env.frontiers),
            )
            name = mcts(env.state, goal, max_iterations=5000, c=300, max_depth=10,
                        heuristic_multiplier=1.5)
            if name == "NONE":
                break
            env.act(get_action_by_name(actions, name), loop_callback_fn=cb)
            dash.update(mcts, name)
        dash.show_plots(save_video=video, video_fps=FPS, video_dpi=DPI)

    start = setup.scene.locations["start_loc"]
    start_pose = Pose(float(start[0]), float(start[1]), 0.0)
    robots = sorted(setup.env.objects_by_type["robot"])
    log = build_rollout_log(
        env,
        goal_cell=setup.goal_cell,
        robot_starts={robot: start_pose for robot in robots},
        env_name=env_name,
        seed=seed,
    )
    print(f"deployment: reached={goal.evaluate(env.state.fluents)} "
          f"panos={len(log.pano_records)} grid={log.recorded_grid.shape}")
    return log, setup


def replay_with_video(log, setup, estimator, label: str, video: str):
    """Replay one candidate over the recording, rendering a dashboard video.

    The replay env serves each onboard observation from the robot's pose out of
    the recorded panoramas, so the dashboard's onboard pane tracks the robot's
    actual trajectory.
    """
    env = ReplayEnvironment.from_log(log, estimator)
    env.scene = setup.scene  # type: ignore[attr-defined]  # for dashboard overhead
    goal = goal_fluent(log.robots)
    with PlannerDashboard(goal, env, fluent_filter=_fluent_filter) as dash:
        cb = dash.make_act_callback()
        dash.console.print(f"[bold]Replay — {label}[/bold]")
        for _ in range(120):
            if goal.evaluate(env.state.fluents):
                break
            actions = env.get_actions()
            if not actions:
                break
            mcts = MCTSPlanner(actions)
            name = mcts(env.state, goal, max_iterations=2000, c=10, max_depth=20,
                        heuristic_multiplier=5)
            if name in ("NONE", ""):
                break
            env.act(get_action_by_name(actions, name), loop_callback_fn=cb)
            dash.update(mcts, name)
        dash.show_plots(save_video=video, video_fps=FPS, video_dpi=DPI)
    # Makespan (seconds), same unit as the deployment's actual_total_cost.
    total = float(env.state.time)
    return accumulate_bounds(env.replay_commits, total), total


def main(seed: int = 1, num_robots: int = 1, env_name: str = "maze") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log, setup = deploy_and_record(
        env_name, seed, str(OUT_DIR / "deployment.mp4"), num_robots=num_robots
    )
    try:
        # Persist + reload the log (panoramas survive the round-trip).
        save_rollout_log(log, OUT_DIR / f"seed_{seed}")
        log = load_rollout_log(OUT_DIR / f"seed_{seed}")
        print(f"saved + reloaded log (panos={len(log.pano_records)} "
              f"robots={log.robots})")

        # Candidate "learned" policies; only the model output is faked.
        # SWAP: LearnedFrontierStatistics(load_frontier_statistics_model("LSPFrontierNet.pt"))
        policies = {
            "learned[optimistic]": LearnedFrontierStatistics(preset_model("optimistic")),
            "learned[cautious]": LearnedFrontierStatistics(preset_model("cautious")),
            "fixed-prior": FixedPriorFrontierStatistics(prob_feasible=0.5),
        }

        # Text comparison across all candidates (selection precursor).
        arena = ReplayEnvironment.from_log(log)
        results = {
            name: run_replay(arena, est, max_planning_iterations=80, mcts_iterations=2000)
            for name, est in policies.items()
        }
        ranked = sorted(results.items(), key=lambda kv: kv[1].bounds.simply_connected_lb)

        # Render a replay video for the top candidate.
        best_name = ranked[0][0]
        bounds, total = replay_with_video(
            log, setup, policies[best_name], best_name, str(OUT_DIR / "replay.mp4")
        )

        print("\n========== POLICY COMPARISON (replayed over one deployment) ==========")
        for name, result in ranked:
            print(f"  {name:22s}  C_sc={result.total_cost:7.1f}  "
                  f"C_opt={result.bounds.optimistic_lb:7.1f}  reached={result.goal_reached}")
        print(f"\nvideo: deployment.mp4 (oracle) + replay.mp4 ({best_name}, "
              f"C_sc={total:.1f} C_opt={bounds.optimistic_lb:.1f})")
        print(f"saved to {OUT_DIR}/")
    finally:
        setup.scene.release()


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    num_robots = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    env_name = sys.argv[3] if len(sys.argv) > 3 else "maze"
    main(seed, num_robots, env_name)

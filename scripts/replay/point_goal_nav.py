"""Learned-policy offline replay (point-goal), end to end, with videos.

Demonstrates the real served-vantage panorama pipeline with the *model* faked:

1. Deploy an oracle point-goal policy in a railsim maze with a VISUAL env
   (records panoramas; needs OpenGL — run with RAILSIM_GL_BACKEND=egl or cpu).
   Renders ``deployment.mp4`` via the shared dashboard loop.
2. Record a RolloutLog (carries the panoramas), save+reload it (panos persist).
3. Replay three candidate "learned" policies over the recording. Each replays on
   a fresh ``build_replay_env(log)`` arena via ``run_replay`` with the dashboard,
   rendering its own ``replay_<policy>.mp4`` and reporting its bounds. Each policy
   is ``LearnedFrontierStatistics(model)`` fed the recorded panoramas; only the
   model's numeric output is faked (preset_model) — a trained net is a drop-in.

Deployment and replay share one plan->act loop AND one ``MctsConfig`` (``MCTS``),
so the counterfactual holds the planner fixed and varies only the policy.

Usage:  RAILSIM_GL_BACKEND=egl uv run python scripts/replay/point_goal_nav.py \
            [--seed S] [--num-robots N] [--env maze|office]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from railroad.replay import MctsConfig

OUT_DIR = Path("data/replay/point_goal_nav")

# One planner config shared by the deployment and every replay: the counterfactual
# must hold the planner fixed and vary only the policy.
MCTS = MctsConfig(iterations=2000, c=10.0, max_depth=20, heuristic_multiplier=5.0)
MAX_ITERS = 200


def _slug(name: str) -> str:
    """A filesystem-safe token for a policy name (for the per-policy video file)."""
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def main(seed: int = 1, num_robots: int = 2, env_name: str = "maze") -> None:
    """Deploy + record, then replay candidate learned policies and rank by bound."""
    from railroad.environment.types import Pose
    from railroad.lsp.frontier_statistics import (
        FixedPriorFrontierStatistics,
        LearnedFrontierStatistics,
    )
    from railroad.replay import (
        CandidatePolicy,
        build_replay_env,
        build_rollout_log,
        load_rollout_log,
        mcts_selector,
        preset_model,
        run_dashboard_loop,
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
    run_dashboard_loop(
        dep_env,
        goal,
        select=mcts_selector(MCTS),
        max_iterations=MAX_ITERS,
        fluent_keywords=("at", "explored", "revealed"),
        scene=setup.scene,
        save_video=str(OUT_DIR / "deployment.mp4"),
        label="Deployment (oracle planner)",
    )
    dep_cost = float(dep_env.state.time)

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

    # Persist + reload the log (panoramas survive the round-trip).
    save_rollout_log(log, OUT_DIR / f"seed_{seed}")
    log = load_rollout_log(OUT_DIR / f"seed_{seed}")
    print(f"saved + reloaded log (panos={len(log.pano_records)} "
          f"robots={log.robots})")

    # ------------------------------------------------------------------
    # Replay candidate policies + rank by bound (selection precursor)
    # ------------------------------------------------------------------

    # Candidate "learned" policies; only the model output is faked.
    # SWAP: LearnedFrontierStatistics(load_frontier_statistics_model("LSPFrontierNet.pt"))
    policies = {
        "learned[optimistic]": CandidatePolicy(
            name="learned[optimistic]",
            frontier_statistics=LearnedFrontierStatistics(preset_model("optimistic")),
        ),
        "learned[cautious]": CandidatePolicy(
            name="learned[cautious]",
            frontier_statistics=LearnedFrontierStatistics(preset_model("cautious")),
        ),
        "fixed-prior": CandidatePolicy(
            name="fixed-prior",
            frontier_statistics=FixedPriorFrontierStatistics(prob_feasible=0.5),
        ),
    }

    # Replay each candidate over a fresh arena (same MCTS) and render its video.
    # The replay env serves each onboard observation from the robot's pose out of
    # the recorded panoramas, so the dashboard's onboard pane tracks it; the
    # dashboard run returns the same bounds a silent run would.
    results = {
        name: run_replay(
            build_replay_env(log),
            policy,
            dashboard=True,
            scene=setup.scene,
            save_video=str(OUT_DIR / f"replay_{_slug(name)}.mp4"),
            label=f"Replay — {name}",
            mcts=MCTS,
            max_planning_iterations=MAX_ITERS,
        )
        for name, policy in policies.items()
    }
    ranked = sorted(results.items(), key=lambda kv: kv[1].bounds.simply_connected_lb)

    print("\n========== POLICY COMPARISON (replayed over one deployment) ==========")
    for name, result in ranked:
        print(f"  {name:22s}  C_sc={result.total_cost:7.1f}  "
              f"C_opt={result.bounds.optimistic_lb:7.1f}  reached={result.goal_reached}")
    print("\nvideos: deployment.mp4 (oracle) + "
          + ", ".join(f"replay_{_slug(n)}.mp4" for n in policies))
    print(f"saved to {OUT_DIR}/")
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

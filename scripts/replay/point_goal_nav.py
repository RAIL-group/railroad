"""Offline replay for point-goal navigation in unknown space (railsim).

Drives the real served-vantage panorama pipeline end to end, with only the
*model* faked. The eight steps below are the shared shape of all three replay
scripts; only scene setup and ``problem_class`` differ between them.

1. Build the scene and the deployment environment.
2. Build the policies this run can choose from (``build_policies``); only the
   oracle takes the scene, so this is the one place ground truth enters.
3. Pick the ``--deploy-policy`` and install it on the already-built environment.
4. Deploy: run the plan->act loop and record a ``RolloutLog`` (carrying the
   panoramas). Saved and reloaded here, to prove panoramas survive the round trip.
5. Pick the ``--replay-policy`` candidates.
6. Build a fresh replay arena per candidate from the log.
7. Replay each candidate over it.
8. Report cost bounds.

Every policy works in both roles. ``oracle`` labels frontiers against the scene's
true map in *both* deployment and replay: it is a black box to the bound, so
consulting ground truth is fine — the replayed cost accounting still reads only
what the deployment recorded. ``optimistic`` / ``cautious`` / ``uniform`` run the
real served-vantage pipeline with a preset model (a trained net drops in at the
same call site via ``--replay-policy learned --network-file``); ``fixed-prior``
bypasses perception entirely.

Deployment and replay share one plan->act loop AND one ``MctsConfig`` (``MCTS``),
so the counterfactual holds the planner fixed and varies only the policy.

Usage:  RAILSIM_GL_BACKEND=egl uv run python scripts/replay/point_goal_nav.py \
            [--deploy-policy P] [--replay-policy P[,P...]] [--seed S] \
            [--num-robots N] [--env maze|office] [--network-file PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from railroad.lsp.frontier_statistics import (
    FixedPriorFrontierStatistics,
    FrontierStatisticsEstimator,
)
from railroad.replay import (
    MctsConfig,
    constant_frontier_statistics,
    learned_frontier_statistics,
    oracle_frontier_statistics,
)

OUT_DIR = Path("data/replay/point_goal_nav")

# One planner config shared by the deployment and every replay: the counterfactual
# must vary only the policy, so the planner is held fixed.
MCTS = MctsConfig(iterations=4000, c=10.0, max_depth=20, heuristic_multiplier=5.0)
MAX_ITERS = 300

# ----------------------------------------------------------------------
# Policies this experiment compares
# ----------------------------------------------------------------------
#
# For point-goal navigation a policy IS a ``FrontierStatisticsEstimator``:
# "does this frontier lead to the goal, and at what cost?". Nothing about object
# search appears here, because none of it applies.
#
# The library supplies the belief models; *which* of them this study compares,
# under what names and tuning, is an experiment choice and lives here so it is
# visible where it is varied. One built estimator per name, shared by both roles:
# safe because every refresh() *replaces* its cache rather than accumulating, and
# the deployment finishes before any replay begins.

POLICY_NAMES = ("cautious", "fixed-prior", "learned", "optimistic", "oracle", "uniform")


def build_policies(
    scene: Any, *, network_file: str | None = None
) -> dict[str, FrontierStatisticsEstimator]:
    """The navigation policies this run offers, by name.

    ``learned`` appears only when weights were supplied — a run that never asks
    for it should not have to own a network file.
    """
    policies: dict[str, FrontierStatisticsEstimator] = {
        # Perfect knowledge: frontier labels against the scene's true map. The
        # only entry that needs the scene, being the only one consulting truth.
        "oracle": oracle_frontier_statistics(scene),
        # Constant beliefs that still run the REAL served-vantage pipeline —
        # best-vantage selection and panorama serving happen, only the numbers
        # are faked, so "learned" is a drop-in at the same call site.
        "optimistic": constant_frontier_statistics(0.9, exploration_cost=8.0),
        "cautious": constant_frontier_statistics(0.3, exploration_cost=20.0),
        "uniform": constant_frontier_statistics(0.5),
        # The control: bypasses perception entirely, so comparing it against
        # "uniform" answers "is perception doing anything?".
        "fixed-prior": FixedPriorFrontierStatistics(prob_feasible=0.5),
    }
    if network_file is not None:
        policies["learned"] = learned_frontier_statistics(network_file)
    return policies


def _slug(name: str) -> str:
    """Filesystem-safe form of a policy name (for video filenames)."""
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def main(
    seed: int = 1,
    num_robots: int = 2,
    env_name: str = "maze",
    deploy_policy: str = "oracle",
    replay_policies: tuple[str, ...] = ("optimistic",),
    network_file: str | None = None,
) -> None:
    """Deploy one policy, record it, and replay candidates over the recording."""
    from railroad.environment.types import Pose
    from railroad.replay import (
        build_replay_env,
        load_rollout_log,
        run_deployment,
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

    if num_robots < 1:
        raise ValueError("num_robots must be >= 1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1. scene + deployment environment ----------------------------
    # Built with a cheap fixed prior: the real policy is installed in step 3, so
    # naming an estimator here would only make init do work we discard.
    setup = build_point_goal_setup(
        env_name,
        seed,
        frontier_statistics_name="fixed-prior",
        num_robots=num_robots,
    )
    scene, dep_env, goal = setup.scene, setup.env, setup.goal

    # -- 2. the policies this run can choose from ---------------------
    # Only the oracle takes the scene — it is the only one that consults ground
    # truth. The rest need nothing, or just the weights path.
    policies = build_policies(scene, network_file=network_file)

    missing = sorted({deploy_policy, *replay_policies} - set(policies))
    if missing:
        raise SystemExit(
            f"policies {missing} are unavailable in this run; "
            "'learned' needs --network-file"
        )

    # -- 3. pick a policy to deploy and install it --------------------

    # lsp-explore reads the estimator live, so this takes effect on the
    # already-built environment — no scene or GL context is rebuilt per policy.
    dep_env.frontier_statistics = policies[deploy_policy]
    print(f"deploy-policy={deploy_policy}  replay-policies={list(replay_policies)}")

    # -- 4. deploy and record -----------------------------------------
    start = scene.locations["start_loc"]
    start_pose = Pose(float(start[0]), float(start[1]), 0.0)
    robots = sorted(dep_env.objects_by_type["robot"])
    deployment = run_deployment(
        dep_env,
        goal,
        goal_cell=setup.goal_cell,
        robot_starts={robot: start_pose for robot in robots},
        problem_class="navigation",
        mcts=MCTS,
        max_planning_iterations=MAX_ITERS,
        dashboard=True,
        scene=scene,
        save_video=str(OUT_DIR / f"deployment_{_slug(deploy_policy)}.mp4"),
        label=f"Deployment ({deploy_policy} planner)",
        fluent_keywords=("at", "explored", "revealed"),
        env_name=env_name,
        seed=seed,
    )
    log = deployment.log
    print(f"deployment: reached={deployment.goal_reached} "
          f"cost={deployment.total_cost:.1f}s panos={len(log.pano_records)} "
          f"grid={log.recorded_grid.shape}")

    # Persist + reload (panoramas survive the round-trip).
    save_rollout_log(log, OUT_DIR / f"seed_{seed}")
    log = load_rollout_log(OUT_DIR / f"seed_{seed}")
    print(f"saved + reloaded log (panos={len(log.pano_records)} robots={log.robots})")

    # -- 5-8. replay each candidate over a fresh arena ----------------
    # Each candidate gets its own arena. The replay env serves every onboard
    # observation from the robot's pose out of the recorded panoramas, so the
    # dashboard's onboard pane tracks it.
    results = {}
    for name in replay_policies:
        candidate = policies[name]
        results[name] = run_replay(
            build_replay_env(log),
            candidate,
            dashboard=True,
            scene=scene,
            save_video=str(
                OUT_DIR / f"replay_{_slug(name)}_from_{_slug(deploy_policy)}.mp4"
            ),
            label=f"Replay — {name}",
            mcts=MCTS,
            max_planning_iterations=MAX_ITERS,
        )

    print("\n========== REPLAY (over one deployment) ==========")
    print(f"deployment ({deploy_policy}) cost = {deployment.total_cost:.1f}s")
    for name, result in sorted(
        results.items(), key=lambda kv: kv[1].bounds.simply_connected_lb
    ):
        print(f"  {name:14s}  C_sc={result.bounds.simply_connected_lb:7.1f}  "
              f"C_opt={result.bounds.optimistic_lb:7.1f}  reached={result.goal_reached}")
    print(f"\nvideos: deployment_{_slug(deploy_policy)}.mp4 + "
          + ", ".join(
              f"replay_{_slug(n)}_from_{_slug(deploy_policy)}.mp4" for n in results
          ))
    print(f"saved to {OUT_DIR}/")
    scene.release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1, help="railsim world seed")
    parser.add_argument(
        "--num-robots", type=int, default=2,
        help="robots deployed (all start co-located; any reaching the goal wins)",
    )
    parser.add_argument(
        "--env", dest="env_name", choices=("maze", "office"), default="maze",
        help="railsim world to deploy in",
    )
    parser.add_argument(
        "--deploy-policy", choices=POLICY_NAMES, default="oracle",
        help="policy that runs the live deployment (generates the recording)",
    )
    parser.add_argument(
        "--replay-policy", dest="replay_policies", default="optimistic",
        help=(
            "policy replayed over the recording; comma-separate several to "
            f"rank them over one deployment. One of: {', '.join(POLICY_NAMES)}"
        ),
    )
    parser.add_argument(
        "--network-file", default=None,
        help="trained frontier-statistics weights (required by the 'learned' policy)",
    )
    args = parser.parse_args(argv)
    args.replay_policies = tuple(
        name.strip() for name in args.replay_policies.split(",") if name.strip()
    )
    unknown = sorted(set(args.replay_policies) - set(POLICY_NAMES))
    if unknown:
        parser.error(f"unknown --replay-policy {unknown}; choose from {POLICY_NAMES}")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        seed=args.seed,
        num_robots=args.num_robots,
        env_name=args.env_name,
        deploy_policy=args.deploy_policy,
        replay_policies=args.replay_policies,
        network_file=args.network_file,
    )

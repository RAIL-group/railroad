"""Offline replay for object search in an *unknown* ProcTHOR environment.

The robot must explore frontiers and search containers for a two-object goal;
the observed map is partial throughout. (See known_map_search.py for the
fully-known-map counterpart.)

The eight steps are the shared shape of all three replay scripts; only scene
setup and ``problem_class`` differ.

1. Build the scene and the deployment environment.
2. Build the policies this run can choose from (``build_policies``); only the
   oracle takes the scene, so this is the one place ground truth enters.
3. Pick the ``--deploy-policy``; the env reads it live through
   ``object_find_statistics``, so it can be chosen after the env exists.
4. Deploy: run the plan->act loop; the trajectory plus the contents of every
   container inspected are recorded into a ``RolloutLog``.
5. Pick the ``--replay-policy`` candidates.
6. Build a fresh replay arena per candidate — it confines the robot to the
   observed map and resolves every search outcome from the recording.
7. Replay each candidate.
8. Report the simply-connected (actual makespan) cost vs the optimistic lower
   bound — both in deployment units (seconds), so they compare directly.

Every policy works in both roles. ``oracle`` is a *full* oracle here: decisive
ground-truth container belief, plus a frontier oracle that asks whether a
still-hidden target container lies behind each frontier (``compute_oracle_frontier_labels``
run per target container). It is a black box to the bound — the replayed cost
accounting still reads only what the deployment recorded.

Deployment and replay share one plan->act loop and one ``MctsConfig`` (``MCTS``).

Usage:  uv run python scripts/replay/unknown_map_search.py \\
            [--deploy-policy P] [--replay-policy P[,P...]] [--seed S] [--num-robots N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from railroad.experimental.unknown_search import (
    FixedObjectFind,
    ObjectFindEstimator,
)
from railroad.replay import MctsConfig, oracle_object_find

if TYPE_CHECKING:
    import numpy as np
    from railroad.environment.procthor import ProcTHORScene

OUT_DIR = Path("data/replay/unknown_map_search")

# One planner config shared by the deployment and every replay.
MCTS = MctsConfig(iterations=10000, c=300.0, max_depth=20, heuristic_multiplier=2.0)
MAX_ITERS = 100

# ----------------------------------------------------------------------
# Policies this experiment compares
# ----------------------------------------------------------------------
#
# For object search a policy IS an ``ObjectFindEstimator``: "is the object
# beyond this frontier / inside this container?". Nothing about point-goal
# navigation appears here — no frontier exploration costs, no goal cell —
# because none of it applies.
#
# The library supplies the belief models; *which* of them this study compares,
# under what names and tuning, is an experiment choice and lives here so it is
# visible where it is varied. One built estimator per name, shared by both roles:
# safe because every refresh() *replaces* its cache rather than accumulating, and
# the deployment finishes before any replay begins.

POLICY_NAMES = ("cautious", "optimistic", "oracle", "uniform")


def build_policies(
    scene: Any, *, target_objects: Sequence[str] = ()
) -> dict[str, ObjectFindEstimator]:
    """The object-search policies this run offers, by name.

    No ``learned`` entry: a learned frontier belief predicts from panoramas, and
    this pipeline records none — the deployment env has no ``pano_records``, so
    the log carries none and replay has nothing to serve. Offering it would load
    a network and then silently plan on the default prior. Point-goal navigation
    does have a panorama pipeline (railsim), so ``learned`` lives there.
    """
    policies: dict[str, ObjectFindEstimator] = {
        # Perfect knowledge: decisive container truth (plus frontier truth where
        # the map is unknown). The only entry that needs the scene.
        "oracle": oracle_object_find(scene, target_objects=target_objects),
        # Flat beliefs over containers and frontiers alike.
        "optimistic": FixedObjectFind(0.9),
        "cautious": FixedObjectFind(0.3),
        "uniform": FixedObjectFind(0.5),
    }
    return policies


# ----------------------------------------------------------------------
# Scene setup (mirrors examples/frontier_search.py::_setup_procthor)
# ----------------------------------------------------------------------


def setup_scene(seed: int) -> tuple[
    "ProcTHORScene",
    "np.ndarray",
    dict[str, tuple[int, int]],
    dict[str, set[str]],
    tuple[int, int],
]:
    """Load a ProcTHOR scene and extract grid, hidden sites, and object locations."""
    import numpy as np

    try:
        from railroad.environment.procthor import ProcTHORScene
    except ImportError as e:
        raise ImportError(
            "ProcTHOR dependencies not installed. "
            "Install with: pip install railroad[procthor]"
        ) from e

    scene = ProcTHORScene(seed=seed)
    true_grid = np.asarray(scene.grid, dtype=float)
    hidden_sites = {
        name: (int(loc[0]), int(loc[1]))
        for name, loc in scene.locations.items()
        if name != "start_loc"
    }
    true_object_locations = scene.object_locations
    start_coord = scene.locations["start_loc"]
    return scene, true_grid, hidden_sites, true_object_locations, start_coord


def pick_two_objects(object_locations: dict[str, set[str]]) -> list[str]:
    """Two objects in DISTINCT containers, so the search must visit two sites."""
    targets: list[str] = []
    used: set[str] = set()
    for container in sorted(object_locations):
        objs = sorted(object_locations[container])
        if objs and container not in used:
            targets.append(objs[0])
            used.add(container)
        if len(targets) == 2:
            break
    if len(targets) < 2:
        raise ValueError("scene has fewer than two containers with objects")
    return targets


def main(
    seed: int = 1089,
    num_robots: int = 2,
    deploy_policy: str = "oracle",
    replay_policies: tuple[str, ...] = ("optimistic",),
) -> None:
    """Deploy one policy, record it, and replay candidates over the recording."""
    import numpy as np

    from railroad._bindings import State
    from railroad.core import Fluent as F
    from railroad.environment.skill import NavigationMoveSkill
    from railroad.environment.symbolic import LocationRegistry
    from railroad.environment.types import Pose
    from railroad.experimental.unknown_search import (
        NavigationConfig,
        UnknownSpaceSearchEnvironment,
    )
    from railroad.replay import (
        build_replay_env,
        run_deployment,
        run_replay,
    )

    if num_robots < 1:
        raise ValueError("num_robots must be >= 1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    robots = [f"robot{i + 1}" for i in range(num_robots)]

    # -- 1. scene + deployment environment ----------------------------
    scene, true_grid, hidden_sites, true_obj_locs, start = setup_scene(seed)
    targets = pick_two_objects(true_obj_locs)
    goal = F(f"found {targets[0]}") & F(f"found {targets[1]}")
    true_containers = {
        o: next(c for c, objs in true_obj_locs.items() if o in objs) for o in targets
    }

    fluents: set = set()
    robot_initial_poses: dict = {}
    for robot in robots:
        fluents |= {F(f"at {robot} start_loc"), F(f"free {robot}"), F("revealed start_loc")}
        robot_initial_poses[robot] = Pose(float(start[0]), float(start[1]), 0.0)

    dep_env = UnknownSpaceSearchEnvironment(
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": {"start_loc"},
            "container": set(),
            "frontier": set(),
            "object": set(targets),
        },
        skill_overrides={"move": NavigationMoveSkill},
        true_grid=true_grid,
        true_object_locations=true_obj_locs,
        robot_initial_poses=robot_initial_poses,
        location_registry=LocationRegistry({"start_loc": np.array(start, dtype=float)}),
        hidden_sites=hidden_sites,
        config=NavigationConfig(
            sensor_range=60.0,
            max_move_action_time=10_000.0,
            interrupt_min_new_cells=30000,
            interrupt_min_dt=30000.0,
        ),
    )

    print(f"Grid: {true_grid.shape[0]}x{true_grid.shape[1]}  targets={targets}  "
          f"true_containers={true_containers}")
    print(f"Containers: {list(hidden_sites)}  start={start}  robots={robots}")

    # -- 2. the policies this run can choose from ---------------------
    # Only the oracle takes the scene — it is the only one that consults ground
    # truth. The rest need nothing, or just the weights path.
    policies = build_policies(scene, target_objects=targets)

    missing = sorted({deploy_policy, *replay_policies} - set(policies))
    if missing:
        raise SystemExit(
            f"policies {missing} are unavailable in this run; "
            "'learned' needs --network-file"
        )


    # -- 3. pick a policy to deploy and install it --------------------
    dep_env.object_find_statistics = policies[deploy_policy]
    print(f"deploy-policy={deploy_policy}  replay-policies={list(replay_policies)}")

    # -- 4. deploy and record -----------------------------------------
    deployment = run_deployment(
        dep_env,
        goal,
        goal_cell=(int(start[0]), int(start[1])),
        robot_starts={
            robot: Pose(float(start[0]), float(start[1]), 0.0) for robot in robots
        },
        problem_class="object-search",
        mcts=MCTS,
        max_planning_iterations=MAX_ITERS,
        dashboard=True,
        scene=scene,
        save_video=str(OUT_DIR / f"deployment_{deploy_policy}.mp4"),
        label=f"Deployment ({deploy_policy})",
        fluent_keywords=("at", "found", "searched"),
        env_name="procthor",
        seed=seed,
    )
    log = deployment.log
    print(f"deployment: found={deployment.goal_reached} "
          f"cost={deployment.total_cost:.1f}s")
    print(f"recorded: containers={[s.signature for s in log.subgoals]} "
          f"actual_total_cost={log.actual_total_cost:.1f}s")

    # -- 5-8. replay each candidate over a fresh arena ----------------
    # build_replay_env reconstructs the arena (hidden_sites + recorded contents +
    # goal come straight from the log); run_replay applies the candidate. The
    # find-probabilities only steer MCTS belief — outcomes resolve from the record.
    results = {}
    for name in replay_policies:
        results[name] = run_replay(
            build_replay_env(log),
            policies[name],
            dashboard=True,
            scene=scene,
            save_video=str(OUT_DIR / f"replay_{name}_from_{deploy_policy}.mp4"),
            label=f"Replay ({name})",
            mcts=MCTS,
            max_planning_iterations=MAX_ITERS,
        )

    # Bounds (all in deployment units: seconds / makespan). Each not-found search
    # in replay is a commit to a subgoal the deployment never verified;
    # optimistically the object is immediately at/past it, so C_opt = min over
    # those commit times.
    print("\n================ RESULTS ================")
    print(f"deployment ({deploy_policy}) cost = {deployment.total_cost:.1f}s")
    print("========== REPLAY (over one deployment) ==========")
    for name, result in sorted(
        results.items(), key=lambda kv: kv[1].bounds.simply_connected_lb
    ):
        print(f"  {name:14s}  C_sc={result.bounds.simply_connected_lb:7.1f}  "
              f"C_opt={result.bounds.optimistic_lb:7.1f}  found={result.goal_reached}")
    print(f"\nvideos: deployment_{deploy_policy}.mp4 + "
          + ", ".join(f"replay_{n}_from_{deploy_policy}.mp4" for n in results))
    print(f"saved to {OUT_DIR}/")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1089, help="ProcTHOR scene seed")
    parser.add_argument(
        "--num-robots", type=int, default=2,
        help="robots deployed (co-located at start; any finding the object wins)",
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
        deploy_policy=args.deploy_policy,
        replay_policies=args.replay_policies,
    )

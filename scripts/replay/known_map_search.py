"""Offline replay for known-map object search on ProcTHOR.

Unlike scripts/replay/unknown_map_search.py (an *unknown* environment: the robot
must explore frontiers and the observed map is partial), here the **map is fully
known** — the ProcTHOR floorplan is given — and only the objects' *presence in
containers* is unknown.

The eight steps are the shared shape of all three replay scripts; only scene
setup and ``problem_class`` differ.

1. Build the scene and the deployment environment.
2. Build the policies this run can choose from (``build_policies``); only the
   oracle takes the scene, so this is the one place ground truth enters.
3. Pick the ``--deploy-policy``; the env reads it live through
   ``object_find_statistics``, so it can be chosen after the env exists.
4. Deploy: run the plan->act loop and record which containers were inspected and
   what was in them.
5. Pick the ``--replay-policy`` candidates.
6. Build a fresh replay arena per candidate.
7. Replay each on the SAME known map. Its only knowledge of object presence is
   the recording (belief steers search order, never the outcome) — realized by
   restricting the replay env's ``_objects_at_locations`` to the recorded
   contents, so the existing deterministic search resolution replays searched
   containers exactly.
8. Report bounds. Travel is exact (map known), but object presence is known only
   where the deployment searched: a revealed-but-unsearched container is an
   unverified subgoal (we do not assume one container per object), so searching
   it commits. Each candidate's cost is a commit-based lower bound — ``C_opt``
   (min over those commits) vs. ``C_sc`` (its makespan) — collapsing onto the
   exact makespan only when every container was searched.

Every policy works in both roles. ``oracle`` uses decisive ground-truth container
belief in *both*: it is a black box to the bound, and the replayed cost
accounting still reads only what the deployment recorded.

Costs are makespan (``state.time``, seconds), so deployment and replay compare
directly.

Usage:  uv run python scripts/replay/known_map_search.py \\
            [--deploy-policy P] [--replay-policy P[,P...]] [--seed S] [--num-robots N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from railroad.experimental.unknown_search import (
    FixedObjectFind,
    ObjectFindEstimator,
)
from railroad.replay import (
    MctsConfig,
    learned_container_find,
    oracle_object_find,
)

OUT_DIR = Path("data/replay/known_map_search")

# One planner config shared by the deployment and every replay.
MCTS = MctsConfig(iterations=4000, c=300.0, max_depth=20, heuristic_multiplier=2.0)
MAX_ITERS = 40

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

POLICY_NAMES = ("cautious", "learned", "optimistic", "oracle", "uniform")


def build_policies(
    scene: Any, *, target_objects: Sequence[str] = (), network_file: str | None = None
) -> dict[str, ObjectFindEstimator]:
    """The object-search policies this run offers, by name.

    ``learned`` is ProcTHOR's ``FCNNforObjectSearch``, which scores each
    (room, container, object) triple from sentence embeddings of their names —
    the right model for a known map, where container choice IS the problem and
    there are no frontiers. Its trained checkpoint ships with ProcTHOR, so it
    needs no ``--network-file`` (pass one to override).
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
    # The trained checkpoint is packaged, so this always works.
    policies["learned"] = learned_container_find(scene, network_file)
    return policies


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
    network_file: str | None = None,
) -> None:
    """Deploy one policy on a known map, record it, and replay candidates."""
    from railroad.environment.procthor import ProcTHOREnvironment, ProcTHORScene

    from railroad import operators
    from railroad._bindings import State
    from railroad.core import Fluent as F, Operator
    from railroad.environment.types import Pose
    from railroad.experimental.unknown_search import (
        FixedObjectFind,
    )
    from railroad.replay import (
        build_replay_env,
        run_deployment,
        run_replay,
    )

    class SearchProcTHOREnvironment(ProcTHOREnvironment):
        """Known-map ProcTHOR search env: ``move`` + ``search`` over known locations.

        The find-probability only drives MCTS belief — the actual search outcome
        is resolved deterministically from ``_objects_at_locations`` (the ground
        truth, which replay restricts to the recorded contents).

        ``search`` reads the probability through ``self.object_find_statistics``
        rather than closing over a callable, because operators are built once at
        construction and would otherwise freeze whichever policy was installed
        then. That indirection is what lets step 3 pick a policy *after* the env
        exists — and it mirrors ``ReplayKnownMapSearchEnvironment``.
        """

        def __init__(self, **kwargs) -> None:
            self.object_find_statistics: ObjectFindEstimator = FixedObjectFind()
            super().__init__(**kwargs)

        def _container_find_prob(self, robot: str, loc: str, obj: str) -> float:
            return self.object_find_statistics.container_probability(robot, loc, obj)

        def define_operators(self) -> list[Operator]:
            return [
                operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0),
                operators.construct_move_operator_blocking(self.estimate_move_time),
                operators.construct_search_operator(self._container_find_prob, 10.0),
            ]

    if num_robots < 1:
        raise ValueError("num_robots must be >= 1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    robots = [f"robot{i + 1}" for i in range(num_robots)]

    # -- 1. scene + deployment environment ----------------------------
    scene = ProcTHORScene(seed=seed)
    containers = sorted(scene.object_locations.keys())
    targets = pick_two_objects(scene.object_locations)
    goal = F(f"found {targets[0]}") & F(f"found {targets[1]}")
    true_containers = {
        o: next(c for c, objs in scene.object_locations.items() if o in objs)
        for o in targets
    }

    fluents = {F("revealed start_loc")}
    for robot in robots:
        fluents |= {F(f"at {robot} start_loc"), F(f"free {robot}")}

    dep_env = SearchProcTHOREnvironment(
        seed=seed,
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": set(containers) | {"start_loc"},
            "object": set(targets),
        },
        validate=False,
    )

    print(f"Grid: {scene.grid.shape[0]}x{scene.grid.shape[1]}  "
          f"#containers={len(containers)}  targets={targets}  "
          f"true_containers={true_containers}  robots={robots}")

    # -- 2. the policies this run can choose from ---------------------
    # Only the oracle takes the scene — it is the only one that consults ground
    # truth. The rest need nothing, or just the weights path.
    policies = build_policies(
        scene, target_objects=targets, network_file=network_file
    )

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
    start = scene.locations["start_loc"]
    deployment = run_deployment(
        dep_env,
        goal,
        goal_cell=(int(start[0]), int(start[1])),
        robot_starts={
            robot: Pose(float(start[0]), float(start[1]), 0.0) for robot in robots
        },
        problem_class="known-map-search",
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
    searched = sorted({s.signature for s in log.subgoals if s.searched})
    print(f"deployment: found={deployment.goal_reached} "
          f"cost={deployment.total_cost:.1f}s searched={searched}")

    # -- 5-8. replay each candidate over a fresh arena ----------------
    # container_find_prob only steers MCTS belief — outcomes resolve from the
    # recorded contents. Each candidate replays on its own arena.
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

    # Bounds (seconds / makespan). A search at a container the deployment never
    # searched forces not-found and commits: optimistically the object was right
    # there (optimistic_to_goal=0). C_opt is the min over those commits; it
    # collapses onto C_sc only if the deployment searched every container.
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
        help="robots deployed (all start at start_loc; any finding the object wins)",
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
        help="override the packaged FCNN weights used by the 'learned' policy",
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
        network_file=args.network_file,
    )

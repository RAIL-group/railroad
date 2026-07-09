"""Known-map object-search replay on ProcTHOR.

Unlike scripts/replay/unknown_map_search.py (an *unknown* environment: the robot
must explore frontiers and the observed map is partial), here the **map is fully
known** — the ProcTHOR floorplan is given — and only the objects' *presence in
containers* is unknown. This mirrors examples/procthor_search.py (a
``ProcTHOREnvironment`` with ``move`` + ``search`` operators), the standard
known-map search task.

Pipeline:

1. Deployment (``deployment.mp4``): an *informed* policy (high find-prob at the
   container that truly holds each object) searches known containers for a
   two-object goal (``found A & found B``, objects in distinct containers) until
   it finds both. We record which containers it inspected and their contents.
2. Replay (``replay_<policy>.mp4``): three candidate policies (uniform vs.
   optimistic vs. cautious belief) are each replayed on the SAME known map. Their
   only knowledge of object presence is the recording (belief only steers search
   order, never the outcome) — realized by restricting the replay env's
   ``_objects_at_locations`` to the recorded contents, so the existing
   deterministic search resolution replays searched containers exactly. Travel is
   exact (map known), but object presence is known only where the deployment
   searched: a revealed-but-unsearched container is an unverified subgoal (we do
   not assume one container per object), so searching it commits. Each candidate's
   cost is a commit-based lower bound — ``C_opt`` (min over those commits) vs.
   ``C_sc`` (the candidate's makespan) — collapsing onto the exact makespan only
   when every container was searched.

Costs are makespan (``state.time``, seconds), so deployment and replay compare
directly.

Usage:  uv run python scripts/replay/known_map_search.py [--seed S] [--num-robots N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from railroad.replay import MctsConfig

OUT_DIR = Path("data/replay/known_map_search")

ProbFn = Callable[[str, str, str], float]

# One planner config shared by the deployment and the replay.
MCTS = MctsConfig(iterations=4000, c=300.0, max_depth=20, heuristic_multiplier=2.0)
MAX_ITERS = 40


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


def main(seed: int = 1089, num_robots: int = 2) -> None:
    """Deploy an informed policy on a known map, record it, and replay candidates."""
    from railroad.environment.procthor import ProcTHOREnvironment, ProcTHORScene

    from railroad import operators
    from railroad._bindings import State
    from railroad.core import Fluent as F, Operator
    from railroad.environment.types import Pose
    from railroad.replay import (
        CandidatePolicy,
        build_replay_env,
        build_rollout_log,
        mcts_selector,
        run_dashboard_loop,
        run_replay,
    )

    class SearchProcTHOREnvironment(ProcTHOREnvironment):
        """Known-map ProcTHOR search env: ``move`` + ``search`` over known locations.

        The find-probability callable is the swappable "policy" (informed vs
        naive); it only drives MCTS belief — the actual search outcome is
        resolved deterministically from ``_objects_at_locations`` (the ground
        truth, which replay restricts to the recorded contents).
        """

        def __init__(self, *, object_find_prob_fn: ProbFn, **kwargs) -> None:
            self._object_find_prob_fn = object_find_prob_fn
            super().__init__(**kwargs)

        def define_operators(self) -> list[Operator]:
            return [
                operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0),
                operators.construct_move_operator_blocking(self.estimate_move_time),
                operators.construct_search_operator(self._object_find_prob_fn, 10.0),
            ]

    # ------------------------------------------------------------------
    # Setup: known scene, robots, target
    # ------------------------------------------------------------------

    if num_robots < 1:
        raise ValueError("num_robots must be >= 1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    robots = [f"robot{i + 1}" for i in range(num_robots)]

    scene = ProcTHORScene(seed=seed)
    containers = sorted(scene.object_locations.keys())
    targets = pick_two_objects(scene.object_locations)
    goal = F(f"found {targets[0]}") & F(f"found {targets[1]}")
    true_containers = {
        o: next(c for c, objs in scene.object_locations.items() if o in objs)
        for o in targets
    }

    print(f"Grid: {scene.grid.shape[0]}x{scene.grid.shape[1]}  #containers={len(containers)}  "
          f"targets={targets}  true_containers={true_containers}  robots={robots}")

    # ------------------------------------------------------------------
    # Deployment: informed policy (knows which container truly holds each object)
    # ------------------------------------------------------------------

    def informed(robot: str, loc: str, obj: str) -> float:
        return 0.8 if obj in scene.object_locations.get(loc, set()) else 0.1

    fluents = {F("revealed start_loc")}
    for robot in robots:
        fluents |= {F(f"at {robot} start_loc"), F(f"free {robot}")}

    dep_env = SearchProcTHOREnvironment(
        object_find_prob_fn=informed,
        seed=seed,
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": set(containers) | {"start_loc"},
            "object": set(targets),
        },
        validate=False,
    )
    start = scene.locations["start_loc"]
    run_dashboard_loop(
        dep_env,
        goal,
        select=mcts_selector(MCTS),
        max_iterations=MAX_ITERS,
        fluent_keywords=("at", "found", "searched"),
        scene=scene,
        save_video=str(OUT_DIR / "deployment.mp4"),
        label="Deployment (informed)",
    )
    dep_cost = float(dep_env.state.time)
    searched = sorted({f.args[0] for f in dep_env.state.fluents if f.name == "searched"})
    print(f"deployment: found={goal.evaluate(dep_env.state.fluents)} "
          f"cost={dep_cost:.1f}s searched={searched}")

    # ------------------------------------------------------------------
    # Record via the package recorder (known grid + inspected contents)
    # ------------------------------------------------------------------

    log = build_rollout_log(
        dep_env,
        goal_cell=(int(start[0]), int(start[1])),
        robot_starts={
            robot: Pose(float(start[0]), float(start[1]), 0.0) for robot in robots
        },
        env_name="procthor",
        seed=seed,
        problem_class="known-map-search",
        goal=goal,
    )

    # ------------------------------------------------------------------
    # Replay three candidate policies (uniform vs. optimistic vs. cautious belief)
    # ------------------------------------------------------------------

    # container_find_prob only steers MCTS belief — outcomes resolve from the
    # recorded contents. Each candidate replays on a fresh arena, same MCTS, and
    # renders its own video.
    def _policy(name: str, prob: float) -> CandidatePolicy:
        return CandidatePolicy(name=name, container_find_prob=lambda r, l, o: prob)

    policies = {
        "uniform": _policy("uniform", 0.5),
        "optimistic": _policy("optimistic", 0.9),
        "cautious": _policy("cautious", 0.1),
    }

    results = {
        name: run_replay(
            build_replay_env(log),
            policy,
            dashboard=True,
            scene=scene,
            save_video=str(OUT_DIR / f"replay_{name}.mp4"),
            label=f"Replay ({name})",
            mcts=MCTS,
            max_planning_iterations=MAX_ITERS,
        )
        for name, policy in policies.items()
    }
    ranked = sorted(results.items(), key=lambda kv: kv[1].bounds.simply_connected_lb)

    # ------------------------------------------------------------------
    # Bounds (seconds / makespan)
    # ------------------------------------------------------------------

    # Commit-based: a container the deployment did not search is an unverified
    # subgoal (we do not assume one container per object), so searching it commits
    # (optimistic_to_goal=0). C_opt is the min over those commits; it collapses
    # onto C_sc only if the deployment searched every container.
    print("\n================ RESULTS ================")
    print(f"deployment (informed) cost   = {dep_cost:.1f}s")
    print("========== POLICY COMPARISON (replayed over one deployment) ==========")
    for name, res in ranked:
        print(f"  {name:12s}  C_sc={res.bounds.simply_connected_lb:7.1f}  "
              f"C_opt={res.bounds.optimistic_lb:7.1f}  found={res.goal_reached}")
    print("\nvideos: deployment.mp4 + "
          + ", ".join(f"replay_{n}.mp4" for n in policies))
    print(f"saved to {OUT_DIR}/")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Known-map object-search replay on ProcTHOR."
    )
    parser.add_argument("--seed", type=int, default=1089, help="ProcTHOR scene seed")
    parser.add_argument(
        "--num-robots", type=int, default=2,
        help="robots deployed (all start at start_loc; any finding the object wins)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(seed=args.seed, num_robots=args.num_robots)

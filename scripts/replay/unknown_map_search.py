"""Offline-replay demonstration for OBJECT SEARCH in an unknown environment.

Built on the example pattern (see ``examples/frontier_search.py``): the
planning task is run the standard way — ``UnknownSpaceEnvironment`` + the
unknown-search operators + an ``MCTSPlanner`` loop + ``PlannerDashboard`` for
the video. Nothing about *running the planner* is bespoke.

This drives the **real replay package API** end to end on a ProcTHOR scene:

    build_rollout_log(dep_env)      # records map + searched container contents
        -> build_replay_env(log)    # reconstruct the policy-agnostic arena
        -> run_replay(env, policy)  # replay a candidate -> cost bounds

Runs:

1. Deployment (``deployment.mp4``): an *informed* policy (decisive find-prob at
   the container that truly holds each object) searches the unknown ProcTHOR
   scene for a two-object goal (``found A & found B``, objects in distinct
   containers) and finds both. Its trajectory + the contents of every container
   it inspected are recorded into a ``RolloutLog`` by ``build_rollout_log``.
2. Replay (``replay_<policy>.mp4``): three candidate policies (uniform vs.
   optimistic vs. cautious belief) are each replayed over that recording with the
   package replay env, which confines the robot to the observed map and resolves
   every search outcome from the recorded ground truth. Each renders its own video
   and reports its simply-connected (actual makespan) cost vs the optimistic lower
   bound — both in deployment units (seconds), so they compare with the deployment.

Deployment and replay share one plan->act loop and one ``MctsConfig`` (``MCTS``).

Usage:  uv run python scripts/replay/unknown_map_search.py [--seed S] [--num-robots N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from railroad.replay import MctsConfig

if TYPE_CHECKING:
    import numpy as np
    from railroad.environment.procthor import ProcTHORScene

OUT_DIR = Path("data/replay/unknown_map_search")

# One planner config shared by the deployment and the replay.
MCTS = MctsConfig(iterations=4000, c=300.0, max_depth=20, heuristic_multiplier=2.0)
MAX_ITERS = 80


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


def main(seed: int = 1089, num_robots: int = 2) -> None:
    """Deploy an informed policy, record it, and replay a naive policy."""
    import numpy as np

    from railroad._bindings import State
    from railroad.core import Fluent as F
    from railroad.environment.symbolic import LocationRegistry
    from railroad.environment.skill import NavigationMoveSkill
    from railroad.environment.types import Pose
    from railroad.experimental.unknown_search import (
        NavigationConfig,
        UnknownSpaceEnvironment,
    )
    from railroad.experimental.unknown_search.operators import (
        construct_move_navigable_operator,
        construct_search_at_site_operator,
        construct_search_frontier_operator,
    )
    from railroad.operators import construct_no_op_operator
    from railroad.replay import (
        CandidatePolicy,
        build_replay_env,
        build_rollout_log,
        mcts_selector,
        run_dashboard_loop,
        run_replay,
    )

    # ------------------------------------------------------------------
    # Setup: scene, robots, target
    # ------------------------------------------------------------------

    if num_robots < 1:
        raise ValueError("num_robots must be >= 1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    robots = [f"robot{i + 1}" for i in range(num_robots)]
    scene, true_grid, hidden_sites, true_obj_locs, start = setup_scene(seed)
    targets = pick_two_objects(true_obj_locs)
    goal = F(f"found {targets[0]}") & F(f"found {targets[1]}")
    true_containers = {
        o: next(c for c, objs in true_obj_locs.items() if o in objs) for o in targets
    }

    print(f"Grid: {true_grid.shape[0]}x{true_grid.shape[1]}  targets={targets}  "
          f"true_containers={true_containers}")
    print(f"Containers: {list(hidden_sites)}  start={start}  robots={robots}")

    # ------------------------------------------------------------------
    # Deployment: informed policy (knows where the object likely is)
    # ------------------------------------------------------------------

    # Decisive (1/0) so the deployment finds the object only at its true
    # container — no false positives that would skip revealing it.
    def informed_prob(robot: str, loc: str, obj: str) -> float:
        return 1.0 if obj in true_obj_locs.get(loc, set()) else 0.0

    # The move operator's time function needs the env, which doesn't exist yet;
    # defer the binding through env_ref and use the env's safe estimator.
    env_ref: list = [None]

    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        if env_ref[0] is None:
            return 5.0
        return env_ref[0].estimate_move_time_safe(robot, loc_from, loc_to)

    operators = [
        construct_move_navigable_operator(move_time_fn),
        construct_search_frontier_operator(
            object_find_prob=lambda r, f, o: 0.5, search_time=20.0
        ),
        construct_search_at_site_operator(
            informed_prob, search_time=20.0, container_type="container"
        ),
        construct_no_op_operator(no_op_time=300.0, extra_cost=100.0),
    ]

    fluents: set = set()
    robot_initial_poses: dict = {}
    for robot in robots:
        fluents |= {F(f"at {robot} start_loc"), F(f"free {robot}"), F("revealed start_loc")}
        robot_initial_poses[robot] = Pose(float(start[0]), float(start[1]), 0.0)

    dep_env = UnknownSpaceEnvironment(
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": {"start_loc"},
            "container": set(),
            "frontier": set(),
            "object": set(targets),
        },
        operators=operators,
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
    env_ref[0] = dep_env
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
    print(f"deployment cost={dep_cost:.1f}s found={goal.evaluate(dep_env.state.fluents)}")

    # ------------------------------------------------------------------
    # Record via the package recorder (captures map + searched contents)
    # ------------------------------------------------------------------

    log = build_rollout_log(
        dep_env,
        goal_cell=(int(start[0]), int(start[1])),
        robot_starts={
            robot: Pose(float(start[0]), float(start[1]), 0.0) for robot in robots
        },
        env_name="procthor",
        seed=seed,
        problem_class="object-search",
        goal=goal,
    )
    print(f"recorded: containers={[s.signature for s in log.subgoals]} "
          f"actual_total_cost={log.actual_total_cost:.1f}s")

    # ------------------------------------------------------------------
    # Replay three candidate policies (uniform vs. optimistic vs. cautious belief)
    # ------------------------------------------------------------------

    # build_replay_env reconstructs the arena (hidden_sites + recorded contents +
    # goal come straight from the log); run_replay applies the candidate. The
    # find-probabilities only steer MCTS belief — outcomes resolve from the record.
    def _policy(name: str, prob: float) -> CandidatePolicy:
        return CandidatePolicy(
            name=name,
            frontier_find_prob=lambda r, f, o: prob,
            container_find_prob=lambda r, l, o: prob,
        )

    policies = {
        "uniform": _policy("uniform", 0.5),
        "optimistic": _policy("optimistic", 0.9),
        "cautious": _policy("cautious", 0.1),
    }

    # A fresh arena per candidate, same MCTS; each renders its own replay video.
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
    # Bounds (all in deployment units: seconds / makespan)
    # ------------------------------------------------------------------

    # Commit-based, exactly as in navigation: each not-found search is a commit
    # to a subgoal the deployment never verified; optimistically the object is
    # immediately at/past it, so C_opt = min over those commit times.
    print("\n================ RESULTS ================")
    print(f"deployment (informed) cost/time = {dep_cost:.1f}s")
    print("========== POLICY COMPARISON (replayed over one deployment) ==========")
    for name, res in ranked:
        print(f"  {name:12s}  C_sc={res.bounds.simply_connected_lb:7.1f}  "
              f"C_opt={res.bounds.optimistic_lb:7.1f}  found={res.goal_reached}")
    print("\nvideos: deployment.mp4 + "
          + ", ".join(f"replay_{n}.mp4" for n in policies))
    print(f"saved to {OUT_DIR}/")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline-replay demo for object search in an unknown ProcTHOR scene."
    )
    parser.add_argument("--seed", type=int, default=1089, help="ProcTHOR scene seed")
    parser.add_argument(
        "--num-robots", type=int, default=2,
        help="robots deployed (co-located at start; any finding the object wins)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(seed=args.seed, num_robots=args.num_robots)

"""Offline-replay demonstration for OBJECT SEARCH in an unknown environment.

Built on the example pattern (see ``examples/frontier_search.py``): the
planning task is run the standard way — ``UnknownSpaceEnvironment`` + the
unknown-search operators + an ``MCTSPlanner`` loop + ``PlannerDashboard`` for
the video. Nothing about *running the planner* is bespoke.

This drives the **real replay package API** end to end on a ProcTHOR scene:

    build_rollout_log(dep_env)            # records map + container contents
        -> SearchReplayEnvironment.from_log / build_search_replay_env
        -> the standard plan->act loop over the recording

Two runs:

1. Deployment (``deployment.mp4``): an *informed* policy (decisive find-prob at
   the container that truly holds the object) searches the unknown ProcTHOR
   scene and finds the target. Its trajectory + the contents of every container
   it inspected are recorded into a ``RolloutLog`` by ``build_rollout_log``.
2. Replay (``replay.mp4``): a *naive* policy (uniform find-prob) is replayed
   over that recording with the package ``SearchReplayEnvironment``, which
   confines the robot to the observed map and resolves every search outcome from
   the recorded ground truth. We report its simply-connected (actual makespan)
   cost vs the optimistic lower bound (straight to the true container) — both in
   deployment units (seconds), so they compare directly with the deployment.

Usage:  uv run python scripts/replay/unknown_map_search.py [--seed S] [--num-robots N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from railroad.environment.procthor import ProcTHORScene

OUT_DIR = Path("data/replay/unknown_map_search")


# ----------------------------------------------------------------------
# Scene setup (mirrors examples/frontier_search.py::_setup_procthor)
# ----------------------------------------------------------------------


def setup_scene(seed: int, target_object: str | None = None) -> tuple[
    "ProcTHORScene",
    "np.ndarray",
    dict[str, tuple[int, int]],
    dict[str, set[str]],
    tuple[int, int],
    str,
]:
    """Load a ProcTHOR scene and extract grid, hidden sites, and the target."""
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

    if target_object is None:
        # Pick an object that sits in a container away from the start, so the
        # search is non-trivial.
        all_objs = sorted({o for objs in true_object_locations.values() for o in objs})
        target_object = all_objs[0]
    return scene, true_grid, hidden_sites, true_object_locations, start_coord, target_object


# ----------------------------------------------------------------------
# Planning loop (standard MCTS loop with PlannerDashboard) — example pattern
# ----------------------------------------------------------------------


def run_planning(env, goal, label: str, save_video: str, *, max_iterations: int = 80) -> float:
    """Drive one MCTS plan->act loop with a dashboard; return the makespan."""
    from railroad.core import get_action_by_name
    from railroad.dashboard import PlannerDashboard
    from railroad.planner import MCTSPlanner

    def fluent_filter(f):  # noqa: ANN001
        return any(kw in f.name for kw in ["at", "found", "searched"])

    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        act_callback = dashboard.make_act_callback()
        dashboard.console.print(f"[bold]{label}[/bold]")
        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Object found![/green]")
                break
            actions = env.get_actions()
            if not actions:
                dashboard.console.print("[red]No actions available — stuck.[/red]")
                break
            mcts = MCTSPlanner(actions)
            action_name = mcts(
                env.state, goal,
                max_iterations=4000, c=300, max_depth=20, heuristic_multiplier=2,
            )
            if action_name == "NONE":
                dashboard.console.print("[yellow]Planner returned NONE — stopping.[/yellow]")
                break
            action = get_action_by_name(actions, action_name)
            env.act(action, loop_callback_fn=act_callback)
            dashboard.update(mcts, action_name)

    dashboard.show_plots(save_video=save_video, video_fps=10, video_dpi=130)
    return float(env.state.time)


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
    from railroad.replay import build_rollout_log
    from railroad.replay.cost import accumulate_bounds
    from railroad.replay.search_replay_env import build_search_replay_env

    # ------------------------------------------------------------------
    # Setup: scene, robots, target
    # ------------------------------------------------------------------

    if num_robots < 1:
        raise ValueError("num_robots must be >= 1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    robots = [f"robot{i + 1}" for i in range(num_robots)]
    scene, true_grid, hidden_sites, true_obj_locs, start, target = setup_scene(seed)
    true_container = next(c for c, objs in true_obj_locs.items() if target in objs)
    goal = F(f"found {target}")

    print(f"Grid: {true_grid.shape[0]}x{true_grid.shape[1]}  target={target}  "
          f"true_container={true_container}")
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
            "object": {target},
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
    dep_env.scene = scene  # type: ignore[attr-defined]  # expose to dashboard for overhead map
    env_ref[0] = dep_env
    dep_cost = run_planning(
        dep_env, goal, "Deployment (informed policy)", str(OUT_DIR / "deployment.mp4")
    )
    print(f"deployment cost={dep_cost:.1f}s found={goal.evaluate(dep_env.state.fluents)}")

    # ------------------------------------------------------------------
    # Record via the package recorder (captures map + container contents)
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
    )
    print(f"recorded: containers={[s.signature for s in log.subgoals]} "
          f"actual_total_cost={log.actual_total_cost:.1f}s")

    # ------------------------------------------------------------------
    # Replay: naive policy (uniform belief), exact outcomes from the record
    # ------------------------------------------------------------------

    # Built with the package SearchReplayEnvironment: hidden_sites + recorded
    # contents come straight from the log's container subgoals.
    hidden = {s.signature: (int(s.centroid[0]), int(s.centroid[1])) for s in log.subgoals}
    recorded = {s.signature: set(s.contents) for s in log.subgoals}

    rep_env = build_search_replay_env(
        log,
        frontier_find_prob=lambda r, f, o: 0.5,
        container_find_prob=lambda r, l, o: 0.5,
        hidden_sites=hidden,
        target_object=target,
        recorded_object_locations=recorded,
    )
    rep_env.scene = scene  # type: ignore[attr-defined]  # expose to dashboard for overhead map
    rep_cost = run_planning(
        rep_env, goal, "Replay (naive policy)", str(OUT_DIR / "replay.mp4")
    )

    # ------------------------------------------------------------------
    # Bounds (all in deployment units: seconds / makespan)
    # ------------------------------------------------------------------

    # Commit-based, exactly as in navigation: each not-found search is a commit
    # to a subgoal the deployment never verified; optimistically the object is
    # immediately at/past it, so optimistic_lb = min over those commit times.
    sc = float(rep_env.state.time)
    bounds = accumulate_bounds(rep_env.replay_commits, sc)
    searched = [loc for loc, _, _ in rep_env.search_log]
    print("\n================ RESULTS ================")
    print(f"deployment (informed) cost/time = {dep_cost:.1f}s")
    print(f"replay (naive) found={goal.evaluate(rep_env.state.fluents)} "
          f"cost/time = {rep_cost:.1f}s; searched {searched}")
    print(
        f"C_sc (simply-connected: naive policy's exact replay makespan) = {sc:.1f}s\n"
        f"C_opt (optimistic: object at the candidate's earliest unverified "
        f"commit) = {bounds.optimistic_lb:.1f}s"
    )
    print(f"saved videos to {OUT_DIR}/deployment.mp4 and {OUT_DIR}/replay.mp4")


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

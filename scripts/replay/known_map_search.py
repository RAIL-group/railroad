"""Known-map object-search replay on ProcTHOR.

Unlike scripts/replay/unknown_map_search.py (an *unknown* environment: the robot
must explore frontiers and the observed map is partial), here the **map is fully
known** — the ProcTHOR floorplan is given — and only the objects' *presence in
containers* is unknown. This mirrors examples/procthor_search.py (a
``ProcTHOREnvironment`` with ``move`` + ``search`` operators), the standard
known-map search task.

Pipeline:

1. Deployment (``deployment.mp4``): an *informed* policy (high find-prob at the
   container that truly holds the target) searches known containers until it
   finds the object. We record which containers it inspected and their contents.
2. Replay (``replay.mp4``): a *naive* policy (uniform belief) is replayed on the
   SAME known map. Its only knowledge of object presence is the recording —
   realized by restricting the replay env's ``_objects_at_locations`` to the
   recorded contents, so the existing deterministic search resolution becomes
   exact replay from the recording. Travel is exact (map known) and,
   because the deployment revealed the truth, the counterfactual cost is exact —
   not just a lower bound.

Costs are makespan (``state.time``, seconds), so deployment and replay compare
directly. ``C_opt`` is the optimal "straight to the true container" cost.

Usage:  uv run python scripts/replay/known_map_search.py [--seed S] [--num-robots N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

OUT_DIR = Path("data/replay/known_map_search")

ProbFn = Callable[[str, str, str], float]


# ----------------------------------------------------------------------
# Planning loop (standard MCTS loop with PlannerDashboard) — example pattern
# ----------------------------------------------------------------------


def run_planning(env, goal, label: str, save_video: str, *, max_iterations: int = 40) -> float:
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


def main(seed: int = 4001, num_robots: int = 2) -> None:
    """Deploy an informed policy on a known map, record it, and replay a naive one."""
    from railroad.environment.procthor import ProcTHOREnvironment, ProcTHORScene

    from railroad import operators
    from railroad._bindings import State
    from railroad.core import Fluent as F, Operator
    from railroad.environment.types import Pose
    from railroad.replay import (
        build_known_map_search_log,
        build_known_map_search_replay_env,
    )
    from railroad.replay.cost import accumulate_bounds

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
    target = sorted({o for objs in scene.object_locations.values() for o in objs})[0]
    true_location = next(c for c, objs in scene.object_locations.items() if target in objs)
    goal = F(f"found {target}")

    print(f"Grid: {scene.grid.shape[0]}x{scene.grid.shape[1]}  #containers={len(containers)}  "
          f"target={target}  true_location={true_location}  robots={robots}")

    # ------------------------------------------------------------------
    # Deployment: informed policy (knows where the target likely is)
    # ------------------------------------------------------------------

    def informed(robot: str, loc: str, obj: str) -> float:
        return 0.8 if loc == true_location else 0.1

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
            "object": {target},
        },
        validate=False,
    )
    dep_cost = run_planning(
        dep_env, goal, "Deployment (informed)", str(OUT_DIR / "deployment.mp4")
    )
    searched = sorted({f.args[0] for f in dep_env.state.fluents if f.name == "searched"})
    print(f"deployment: found={goal.evaluate(dep_env.state.fluents)} "
          f"cost={dep_cost:.1f}s searched={searched}")

    # ------------------------------------------------------------------
    # Record via the package recorder (known grid + inspected contents)
    # ------------------------------------------------------------------

    start = scene.locations["start_loc"]
    log = build_known_map_search_log(
        dep_env,
        robot_starts={
            robot: Pose(float(start[0]), float(start[1]), 0.0) for robot in robots
        },
        env_name="procthor",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Replay: naive policy, built by the package from the recording
    # ------------------------------------------------------------------

    # Outcomes resolve from the recorded contents (run_known_map_search_replay is
    # the headless equivalent of this loop + bounds).
    def naive(robot: str, loc: str, obj: str) -> float:
        return 0.5

    rep_env = build_known_map_search_replay_env(
        log, container_find_prob=naive, target_object=target
    )
    rep_env.scene = scene  # type: ignore[attr-defined]  # expose to dashboard for overhead map
    rep_cost = run_planning(
        rep_env, goal, "Replay (naive)", str(OUT_DIR / "replay.mp4")
    )
    rep_searched = [f.args[0] for f in rep_env.state.fluents if f.name == "searched"]

    # ------------------------------------------------------------------
    # Bounds (seconds / makespan)
    # ------------------------------------------------------------------

    # Commit-based: a container the deployment did not search is an unverified
    # subgoal (we do not assume one container per object), so searching it commits
    # (optimistic_to_goal=0). C_opt is the min over those commits; it collapses
    # onto C_sc only if the deployment searched every container.
    bounds = accumulate_bounds(rep_env.replay_commits, rep_cost)
    c_oracle = rep_env.estimate_move_time("robot1", "start_loc", true_location) + 10.0
    print("\n================ RESULTS ================")
    print(f"deployment (informed) cost   = {dep_cost:.1f}s")
    print(f"replay (naive) found={goal.evaluate(rep_env.state.fluents)} "
          f"cost = {rep_cost:.1f}s; searched {sorted(set(rep_searched))}")
    print(f"C_sc (naive policy's replay makespan)        = {rep_cost:.1f}s")
    print(f"C_opt (min over unsearched-container commits) = {bounds.optimistic_lb:.1f}s")
    print(f"(oracle straight-to-true-container baseline    = {c_oracle:.1f}s)")
    print(f"saved videos to {OUT_DIR}/deployment.mp4 and {OUT_DIR}/replay.mp4")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Known-map object-search replay on ProcTHOR."
    )
    parser.add_argument("--seed", type=int, default=4001, help="ProcTHOR scene seed")
    parser.add_argument(
        "--num-robots", type=int, default=2,
        help="robots deployed (all start at start_loc; any finding the object wins)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(seed=args.seed, num_robots=args.num_robots)

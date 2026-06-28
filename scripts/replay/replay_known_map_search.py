"""Known-map object-search replay on ProcTHOR (design doc §7 / §7.1).

Unlike scripts/replay/replay_object_search.py (an *unknown* environment: the robot must
explore frontiers and the observed map is partial), here the **map is fully
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
   exact replay from the recording (§7.1). Travel is exact (map known) and,
   because the deployment revealed the truth, the counterfactual cost is exact —
   not just a lower bound.

Costs are makespan (``state.time``, seconds), so deployment and replay compare
directly. ``C_opt`` is the optimal "straight to the true container" cost.

Usage:  uv run python scripts/replay/replay_known_map_search.py [seed]
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Set

from railroad import operators
from railroad._bindings import State
from railroad.core import Fluent as F, Operator, get_action_by_name
from railroad.dashboard import PlannerDashboard
from railroad.environment.procthor import ProcTHOREnvironment
from railroad.environment.types import Pose
from railroad.planner import MCTSPlanner
from railroad.replay import (
    build_known_map_search_log,
    build_known_map_search_replay_env,
)

OUT_DIR = Path("data/replay_known_map_search")

ProbFn = Callable[[str, str, str], float]


class SearchProcTHOREnvironment(ProcTHOREnvironment):
    """Known-map ProcTHOR search env: ``move`` + ``search`` over known locations.

    The find-probability callable is the swappable "policy" (informed vs naive);
    it only drives MCTS belief — the actual search outcome is resolved
    deterministically from ``_objects_at_locations`` (the ground truth, which
    replay restricts to the recorded contents).
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


def build_env(seed: int, target: str, locations: Set[str], prob_fn: ProbFn):
    fluents = {F("revealed start_loc"), F("at robot1 start_loc"), F("free robot1")}
    return SearchProcTHOREnvironment(
        object_find_prob_fn=prob_fn,
        seed=seed,
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": {"robot1"},
            "location": set(locations) | {"start_loc"},
            "object": {target},
        },
        validate=False,
    )


def run_planning(env, goal, label, video, *, max_iterations=40) -> float:
    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "found", "searched"])

    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dash:
        cb = dash.make_act_callback()
        dash.console.print(f"[bold]{label}[/bold]")
        for step in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dash.console.print("[green]Object found![/green]")
                break
            actions = env.get_actions()
            if not actions:
                break
            mcts = MCTSPlanner(actions)
            name = mcts(env.state, goal, max_iterations=4000, c=300, max_depth=20,
                        heuristic_multiplier=2)
            if name == "NONE":
                break
            print(f"  [{label}] step {step}: {name} (t={env.state.time:.1f})", flush=True)
            env.act(get_action_by_name(actions, name), loop_callback_fn=cb)
            dash.update(mcts, name)
        dash.show_plots(save_video=video, video_fps=10, video_dpi=130)
    return float(env.state.time)


def main(seed: int = 4001) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Known map: pick a target and the container that truly holds it. Searchable
    # locations are the scene's object-bearing containers (a realistic domain).
    from railroad.environment.procthor import ProcTHORScene

    scene = ProcTHORScene(seed=seed)
    containers = sorted(scene.object_locations.keys())
    target = sorted({o for objs in scene.object_locations.values() for o in objs})[0]
    true_location = next(c for c, objs in scene.object_locations.items() if target in objs)
    print(f"seed={seed} grid={scene.grid.shape} #containers={len(containers)} "
          f"target={target} true_location={true_location}")

    goal = F(f"found {target}")

    # ---- Deployment: informed policy (knows where the target likely is). ----
    def informed(robot, loc, obj):
        return 0.8 if loc == true_location else 0.1

    dep_env = build_env(seed, target, set(containers), informed)
    dep_cost = run_planning(dep_env, goal, "Deployment (informed)",
                            str(OUT_DIR / "deployment.mp4"))
    searched = sorted({f.args[0] for f in dep_env.state.fluents if f.name == "searched"})
    print(f"deployment: found={goal.evaluate(dep_env.state.fluents)} "
          f"cost={dep_cost:.1f}s searched={searched}")

    # ---- Record via the package recorder (known grid + inspected contents). ----
    start = scene.locations["start_loc"]
    log = build_known_map_search_log(
        dep_env,
        robot_starts={"robot1": Pose(float(start[0]), float(start[1]), 0.0)},
        env_name="procthor",
        seed=seed,
    )

    # ---- Replay: naive policy, built by the package from the recording. ----
    # Outcomes resolve from the recorded contents (run_known_map_search_replay is
    # the headless equivalent of this loop + bounds).
    def naive(robot, loc, obj):
        return 0.5

    rep_env = build_known_map_search_replay_env(
        log, container_find_prob=naive, target_object=target
    )
    rep_env.scene = scene  # type: ignore[attr-defined]  # for dashboard overhead
    rep_cost = run_planning(rep_env, goal, "Replay (naive)",
                            str(OUT_DIR / "replay.mp4"))
    rep_searched = [f.args[0] for f in rep_env.state.fluents if f.name == "searched"]

    # ---- Bounds (seconds / makespan). Map known -> exact; truth revealed ->
    #      C_sc is the naive policy's EXACT counterfactual cost, not a bound. ----
    c_opt = rep_env.estimate_move_time("robot1", "start_loc", true_location) + 10.0
    print("\n================ RESULTS ================")
    print(f"deployment (informed) cost   = {dep_cost:.1f}s")
    print(f"replay (naive) found={goal.evaluate(rep_env.state.fluents)} "
          f"cost = {rep_cost:.1f}s; searched {sorted(set(rep_searched))}")
    print(f"C_sc (naive policy's exact replay makespan) = {rep_cost:.1f}s")
    print(f"C_opt (straight to the true container)      = {c_opt:.1f}s")
    print(f"saved videos to {OUT_DIR}/deployment.mp4 and {OUT_DIR}/replay.mp4")


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4001)

#!/usr/bin/env python
"""
Live-coding tutorial for the railroad planner (Brown University).

Edit ONLY the `tutorial_main` function and its `add_cases([...])` sweep below.
Everything else is harness plumbing you can ignore during the talk.

Run it two ways:

    # 1. Single run with the live interactive dashboard (terminal TUI)
    uv run python scripts/tutorial_brown.py

    # 2. Benchmark sweep: the SAME function, many repeats + a parameter sweep,
    #    logged to one persistent MLflow experiment. --label is REQUIRED.
    uv run python scripts/tutorial_brown.py --bench --label demo-v1 --repeat 3

Then view/compare benchmark runs in the browser:

    uv run railroad benchmarks dashboard
    # open the "tutorial_brown_university" experiment

Comparison semantics:
  * Each --bench run registers as benchmark "tutorial::<label>".
  * A NEW --label  -> a new violin group, side-by-side with the others.
  * The SAME --label re-run (e.g. after expanding the sweep) -> the same group,
    just with the additional parameter points. All runs accumulate in the one
    "tutorial_brown_university" experiment, so refreshing the page shows them.
"""

import argparse
import os
import sys
import time

# Make the in-repo `railroad` package importable when run as a plain script.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "packages", "railroad", "src"),
)

from rich.console import Console

from railroad.bench.registry import Benchmark, BenchmarkCase, get_all_benchmarks
from railroad.dashboard import PlannerDashboard

#: Stable MLflow experiment all --bench runs accumulate into.
EXPERIMENT_NAME = "tutorial_brown_university"

#: Env var used to (a) signal "bench mode" and (b) carry --label into workers,
#: which re-import this file and must register the identical "tutorial::<label>".
LABEL_ENV_VAR = "RAILROAD_TUTORIAL_LABEL"


class Tutorial:
    """Wraps the editable function: adds `.add_cases()` and the run harness.

    Mirrors the familiar `@benchmark` / `.add_cases([...])` API, but defers
    registration because the benchmark name depends on the runtime `--label`.
    """

    def __init__(self, user_fn, description="", repeat=8, timeout=120.0):
        self.user_fn = user_fn
        self.description = description
        self.repeat = repeat
        self.timeout = timeout
        self.cases: list[dict] = []
        self._label = None  # set by --label
        self._no_media = False  # set by --no-media in single mode
        self.__doc__ = user_fn.__doc__
        self.__name__ = user_fn.__name__

    # -- familiar benchmark-style API ------------------------------------
    def add_cases(self, cases: list[dict]) -> None:
        """Register parameter combinations for the sweep (like @benchmark)."""
        self.cases.extend(cases)

    # -- dual-console wrapper that actually goes into the registry --------
    def _bench_fn(self, case: BenchmarkCase) -> dict:
        """Registered callable. Injects `case.make_dashboard` then runs user code.

        Single mode (no label env var): interactive live TUI dashboard.
        Bench mode: a recording console, auto-harvested into the `log_html`
        artifact the web dashboard renders.
        """
        bench_mode = LABEL_ENV_VAR in os.environ
        recorder: dict = {}

        def make_dashboard(goal, env, *, location_coords=None, **kw):
            # location_coords is consumed here (PlannerDashboard.__init__ does
            # not accept it) and applied at plot/video render time. It is
            # required for SymbolicEnvironment, whose env-derived coords are
            # empty, so without it the trajectory plot is mislocated.
            recorder["location_coords"] = location_coords
            # print_on_exit=True so the dashboard's __exit__ (i.e. the
            # `with case.make_dashboard(...) as dashboard:` block in
            # tutorial_main) tears down the live screen and prints the final
            # history. The `with` form is what actually starts the Rich Live
            # view; constructing without it shows nothing.
            if bench_mode:
                console = Console(record=True, force_terminal=True, width=120)
                recorder["console"] = console
                dashboard = PlannerDashboard(
                    goal, env, console=console, print_on_exit=True, **kw
                )
            else:
                # Single mode is for the live talk: force the interactive TUI
                # so the panels render regardless of headless auto-detection.
                kw.setdefault("force_interactive", True)
                dashboard = PlannerDashboard(goal, env, print_on_exit=True, **kw)
            recorder["dashboard"] = dashboard
            return dashboard

        case.make_dashboard = make_dashboard
        result = self.user_fn(case)

        if not isinstance(result, dict):
            return result

        dashboard = recorder.get("dashboard")
        location_coords = recorder.get("location_coords")
        if bench_mode:
            if "console" in recorder and "log_html" not in result:
                result["log_html"] = recorder["console"].export_html(
                    inline_styles=True
                )
            # Trajectory image (plot.jpg artifact); None when no trajectory.
            if dashboard is not None and "log_plot" not in result:
                plot_image = dashboard.get_plot_image(
                    location_coords=location_coords
                )
                if plot_image is not None:
                    result["log_plot"] = plot_image
        elif dashboard is not None and not self._no_media:
            self._save_media(dashboard, location_coords)
        return result

    def _save_media(self, dashboard, location_coords) -> None:
        """Single/TUI mode: write the trajectory plot + 720p30 video to the
        shared media dir so they're viewable remotely at the dashboard's
        /media/. Named after --label, overwriting any previous run.
        """
        import re

        from railroad.bench.tutorial_media import media_dir

        d = media_dir()
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", self._label or "tutorial")
        base = d / safe
        try:
            dashboard.show_plots(
                save_plot=f"{base}.jpg",
                save_video=f"{base}.mp4",
                video_fps=30,
                video_dpi=100,  # 12.8x7.2in @ 100dpi = 1280x720 (720p)
                location_coords=location_coords,
            )
            print(
                f"\n[tutorial] saved {safe}.jpg / {safe}.mp4 (720p30) to {d}\n"
                "  View remotely:  uv run railroad benchmarks dashboard\n"
                "  then open  http://<host>:8050/media/"
            )
        except Exception as e:  # e.g. ffmpeg missing; don't kill the demo
            print(
                f"\n[tutorial] could not save media ({e}). "
                "Plot/video skipped; planning result is unaffected."
            )

    # -- benchmark registration (idempotent by name) ---------------------
    def _register(self, label: str) -> Benchmark:
        from railroad.bench import registry as _registry

        name = f"tutorial::{label}"
        for bench in get_all_benchmarks():
            if bench.name == name:
                return bench  # already registered (e.g. forked worker)
        # Benchmark() does not auto-register (only the @benchmark decorator
        # does), so append to the registry explicitly.
        bench = Benchmark(
            fn=self._bench_fn,
            name=name,
            description=self.description,
            tags=["tutorial"],
            timeout=self.timeout,
            repeat=self.repeat,
        )
        bench.add_cases(self.cases)
        _registry._BENCHMARKS.append(bench)
        return bench

    # -- CLI -------------------------------------------------------------
    def run_cli(self, argv=None) -> None:
        parser = argparse.ArgumentParser(description="railroad tutorial runner")
        parser.add_argument(
            "--bench",
            action="store_true",
            help="Run the benchmark sweep (repeats + parameter sweep) via MLflow.",
        )
        parser.add_argument(
            "--label",
            required=True,
            help="REQUIRED. Names this run. In --bench it is the comparison "
            "group (benchmark 'tutorial::<label>'); in single mode it names "
            "the saved <label>.jpg/.mp4 (overwriting any previous run).",
        )
        parser.add_argument(
            "--repeat", type=int, default=None, help="Cap repeats per case."
        )
        parser.add_argument(
            "--parallel", type=int, default=None, help="Worker processes."
        )
        parser.add_argument(
            "--filter", default=None, help="pytest-style case filter."
        )
        parser.add_argument(
            "--case",
            type=int,
            default=0,
            help="Single mode: which sweep case index to run (default 0).",
        )
        parser.add_argument(
            "--no-media",
            action="store_true",
            help="Single mode: skip saving the trajectory plot/video.",
        )
        args = parser.parse_args(argv)

        self._label = args.label
        self._no_media = args.no_media
        if args.bench:
            self._run_bench(args)
        else:
            self._run_single(args)

    def _run_single(self, args) -> None:
        params = self.cases[args.case] if self.cases else {}
        case = BenchmarkCase(
            benchmark_name="tutorial",
            case_idx=args.case,
            repeat_idx=0,
            params=dict(params),
        )
        result = self._bench_fn(case)
        result = result if isinstance(result, dict) else {}
        ok = result.get("success")
        cost = result.get("plan_cost")
        wall = result.get("wall_time")
        wall_str = f"{wall:.3f}s" if isinstance(wall, (int, float)) else str(wall)
        print(
            f"\n[tutorial] success={ok}  plan_cost={cost}  wall_time={wall_str}"
        )

    def _run_bench(self, args) -> None:
        # --label is enforced by argparse (required=True).
        # Propagate the label to re-imported worker processes, and register.
        os.environ[LABEL_ENV_VAR] = args.label
        bench = self._register(args.label)

        from railroad.bench.runner import BenchmarkRunner

        runner = BenchmarkRunner(
            benchmarks=[bench],
            repeat_max=args.repeat,
            parallel=args.parallel,
            case_filter=args.filter,
            include_files=[os.path.abspath(__file__)],
            experiment_name=EXPERIMENT_NAME,
            run_name=args.label,
        )
        plan = runner.create_plan()
        runner.run(plan)

        print(
            f"\n[tutorial] logged group 'tutorial::{args.label}' to experiment "
            f"'{EXPERIMENT_NAME}'.\n"
            "  View/compare:  uv run railroad benchmarks dashboard\n"
            f"  Then open the '{EXPERIMENT_NAME}' experiment."
        )


def tutorial(description: str = "", repeat: int = 8, timeout: float = 120.0):
    """Decorator: turn the editable function into a runnable Tutorial."""

    def decorator(fn) -> Tutorial:
        return Tutorial(fn, description=description, repeat=repeat, timeout=timeout)

    return decorator


# ======================================================================
# ====================  EDIT BELOW THIS LINE  ==========================
# ======================================================================


@tutorial(
    description="Heterogeneous robots search for supplies and deliver to base.",
    repeat=8,
    timeout=120.0,
)
def tutorial_main(case: BenchmarkCase) -> dict:
    """
    Heterogeneous multi-robot search-and-deliver.

    Three robots with different capabilities (rover/crawler can pick & place,
    drone can only search but moves twice as fast) must locate `supplies`
    somewhere in the map and bring them back to `start`. This runs many
    planning steps (search -> pick -> move -> place), so the live dashboard
    has plenty to show.

    Parameters come from `case` (see add_cases below):
      * case.mcts.iterations -- MCTS search budget per planning step
      * case.mcts.c          -- UCT exploration constant
    """
    import numpy as np

    from railroad import operators
    from railroad.core import Fluent as F, State, get_action_by_name
    from railroad.environment import SymbolicEnvironment
    from railroad.planner import MCTSPlanner

    # --- Map: location -> (x, y) --------------------------------------
    locations = {
        "start": np.array([0, 0]),
        "location1": np.array([10, 9]),
        "location2": np.array([9, 0]),
        "location3": np.array([1, 2]),
        "location4": np.array([1, 10]),
    }
    # Ground truth: the supplies are actually at location2.
    objects_at_locations = {loc: set() for loc in locations}
    objects_at_locations["location2"] = {"supplies"}

    # --- Heterogeneous robot capabilities -----------------------------
    skills_time = {
        "rover": {"pick": 10.0, "place": 10.0, "search": 10.0},
        "crawler": {"pick": 10.0, "place": 10.0, "search": 10.0},
        "drone": {"search": 10.0},  # drone can only search (plus move)
    }
    speed_multiplier = {"rover": 1.0, "crawler": 1.0, "drone": 2.0}
    robots = ["rover", "drone", "crawler"]

    def skill_time(skill):
        return lambda robot, *a, **k: skills_time.get(robot, {}).get(
            skill, float("inf")
        )

    def move_time(robot, loc_from, loc_to):
        if loc_from not in locations or loc_to not in locations:
            return float("inf")
        dist = float(np.linalg.norm(locations[loc_from] - locations[loc_to]))
        return dist / speed_multiplier.get(robot, 1.0)

    def find_prob(robot, loc, obj):
        return 0.9 if obj in objects_at_locations.get(loc, set()) else 0.1

    # --- Initial state & goal -----------------------------------------
    initial_fluents = {F("revealed", "start")}
    for robot in robots:
        initial_fluents.add(F("at", robot, "start"))
        initial_fluents.add(F("free", robot))
    goal = F("at supplies start")

    objects_by_type = {
        "robot": set(robots),
        "location": set(locations.keys()),
        "object": {"supplies"},
    }

    # --- Operators & environment --------------------------------------
    move_op = operators.construct_move_operator_blocking(move_time)
    search_op = operators.construct_search_operator(
        find_prob, skill_time("search")
    )
    pick_op = operators.construct_pick_operator_blocking(skill_time("pick"))
    place_op = operators.construct_place_operator_blocking(skill_time("place"))
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    env = SymbolicEnvironment(
        state=State(0.0, initial_fluents, []),
        objects_by_type=objects_by_type,
        operators=[no_op, move_op, search_op, pick_op, place_op],
        true_object_locations=objects_at_locations,
    )

    # --- Plan, with the dashboard the harness picked for this mode -----
    def fluent_filter(f):
        return any(
            kw in f.name
            for kw in ["at", "holding", "found", "searched", "free"]
        )

    # SymbolicEnvironment has no coords to give the plotter, so pass them
    # explicitly or the trajectory plot/video is mislocated.
    location_coords = {
        name: (float(c[0]), float(c[1])) for name, c in locations.items()
    }

    start_time = time.perf_counter()
    # `with` is essential: __enter__ starts the Rich Live view; without it
    # nothing renders. __exit__ tears it down and prints the final history.
    with case.make_dashboard(
        goal, env, fluent_filter=fluent_filter, location_coords=location_coords
    ) as dashboard:
        for _ in range(60):
            if goal.evaluate(env.state.fluents):
                break
            all_actions = env.get_actions()
            planner = MCTSPlanner(all_actions)
            action_name = planner(
                env.state,
                goal,
                max_iterations=case.mcts.iterations,
                c=case.mcts.c,
                max_depth=20,
            )
            if action_name == "NONE":
                break
            env.act(get_action_by_name(all_actions, action_name))
            dashboard.update(planner, action_name)

    wall_time = time.perf_counter() - start_time
    actions_taken = [name for name, _ in dashboard.actions_taken]
    return {
        "success": goal.evaluate(env.state.fluents),
        "wall_time": wall_time,
        "plan_cost": float(env.state.time),
        "actions_count": len(actions_taken),
        "actions": actions_taken,
    }


# Parameter sweep for --bench mode (single mode uses index --case, default 0).
# Case 0 (the single-mode default) is the robust budget so the live demo
# reliably solves; the rest sweep smaller/different budgets for comparison.
tutorial_main.add_cases(
    [
        {"mcts.iterations": 10000, "mcts.c": 300},
        {"mcts.iterations": 4000, "mcts.c": 300},
        {"mcts.iterations": 10000, "mcts.c": 100},
    ]
)


# ======================================================================
# ====================  EDIT ABOVE THIS LINE  ==========================
# ======================================================================


# Re-imported worker processes carry the label via the environment; register
# the identical "tutorial::<label>" benchmark so the worker can find it by name.
if os.environ.get(LABEL_ENV_VAR):
    tutorial_main._register(os.environ[LABEL_ENV_VAR])


if __name__ == "__main__":
    tutorial_main.run_cli()

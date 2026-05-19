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

        def make_dashboard(goal, env, **kw):
            if bench_mode:
                console = Console(record=True, force_terminal=True, width=120)
                recorder["console"] = console
                dashboard = PlannerDashboard(
                    goal, env, console=console, print_on_exit=False, **kw
                )
            else:
                dashboard = PlannerDashboard(goal, env, **kw)
            recorder["dashboard"] = dashboard
            return dashboard

        case.make_dashboard = make_dashboard
        result = self.user_fn(case)

        if bench_mode and isinstance(result, dict):
            if "console" in recorder and "log_html" not in result:
                result["log_html"] = recorder["console"].export_html(
                    inline_styles=True
                )
            # Trajectory image (plot.jpg artifact); None when no trajectory.
            if "dashboard" in recorder and "log_plot" not in result:
                plot_image = recorder["dashboard"].get_plot_image()
                if plot_image is not None:
                    result["log_plot"] = plot_image
        return result

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
            default=None,
            help="REQUIRED with --bench. Comparison group name (benchmark "
            "'tutorial::<label>'). Same label = same group + more points.",
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
        args = parser.parse_args(argv)

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
        if not args.label:
            print(
                "error: --label is required with --bench "
                "(it names the comparison group, benchmark 'tutorial::<label>').",
                file=sys.stderr,
            )
            raise SystemExit(2)

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
    description="Single robot navigates to a destination via move.",
    repeat=8,
    timeout=60.0,
)
def tutorial_main(case: BenchmarkCase) -> dict:
    """
    Single-robot navigation.

    The robot starts in the living room and must reach the furthest location.
    Parameters come from `case` (see add_cases below):
      * case.num_locations   -- how many waypoints to lay out
      * case.mcts.iterations -- MCTS search budget per planning step
    """
    import numpy as np

    from railroad import operators
    from railroad.core import Fluent as F, State, get_action_by_name
    from railroad.experimental.environment import (
        EnvironmentInterface,
        SimpleEnvironment,
    )
    from railroad.planner import MCTSPlanner

    num_locations = case.num_locations

    # --- Build the environment ----------------------------------------
    locations = {f"loc{i}": np.array([i * 2.0, 0.0]) for i in range(num_locations)}
    locations["living_room"] = np.array([0.0, 0.0])  # required by SimpleEnvironment
    locations["start"] = np.array([0.0, 0.0])

    env = SimpleEnvironment(
        locations=locations,
        objects_at_locations={},
        robot_locations={"robot1": "living_room"},
    )

    # --- Initial state & goal -----------------------------------------
    initial_state = State(
        time=0,
        fluents={F("at robot1 living_room"), F("free robot1")},
    )
    goal = F(f"at robot1 loc{num_locations - 1}")

    # --- Operators & simulator ----------------------------------------
    move_op = operators.construct_move_operator_blocking(
        env.get_skills_time_fn("move")
    )
    objects_by_type = {
        "robot": {"robot1"},
        "location": set(locations.keys()),
    }
    sim = EnvironmentInterface(
        initial_state, objects_by_type, [move_op], env
    )

    # --- Plan, with the dashboard the harness picked for this mode -----
    start_time = time.perf_counter()
    dashboard = case.make_dashboard(goal, sim)
    planner = MCTSPlanner(sim.get_actions())

    for _ in range(100):
        if goal.evaluate(sim.state.fluents):
            break
        action_name = planner(
            sim.state, goal, max_iterations=case.mcts.iterations, c=100
        )
        if action_name == "NONE":
            break
        sim.advance(get_action_by_name(sim.get_actions(), action_name))
        dashboard.update(planner, action_name)

    wall_time = time.perf_counter() - start_time
    dashboard.print_history()

    actions_taken = [name for name, _ in dashboard.actions_taken]
    return {
        "success": goal.evaluate(sim.state.fluents),
        "wall_time": wall_time,
        "plan_cost": float(sim.state.time),
        "actions_count": len(actions_taken),
        "actions": actions_taken,
    }


# Parameter sweep for --bench mode (single mode uses index --case, default 0).
tutorial_main.add_cases(
    [
        {"num_locations": 3, "mcts.iterations": 100},
        {"num_locations": 5, "mcts.iterations": 100},
        {"num_locations": 5, "mcts.iterations": 400},
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

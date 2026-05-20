"""Shared harness for the Brown University railroad planner tutorial.

The talk-day file (`tutorial_brown.py`) only writes the planning function and
its `add_cases([...])` sweep; everything else -- CLI, dashboard wiring,
MLflow registration, media saving, `--clear` -- lives here so the editable
file stays tiny and readable on a projector.

Usage from the talk file:

    from tutorial_brown_base import BenchmarkCase, LABEL_ENV_VAR, tutorial

    @tutorial(description="...", repeat=8, timeout=120.0)
    def tutorial_main(case: BenchmarkCase) -> dict:
        ...

    tutorial_main.add_cases([...])

    if os.environ.get(LABEL_ENV_VAR):
        tutorial_main._register(os.environ[LABEL_ENV_VAR])

    if __name__ == "__main__":
        tutorial_main.run_cli()
"""

import os
import sys
from types import SimpleNamespace

import rich_click as click

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
#: which re-import the talk file and must register the identical
#: "tutorial::<label>".
LABEL_ENV_VAR = "RAILROAD_TUTORIAL_LABEL"

#: MLflow backend used by both the tutorial runs and the bench dashboard.
#: Cwd-relative, matching the convention in `railroad.bench.tracking`.
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

__all__ = [
    "BenchmarkCase",
    "EXPERIMENT_NAME",
    "LABEL_ENV_VAR",
    "Tutorial",
    "tutorial",
]


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
        """rich-click CLI, styled to match `railroad` (see cli.py)."""
        tutorial = self

        @click.command(
            context_settings={"help_option_names": ["-h", "--help"]}
        )
        @click.option_panel(
            "Run mode", options=["--bench", "--clear", "--label"]
        )
        @click.option_panel(
            "Benchmark options", options=["--repeat", "--parallel", "--filter"]
        )
        @click.option_panel(
            "Single-run options", options=["--case", "--no-media", "--help"]
        )
        @click.option(
            "--bench", is_flag=True, default=False,
            help="Run the benchmark sweep (repeats + parameter sweep) via "
            "MLflow instead of a single live run.",
        )
        @click.option(
            "--clear", "clear_", is_flag=True, default=False,
            help="Delete all tutorial media files and the MLflow experiment "
            f"{EXPERIMENT_NAME!r} (prompts to confirm). --label is not "
            "required with --clear.",
        )
        @click.option(
            "--label", default=None,
            help="Required for run/bench modes. With --bench it is the "
            "comparison group (benchmark 'tutorial::<label>'); in single "
            "mode it names the saved <label>.jpg / <label>.mp4 (overwriting "
            "a prior run). Not used by --clear.",
        )
        @click.option(
            "--repeat", type=int, default=None,
            help="Cap repeats per case (benchmark mode).",
        )
        @click.option(
            "--parallel", type=int, default=None,
            help="Number of worker processes (benchmark mode).",
        )
        @click.option(
            "--filter", "filter_", default=None,
            help="pytest-style case filter (benchmark mode).",
        )
        @click.option(
            "--case", type=int, default=0, show_default=True,
            help="Single mode: which sweep case index to run.",
        )
        @click.option(
            "--no-media", is_flag=True, default=False,
            help="Single mode: skip saving the trajectory plot/video.",
        )
        def _cli(bench, clear_, label, repeat, parallel, filter_, case, no_media):
            """Railroad planner tutorial.

            Default: one live run with the interactive dashboard, saving a
            720p plot/video to the media dir. With --bench: a benchmark sweep
            accumulating into the persistent MLflow experiment. With
            --clear: wipe tutorial_media/ and the MLflow experiment.
            """
            if clear_:
                tutorial._run_clear()
                return
            if not label:
                raise click.UsageError(
                    "--label is required (unless using --clear)."
                )
            args = SimpleNamespace(
                bench=bench, label=label, repeat=repeat, parallel=parallel,
                filter=filter_, case=case, no_media=no_media,
            )
            tutorial._label = label
            tutorial._no_media = no_media
            if bench:
                tutorial._run_bench(args)
            else:
                tutorial._run_single(args)

        _cli.main(args=argv, standalone_mode=True)

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
        # --label is enforced in the CLI body.
        # Propagate the label to re-imported worker processes, and register.
        os.environ[LABEL_ENV_VAR] = args.label
        bench = self._register(args.label)

        from railroad.bench.runner import BenchmarkRunner

        # Walk back up the call stack to find the talk file that defined the
        # tutorial. `include_files` must point at THAT file (not this base
        # module) so workers re-import it and register the same benchmark.
        talk_file = os.path.abspath(self.user_fn.__code__.co_filename)
        runner = BenchmarkRunner(
            benchmarks=[bench],
            repeat_max=args.repeat,
            parallel=args.parallel,
            case_filter=args.filter,
            include_files=[talk_file],
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

    def _run_clear(self) -> None:
        """Wipe tutorial_media/ + the MLflow tutorial experiment, after confirming.

        MLflow's `delete_experiment` is a soft delete, so we follow it with
        `mlflow gc --experiment-ids <id>` to permanently remove rows and
        artifacts from the SQLite store. Scoped by experiment id so other
        soft-deleted experiments in the same backend are not touched.
        """
        import subprocess

        import mlflow
        from mlflow.tracking import MlflowClient

        from railroad.bench.tutorial_media import media_dir

        d = media_dir()
        media_files = sorted(p for p in d.iterdir() if p.is_file())

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        run_count = 0
        labels: list[str] = []
        if experiment is not None:
            runs = client.search_runs(
                [experiment.experiment_id], max_results=10000
            )
            run_count = len(runs)
            # Per-task run names look like 'tutorial::<label>_<case>_<repeat>'.
            # Strip the prefix and the trailing _<case>_<repeat> to recover
            # the user-facing label group.
            seen: set[str] = set()
            for r in runs:
                name = r.info.run_name or ""
                if name.startswith("tutorial::"):
                    name = name[len("tutorial::"):]
                # drop _<repeat> then _<case>; both are integers
                for _ in range(2):
                    head, _, tail = name.rpartition("_")
                    if head and tail.isdigit():
                        name = head
                if name and name not in seen:
                    seen.add(name)
                    labels.append(name)

        if not media_files and experiment is None:
            click.echo(
                "Nothing to clear: no files in "
                f"{d} and no MLflow experiment {EXPERIMENT_NAME!r}."
            )
            return

        click.echo("About to delete:")
        if media_files:
            click.echo(f"  - {d}  ({len(media_files)} files)")
            for p in media_files:
                click.echo(f"      {p.name}")
        if experiment is not None:
            label_str = ", ".join(labels) if labels else "(none)"
            click.echo(
                f"  - MLflow experiment {EXPERIMENT_NAME!r} "
                f"({run_count} runs, labels: {label_str})"
            )
        if not click.confirm("Proceed?", default=False):
            click.echo("Aborted.")
            return

        for p in media_files:
            p.unlink()
        if experiment is not None:
            experiment_id = experiment.experiment_id
            client.delete_experiment(experiment_id)
            gc = subprocess.run(
                [
                    "mlflow", "gc",
                    "--backend-store-uri", MLFLOW_TRACKING_URI,
                    "--experiment-ids", str(experiment_id),
                ],
                capture_output=True, text=True,
            )
            if gc.returncode != 0:
                click.echo(
                    f"Soft delete OK, but `mlflow gc` failed:\n"
                    f"{gc.stderr.strip()}\n"
                    f"Experiment is hidden but rows remain. Run manually:\n"
                    f"  mlflow gc --backend-store-uri {MLFLOW_TRACKING_URI} "
                    f"--experiment-ids {experiment_id}"
                )
        click.echo("Cleared.")


def tutorial(description: str = "", repeat: int = 8, timeout: float = 120.0):
    """Decorator: turn the editable function into a runnable Tutorial."""

    def decorator(fn) -> Tutorial:
        return Tutorial(fn, description=description, repeat=repeat, timeout=timeout)

    return decorator

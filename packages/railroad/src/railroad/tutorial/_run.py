"""Launching things: the demo, its sweep, and the dashboard.

Everything is a subprocess, run **from inside the playground**. That one choice
is what keeps the tutorial self-contained: ``mlflow.db``, ``mlruns/``, the
benchmark cache and the media directory are all resolved relative to the
working directory, so putting the working directory in the playground gives the
tutorial its own of each. Existing results are untouched, and the dashboard
opened from here shows this tutorial and nothing else.

The exception is the ProcTHOR resource tree -- a gigabyte of scenes, models and
the scene cache -- which is borrowed from wherever ``init`` was run via
``PROCTHOR_RESOURCES_DIR`` rather than copied.

A fresh interpreter per run also means no stale module state between saves, the
child inherits the terminal so the live dashboard renders properly, and a
syntax error introduced mid-edit prints a traceback instead of taking the watch
pane down with it.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Sequence

from ._playground import ENV_DIR, Playground

EXPERIMENT = "railroad-tutorial"
"""Every sweep accumulates here, so the dashboard is one page you refresh."""

DASHBOARD_URL = "http://127.0.0.1:8050/"
MEDIA_ENV = "RAILROAD_TUTORIAL_MEDIA_DIR"
PROCTHOR_ENV = "PROCTHOR_RESOURCES_DIR"

DEFAULT_PARALLEL = 12
"""Deliberately below ``cpu_count() - 2``: a sweep that eats every core makes
the machine you are presenting from unusable. Override with ``--parallel``."""


def _railroad(*args: str) -> List[str]:
    """The CLI, run through this interpreter so the venv is never in doubt."""
    return [sys.executable, "-m", "railroad.cli", *args]


def _env_for(playground: Playground) -> dict:
    env = dict(os.environ)
    env[ENV_DIR] = str(playground.root)
    env[MEDIA_ENV] = str(playground.media_dir)
    resources = playground.resources_dir
    if resources.is_dir():
        env[PROCTHOR_ENV] = str(resources)
    return env


def format_command(argv: Sequence[str], cwd: Optional[Path] = None) -> str:
    """The same thing, as you would type it.

    Printed before every run so the tutorial shows its working: nothing here is
    a special mode, it is the ordinary CLI with ordinary arguments.
    """
    parts = list(argv)
    if parts[:3] == [sys.executable, "-m", "railroad.cli"]:
        parts = ["railroad", *parts[3:]]
    elif parts and parts[0] == sys.executable:
        parts = ["python", *parts[1:]]
    if cwd is not None:
        try:
            where = cwd.relative_to(Path.cwd())
        except ValueError:
            where = cwd
        if str(where) != ".":
            return f"cd {shlex.quote(str(where))} && " + shlex.join(parts)
    return shlex.join(parts)


@dataclass(frozen=True)
class RunResult:
    returncode: int
    wall: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def demo_argv(playground: Playground, extra_args: Sequence[str] = ()) -> List[str]:
    return [sys.executable, playground.demo.name, *extra_args]


def sweep_argv(
    playground: Playground,
    *,
    parallel: int = DEFAULT_PARALLEL,
    repeat_max: Optional[int] = None,
    dry_run: bool = False,
) -> List[str]:
    """The benchmark command for the current step.

    Selection is by the ``tutorial`` tag rather than by name: ``--include``
    adds ``demo.py`` to the benchmarks discovered from entry points, and
    without a filter that would drag in every shipped benchmark in the repo.
    """
    args = [
        "benchmarks", "run",
        "--include", playground.demo.name,
        "--tags", "tutorial",
        "--experiment", EXPERIMENT,
        "--run-name", f"step{playground.current_step_id}",
        "--parallel", str(parallel),
    ]
    if repeat_max is not None:
        args += ["--repeat-max", str(repeat_max)]
    if dry_run:
        args.append("--dry-run")
    return _railroad(*args)


def dashboard_argv(host: str = "auto", port: int = 8050) -> List[str]:
    return _railroad("benchmarks", "dashboard", "--host", host, "--port", str(port))


def _run(playground: Playground, argv: Sequence[str]) -> RunResult:
    started = perf_counter()
    completed = subprocess.run(
        list(argv), env=_env_for(playground), cwd=str(playground.root)
    )
    return RunResult(completed.returncode, perf_counter() - started)


def run_demo(playground: Playground, extra_args: Sequence[str] = ()) -> RunResult:
    """Run ``demo.py`` to completion, inheriting this terminal."""
    return _run(playground, demo_argv(playground, extra_args))


def run_sweep(
    playground: Playground,
    *,
    parallel: int = DEFAULT_PARALLEL,
    repeat_max: Optional[int] = None,
    dry_run: bool = False,
) -> RunResult:
    """Run the current step's benchmark sweep."""
    return _run(playground, sweep_argv(
        playground, parallel=parallel, repeat_max=repeat_max, dry_run=dry_run
    ))


def start_dashboard(
    playground: Playground, *, host: str = "auto", port: int = 8050
) -> subprocess.Popen:
    """Start the benchmark dashboard in the background.

    Run from the playground, so it reads the playground's ``mlflow.db`` and
    serves the playground's media -- this tutorial's results and nothing else.
    Its own output goes to ``dashboard.log``; a Flask server logging into the
    middle of a talk is not what anyone wants to look at.
    """
    log = (playground.root / "dashboard.log").open("ab")
    return subprocess.Popen(
        dashboard_argv(host, port),
        env=_env_for(playground),
        cwd=str(playground.root),
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )


def dashboard_urls(port: int = 8050) -> List[str]:
    """Where a dashboard bound to every interface can be reached."""
    try:
        from railroad.bench.dashboard.net import ALL_INTERFACES, url_lines
    except ImportError:
        return [f"  {DASHBOARD_URL}"]
    return url_lines(ALL_INTERFACES, port)


def open_browser(url: str = DASHBOARD_URL) -> None:
    """Best-effort: never let a missing browser interrupt anything."""
    import webbrowser

    try:
        webbrowser.open(url)
    except Exception:
        pass


def editor_command(path: Path) -> List[str]:
    """``$VISUAL``/``$EDITOR`` split into a command, defaulting to vi."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    return [*shlex.split(editor), str(path)]

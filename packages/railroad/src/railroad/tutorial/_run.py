"""Launching things: the demo, its sweep, and the dashboard.

Everything is a subprocess. The demo especially: a fresh interpreter each time
means no stale module state between saves, the child inherits the terminal so
the live dashboard renders properly, and a syntax error introduced mid-edit
prints a traceback instead of taking the watch pane down with it.

The working directory is inherited, never set. ``mlflow.db``, ``mlruns/``,
``.benchmark_cache/`` and the ProcTHOR scene cache are all resolved relative to
it, so the tutorial has to run from wherever those live -- normally the
repository root.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Optional

from ._playground import ENV_DIR, Playground

EXPERIMENT = "railroad-tutorial"
"""Every sweep accumulates here, so the dashboard is one page you refresh."""

DASHBOARD_URL = "http://127.0.0.1:8050/"

DEFAULT_PARALLEL = 12
"""Deliberately below ``cpu_count() - 2``: a sweep that eats every core makes
the machine you are presenting from unusable. Override with ``--parallel``."""


def _railroad(*args: str) -> List[str]:
    """The CLI, run through this interpreter so the venv is never in doubt."""
    return [sys.executable, "-m", "railroad.cli", *args]


def _env_for(playground: Playground) -> dict:
    env = dict(os.environ)
    env[ENV_DIR] = str(playground.root)
    return env


@dataclass(frozen=True)
class RunResult:
    returncode: int
    wall: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_demo(playground: Playground) -> RunResult:
    """Run ``demo.py`` to completion, inheriting this terminal."""
    started = perf_counter()
    completed = subprocess.run(
        [sys.executable, str(playground.demo)],
        env=_env_for(playground),
    )
    return RunResult(completed.returncode, perf_counter() - started)


def run_sweep(
    playground: Playground,
    *,
    parallel: int = DEFAULT_PARALLEL,
    repeat_max: Optional[int] = None,
    dry_run: bool = False,
) -> RunResult:
    """Run the current step's benchmark sweep.

    Selection is by the ``tutorial`` tag rather than by name: ``--include``
    adds ``demo.py`` to the benchmarks discovered from entry points, and
    without a filter that would drag in every shipped benchmark in the repo.
    """
    args = [
        "benchmarks", "run",
        "--include", str(playground.demo),
        "--tags", "tutorial",
        "--experiment", EXPERIMENT,
        "--run-name", f"step{playground.current_step_id}",
        "--parallel", str(parallel),
    ]
    if repeat_max is not None:
        args += ["--repeat-max", str(repeat_max)]
    if dry_run:
        args.append("--dry-run")
    started = perf_counter()
    completed = subprocess.run(_railroad(*args), env=_env_for(playground))
    return RunResult(completed.returncode, perf_counter() - started)


def start_dashboard(playground: Playground) -> subprocess.Popen:
    """Start the benchmark dashboard in the background.

    Its output goes to ``dashboard.log`` in the playground; a Flask server
    logging into the middle of a talk is not what anyone wants to look at.
    """
    log = (playground.root / "dashboard.log").open("ab")
    return subprocess.Popen(
        _railroad("benchmarks", "dashboard"),
        env=_env_for(playground),
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )


def open_browser(url: str = DASHBOARD_URL) -> None:
    """Best-effort: never let a missing browser interrupt anything."""
    import webbrowser

    try:
        webbrowser.open(url)
    except Exception:
        pass


def editor_command(path: Path) -> List[str]:
    """``$VISUAL``/``$EDITOR`` split into a command, defaulting to vi."""
    import shlex

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    return [*shlex.split(editor), str(path)]

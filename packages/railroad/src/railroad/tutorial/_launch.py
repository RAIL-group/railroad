"""Running the notebook, the demo, the sweep and the dashboard -- from the playground.

Four thin wrappers, and one thing they all do: print the command they are
about to run. A wrapper that hides what it does teaches you the wrapper, so
``railroad tutorial bench`` shows the ``railroad benchmarks run`` line it
expands to. Short to type, and still honest about what it is.

Everything runs with the working directory set to the playground, which is the
whole isolation story -- ``mlflow.db``, ``mlruns/``, ``.benchmark_cache/`` and
``media/`` are all resolved from there.

The dashboard is the one that needs more than a subprocess call: it outlives
the command that started it, so its pid is recorded and it can be asked about
and torn down later.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from ._playground import Playground
from ._steps import DEMO_FILE, EXPERIMENT, NOTEBOOK, RUNNER

DASHBOARD_STATE = ".dashboard.json"
DASHBOARD_LOG = "dashboard.log"
DEFAULT_PORT = 8050
NOTEBOOK_PORT = 8888
"""Jupyter's own default, so the URL is the one people expect."""

NOTEBOOK_URL_PATH = f"/notebooks/{NOTEBOOK}"
"""Notebook 7's route to one document: menu, toolbar, cells, and nothing else.

Handed to the interface as its ``default_url``, which is the address ``/``
redirects to -- so the link anyone is given stays short and still opens the
notebook rather than a file listing.

The lab interface answers on the same server at ``/lab`` if you want the file
browser and the tab bar, and at ``/doc/tree/<file>`` for its own one-document
mode.
"""


def _module(module: str, *args: str) -> List[str]:
    """A console script, through this interpreter.

    ``python -m jupyter lab`` rather than ``jupyter lab``, so which environment
    it lands in is never a question of what happens to be on PATH.
    """
    return [sys.executable, "-m", module, *args]


def _railroad(*args: str) -> List[str]:
    return _module("railroad.cli", *args)


def demo_argv(extra: Sequence[str] = ()) -> List[str]:
    return [sys.executable, DEMO_FILE, *extra]


def bench_argv(extra: Sequence[str] = ()) -> List[str]:
    """The sweep for whatever ``demo.py`` currently is.

    ``--tags tutorial`` is load-bearing: ``--include`` *adds* demo.py to the
    benchmarks found through entry points, so without a filter every benchmark
    in the repository would come along. ``--experiment`` is what makes
    successive sweeps accumulate onto one dashboard page instead of each
    minting its own timestamped experiment.
    """
    return _railroad(
        "benchmarks", "run",
        "-i", DEMO_FILE,
        "--tags", "tutorial",
        "--experiment", EXPERIMENT,
        *extra,
    )


def dashboard_argv(host: str = "auto", port: int = DEFAULT_PORT) -> List[str]:
    return _railroad("benchmarks", "dashboard", "--host", host, "--port", str(port))


def notebook_argv(
    extra: Sequence[str] = (), *, host: str = "auto", port: int = NOTEBOOK_PORT
) -> List[str]:
    """Jupyter on the language primer, in the foreground.

    Notebook 7 rather than JupyterLab: the same server and the same packages,
    but a document-centric interface -- no file browser, no tab bar, no
    launcher -- which is what you want on a projector. ``default_url`` is what
    lands a visitor *in the notebook* instead of in a file listing; both the
    printed links and Jupyter's own then point at the document.

    The rest of the defaults assume the machine running this is not the
    machine you are looking at, which is the normal case here: no browser to
    launch, every interface answered, and no token to copy out of a log.

    Each default is *dropped* when you pass the same argument, rather than
    being followed by yours -- traitlets rejects a repeated argument outright
    rather than letting the last one win. So ``--ip 127.0.0.1`` closes it back
    down, and ``--IdentityProvider.token=secret`` puts the key back.

    Unlike the results dashboard this is not backgrounded: Ctrl-C stops it.
    """
    supplied = {arg.split("=", 1)[0] for arg in extra}
    argv: List[str] = []
    if "--JupyterNotebookApp.default_url" not in supplied:
        argv.append(f"--JupyterNotebookApp.default_url={NOTEBOOK_URL_PATH}")
    if not supplied & {"--no-browser", "--browser"}:
        argv.append("--no-browser")
    if "--ip" not in supplied:
        argv += ["--ip", bind_address(host)]
    if "--port" not in supplied:
        argv += ["--port", str(port)]
    if not supplied & {"--IdentityProvider.token", "--ServerApp.token"}:
        argv.append("--IdentityProvider.token=")
    return _module("jupyter", "notebook", NOTEBOOK, *argv, *extra)


def pretty(argv: Sequence[str]) -> str:
    """The same command, as you would type it.

    Absolute interpreter paths and ``-m railroad.cli`` are how a subprocess has
    to spell it; neither is something anyone would type, and printing them
    would undercut the point of printing at all.
    """
    parts = list(argv)
    if parts[:2] == [sys.executable, "-m"]:
        entry = {"railroad.cli": "railroad"}.get(parts[2], parts[2])
        parts = [entry, *parts[3:]]
    elif parts and parts[0] == sys.executable:
        parts = ["python", *parts[1:]]
    return f"{RUNNER} {shlex.join(parts)}"


@dataclass(frozen=True)
class RunResult:
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run(playground: Playground, argv: Sequence[str]) -> RunResult:
    """Run *argv* in the playground, inheriting this terminal.

    A fresh interpreter each time, so there is no stale module state between
    edits and a syntax error introduced mid-sentence prints a traceback rather
    than taking anything else down.
    """
    completed = subprocess.run(list(argv), cwd=str(playground.root))
    return RunResult(completed.returncode)


# -- the dashboard, which outlives the command that starts it -----------------


@dataclass(frozen=True)
class Dashboard:
    """A recorded dashboard process."""

    pid: int
    port: int
    host: str

    @property
    def alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True


def _state_path(playground: Playground) -> Path:
    return playground.root / DASHBOARD_STATE


def log_path(playground: Playground) -> Path:
    return playground.root / DASHBOARD_LOG


def recorded(playground: Playground) -> Optional[Dashboard]:
    """The dashboard this playground last started, if it is still running.

    A stale record -- the process died, or the machine rebooted -- is cleared
    rather than reported, so ``dashboard`` after a crash starts a new one
    instead of insisting an old one is up.
    """
    path = _state_path(playground)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text())
        board = Dashboard(int(state["pid"]), int(state["port"]), str(state["host"]))
    except (OSError, ValueError, KeyError, TypeError):
        path.unlink(missing_ok=True)
        return None
    if not board.alive:
        path.unlink(missing_ok=True)
        return None
    return board


def start(
    playground: Playground, *, host: str = "auto", port: int = DEFAULT_PORT
) -> Dashboard:
    """Start the dashboard in the background and record its pid.

    ``start_new_session`` puts it in its own process group. The server runs
    with hot reload on, so it is a parent and a child; a group is the only
    handle that reliably stops both.
    """
    # Truncated, not appended: the log should describe the dashboard that is
    # running now, so that asking after a failed start shows why it failed.
    handle = log_path(playground).open("wb")
    process = subprocess.Popen(
        dashboard_argv(host, port),
        cwd=str(playground.root),
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    board = Dashboard(process.pid, port, host)
    _state_path(playground).write_text(
        json.dumps({"pid": board.pid, "port": board.port, "host": board.host}) + "\n"
    )
    return board


def stop(playground: Playground, *, timeout: float = 5.0) -> Optional[int]:
    """Tear the dashboard down. Returns the pid stopped, or ``None``."""
    board = recorded(playground)
    _state_path(playground).unlink(missing_ok=True)
    if board is None:
        return None
    try:
        group = os.getpgid(board.pid)
    except OSError:
        return None
    os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not board.alive:
            return board.pid
        time.sleep(0.1)
    # It ignored the polite request; it is a development server, not a database.
    os.killpg(group, signal.SIGKILL)
    return board.pid


def bind_address(host: str = "auto") -> str:
    """Turn a host name into an address to bind, as the dashboard does.

    ``auto`` is every interface, which is what makes a server started over ssh
    visible from the laptop you are actually looking at.
    """
    try:
        from railroad.bench.dashboard.net import resolve_host
    except ImportError:
        return "0.0.0.0"
    return resolve_host(host)


def urls(port: int = DEFAULT_PORT, host: str = "auto") -> List[str]:
    """Every address the dashboard can be reached on, phone and tailnet included."""
    try:
        from railroad.bench.dashboard.net import resolve_host, url_lines
    except ImportError:
        return [f"  http://127.0.0.1:{port}/"]
    return url_lines(resolve_host(host), port)


def log_tail(playground: Playground, lines: int = 5) -> List[str]:
    """The last few lines of the dashboard's own output, for when it will not start."""
    path = log_path(playground)
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-lines:]

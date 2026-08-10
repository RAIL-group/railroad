"""The pane that sits next to your editor.

It does two things: re-runs ``demo.py`` whenever you save it, and turns single
keypresses into tutorial commands, so nothing has to be typed mid-sentence.

Output is append-only rather than a full-screen live view. The demo's own
dashboard already takes over the screen while it runs, and scrollback is worth
more than a tidy frame when someone asks about a number from two steps ago.
"""

from __future__ import annotations

import select
import subprocess
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple

from rich.console import Console
from rich.markup import escape

from . import commands as cmd
from ._playground import Playground
from ._run import (
    DASHBOARD_URL,
    DEFAULT_PARALLEL,
    RunResult,
    editor_command,
    open_browser,
    start_dashboard,
)
from ._steps import get_step, neighbour

POLL_SECONDS = 0.25
SETTLE_SECONDS = 0.15
"""Editors often write in more than one syscall; let the file stop moving."""

KEYMAP = (
    ("n", "next"),
    ("p", "prev"),
    ("k", "peek"),
    ("d", "diff"),
    ("r", "run"),
    ("b", "sweep"),
    ("o", "dashboard"),
    ("c", "compare"),
    ("u", "undo"),
    ("e", "edit"),
    ("q", "quit"),
)


class NotATerminal(RuntimeError):
    """The watch pane needs a real terminal to read single keypresses."""


class KeyReader:
    """Single-keypress input, with an escape hatch for child processes."""

    def __init__(self, stream=None) -> None:
        self._stream = stream or sys.stdin
        self._fd = self._stream.fileno()
        self._saved: Optional[list] = None

    def __enter__(self) -> "KeyReader":
        self._saved = termios.tcgetattr(self._fd)
        # cbreak rather than raw: ISIG stays on, so Ctrl-C still interrupts.
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc_info) -> None:
        self._restore()

    def _restore(self) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    @contextmanager
    def cooked(self) -> Iterator[None]:
        """Hand the terminal back, for a child that wants to own the screen."""
        self._restore()
        try:
            yield
        finally:
            tty.setcbreak(self._fd)

    def poll(self, timeout: float) -> Optional[str]:
        ready, _, _ = select.select([self._stream], [], [], timeout)
        if not ready:
            return None
        return self._stream.read(1)


def _signature(path: Path) -> Optional[Tuple[int, int]]:
    try:
        stat = path.stat()
    except OSError:
        return None  # mid-write; we will catch it on the next poll
    return stat.st_mtime_ns, stat.st_size


def _settled_signature(path: Path) -> Optional[Tuple[int, int]]:
    signature = _signature(path)
    while True:
        time.sleep(SETTLE_SECONDS)
        again = _signature(path)
        if again == signature:
            return again
        signature = again


def _last_run(playground: Playground, step_id: str) -> Optional[dict]:
    for record in reversed(playground.read_runs()):
        if record.get("step") == step_id:
            return record
    return None


def _summary(playground: Playground, step_id: str) -> Optional[str]:
    """'cost 22.1s (step 01 was 44.3s) · 11 actions' for the current step."""
    record = _last_run(playground, step_id)
    if record is None:
        return None
    cost = record.get("cost")
    if not isinstance(cost, (int, float)):
        return None
    parts = [f"cost {cost:.1f}s"]
    previous = neighbour(step_id, -1)
    # Only worth quoting when the previous step solved the same problem; across
    # a change of world the two numbers are not about the same thing.
    if previous is not None and previous["problem"] == get_step(step_id)["problem"]:
        earlier = _last_run(playground, previous["id"])
        if earlier and isinstance(earlier.get("cost"), (int, float)):
            parts.append(f"(step {previous['id']} was {earlier['cost']:.1f}s)")
    parts.append(f"{len(record.get('actions', []))} actions")
    if not record.get("goal_reached"):
        parts.append("[red]goal not reached[/red]")
    return " · ".join(parts)


def banner(console: Console, playground: Playground) -> None:
    step = get_step(playground.current_step_id)
    console.rule(f"[bold]step {step['id']} · {escape(step['title'])}[/bold]")
    console.print(f"  {escape(step['point'])}")
    summary = _summary(playground, step["id"])
    if summary:
        console.print(f"  {summary}")
    if step["sweep"]:
        console.print(f"  [dim]sweep: {escape(step['sweep'])}[/dim]")
    console.print(f"[dim]watching {playground.demo.name} — save to re-run[/dim]")
    console.print("[dim]" + "  ".join(f"{key} {name}" for key, name in KEYMAP) + "[/dim]")


def watch(
    console: Console,
    playground: Playground,
    *,
    parallel: int = DEFAULT_PARALLEL,
    editor_sync: bool = True,
) -> None:
    """Run the pane until ``q``."""
    if not sys.stdin.isatty():
        raise NotATerminal(
            "railroad tutorial watch needs a terminal; use 'railroad tutorial run' "
            "for a one-shot run"
        )

    dashboard: Optional[subprocess.Popen] = None
    signature = _signature(playground.demo)
    banner(console, playground)

    try:
        with KeyReader() as keys:

            def ask(prompt: str) -> str:
                console.print(f"{prompt} [y/N/f] ", end="", markup=False)
                while True:
                    key = keys.poll(1.0)
                    if key is None:
                        continue
                    console.print(key, markup=False)
                    return {"y": "yes", "f": "force"}.get(key.lower(), "no")

            def run_demo_now() -> RunResult:
                with keys.cooked():
                    return cmd.cmd_run(console, playground)

            while True:
                key = keys.poll(POLL_SECONDS)

                if key is None:
                    current = _signature(playground.demo)
                    if current is not None and current != signature:
                        signature = _settled_signature(playground.demo)
                        console.rule("[dim]saved[/dim]")
                        run_demo_now()
                        banner(console, playground)
                    continue

                key = key.lower()
                if key == "q":
                    break
                if key in ("n", "p"):
                    moved = cmd.cmd_step(
                        console, 1 if key == "n" else -1,
                        playground=playground, ask=ask, editor_sync=editor_sync,
                    )
                    if moved:
                        signature = _signature(playground.demo)
                        run_demo_now()
                    banner(console, playground)
                elif key == "k":
                    cmd.cmd_peek(console, playground)
                elif key == "d":
                    cmd.cmd_diff(console, None, playground)
                elif key == "r":
                    run_demo_now()
                    banner(console, playground)
                elif key == "b":
                    with keys.cooked():
                        cmd.cmd_bench(console, playground=playground, parallel=parallel)
                    banner(console, playground)
                elif key == "o":
                    if dashboard is None or dashboard.poll() is not None:
                        dashboard = start_dashboard(playground)
                        console.print(
                            f"[green]dashboard[/green] {DASHBOARD_URL}  "
                            f"[dim](log: {playground.root / 'dashboard.log'})[/dim]"
                        )
                    else:
                        console.print(f"[dim]dashboard already up at {DASHBOARD_URL}[/dim]")
                    open_browser()
                elif key == "c":
                    cmd.cmd_compare(console, playground)
                elif key == "u":
                    cmd.cmd_undo(console, playground)
                    signature = _signature(playground.demo)
                    banner(console, playground)
                elif key == "e":
                    with keys.cooked():
                        subprocess.run(editor_command(playground.demo))
                    signature = _signature(playground.demo)
                    banner(console, playground)
                elif key in ("?", "h"):
                    banner(console, playground)
    except KeyboardInterrupt:
        console.print()
    finally:
        if dashboard is not None and dashboard.poll() is None:
            dashboard.terminate()
    console.print("[dim]bye[/dim]")


__all__ = ["NotATerminal", "banner", "watch"]

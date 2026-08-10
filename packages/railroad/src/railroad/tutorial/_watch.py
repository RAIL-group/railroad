"""The pane that sits below your editor.

A status panel pinned to the bottom of the terminal, with everything else --
diffs, run output, sweep progress -- scrolling above it. Single keypresses drive
the tutorial, so nothing has to be typed mid-sentence.

Runs are **manual**. Saving ``demo.py`` marks the panel as edited; pressing `r`
is what runs it. Re-running on every save sounds convenient and is not: a save
lands mid-thought, and a twenty-second ProcTHOR run starting on its own while
you are still talking is worse than no automation at all.

The panel is a transient ``rich.Live``, torn down around any child process so
the demo's own full-screen dashboard gets the terminal to itself.
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

from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from . import commands as cmd
from ._playground import Playground
from ._run import (
    DASHBOARD_URL,
    DEFAULT_PARALLEL,
    editor_command,
    open_browser,
    start_dashboard,
)
from ._steps import get_step, neighbour

POLL_SECONDS = 0.2
SETTLE_SECONDS = 0.15
"""Editors often write in more than one syscall; let the file stop moving."""

KEYMAP = (
    ("r", "run"),
    ("n", "next"),
    ("p", "prev"),
    ("k", "peek"),
    ("d", "diff"),
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
        parts.append("goal not reached")
    return " · ".join(parts)


class Pane:
    """The bottom panel and the key loop that drives it."""

    def __init__(
        self,
        console: Console,
        playground: Playground,
        *,
        parallel: int = DEFAULT_PARALLEL,
        editor_sync: bool = True,
    ) -> None:
        self.console = console
        self.playground = playground
        self.parallel = parallel
        self.editor_sync = editor_sync
        self.edited = False
        self._signature = _signature(playground.demo)
        self._dashboard: Optional[subprocess.Popen] = None

    # -- rendering -----------------------------------------------------------

    def panel(self) -> Panel:
        step = get_step(self.playground.current_step_id)
        body = Text()
        body.append(escape(step["point"]) + "\n", style="dim")

        summary = _summary(self.playground, step["id"])
        if summary:
            body.append(summary + "\n")
        if step["sweep"]:
            body.append(f"sweep: {escape(step['sweep'])}\n", style="dim")

        if self.edited:
            body.append("demo.py edited — press r to run\n", style="yellow")
        else:
            body.append(f"watching {self.playground.demo.name}\n", style="dim")

        keys = Text()
        for key, name in KEYMAP:
            keys.append(f" {key} ", style="reverse")
            keys.append(f"{name}  ", style="dim")
        return Panel(
            Group(body, keys),
            title=f"step {step['id']} · {escape(step['title'])}",
            title_align="left",
            border_style="blue",
        )

    # -- the loop ------------------------------------------------------------

    def run(self) -> None:
        if not sys.stdin.isatty():
            raise NotATerminal(
                "railroad tutorial watch needs a terminal; use "
                "'railroad tutorial run' for a one-shot run"
            )
        try:
            with KeyReader() as keys, Live(
                self.panel(),
                console=self.console,
                refresh_per_second=8,
                transient=True,
            ) as live:
                self._loop(keys, live)
        except KeyboardInterrupt:
            self.console.print()
        finally:
            if self._dashboard is not None and self._dashboard.poll() is None:
                self._dashboard.terminate()
        self.console.print("[dim]bye[/dim]")

    def _loop(self, keys: KeyReader, live: Live) -> None:
        @contextmanager
        def scrolling() -> Iterator[None]:
            """Drop the panel so output lands cleanly, then put it back.

            Keeps cbreak, so a confirmation prompt can still read one key.
            """
            live.stop()
            try:
                yield
            finally:
                live.start(refresh=True)

        @contextmanager
        def child() -> Iterator[None]:
            """As above, and hand the terminal back: the demo's own dashboard
            takes the whole screen and wants a cooked tty."""
            with scrolling(), keys.cooked():
                yield

        def ask(prompt: str) -> str:
            self.console.print(f"{prompt} [y/N/f] ", end="", markup=False)
            while True:
                key = keys.poll(1.0)
                if key is None:
                    continue
                self.console.print(key, markup=False)
                return {"y": "yes", "f": "force"}.get(key.lower(), "no")

        while True:
            key = keys.poll(POLL_SECONDS)

            if key is None:
                current = _signature(self.playground.demo)
                if current is not None and current != self._signature:
                    self._signature = _settled_signature(self.playground.demo)
                    self.edited = True
                    live.update(self.panel(), refresh=True)
                continue

            key = key.lower()
            if key == "q":
                return
            self._handle(key, scrolling, child, ask)
            live.update(self.panel(), refresh=True)

    def _handle(self, key, scrolling, child, ask) -> None:
        console = self.console
        playground = self.playground

        if key == "r":
            with child():
                cmd.cmd_run(console, playground)
            self._signature = _signature(playground.demo)
            self.edited = False
        elif key in ("n", "p"):
            with scrolling():
                moved = cmd.cmd_step(
                    console, 1 if key == "n" else -1,
                    playground=playground, ask=ask, editor_sync=self.editor_sync,
                )
            self._signature = _signature(playground.demo)
            # Advancing rewrites demo.py, so the panel should say what it says
            # after any other edit: there is something new to run.
            self.edited = bool(moved)
        elif key == "k":
            with scrolling():
                cmd.cmd_peek(console, playground)
        elif key == "d":
            with scrolling():
                cmd.cmd_diff(console, None, playground)
        elif key == "b":
            with child():
                cmd.cmd_bench(console, playground=playground, parallel=self.parallel)
        elif key == "o":
            if self._dashboard is None or self._dashboard.poll() is not None:
                self._dashboard = start_dashboard(playground)
                console.print(f"[green]dashboard[/green] {DASHBOARD_URL}")
            else:
                console.print(f"[dim]dashboard already up at {DASHBOARD_URL}[/dim]")
            open_browser()
        elif key == "c":
            with scrolling():
                cmd.cmd_compare(console, playground)
        elif key == "u":
            with scrolling():
                cmd.cmd_undo(console, playground)
            self._signature = _signature(playground.demo)
            self.edited = True
        elif key == "e":
            with child():
                subprocess.run(editor_command(playground.demo))
            self._signature = _signature(playground.demo)
            self.edited = True


def watch(
    console: Console,
    playground: Playground,
    *,
    parallel: int = DEFAULT_PARALLEL,
    editor_sync: bool = True,
) -> None:
    """Run the pane until ``q``."""
    Pane(console, playground, parallel=parallel, editor_sync=editor_sync).run()


__all__ = ["NotATerminal", "Pane", "watch"]

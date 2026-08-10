"""The tutorial's commands, free of any CLI framework.

``railroad.cli`` is a thin click wrapper over these; the watch pane calls the
same functions when you press a key. Keeping them here means the two entry
points cannot drift, and the interesting logic is testable without a terminal.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import _advance as adv
from ._playground import (
    DEFAULT_DIRNAME,
    Playground,
    PlaygroundError,
    find_playground,
    init_playground,
)
from ._run import (
    DASHBOARD_URL,
    DEFAULT_PARALLEL,
    EXPERIMENT,
    RunResult,
    editor_command,
    open_browser,
    run_demo,
    run_sweep,
    start_dashboard,
)
from ._steps import STEPS, get_step, neighbour, step_index

Ask = Callable[[str], str]
"""Prompt the presenter; return ``"yes"``, ``"no"`` or ``"force"``."""


def _plain_ask(prompt: str) -> str:
    """Line-based confirmation, for the non-interactive CLI entry points."""
    reply = input(f"{prompt} [y/N/f] ").strip().lower()
    return {"y": "yes", "yes": "yes", "f": "force"}.get(reply, "no")


# -- status ------------------------------------------------------------------


def cmd_status(console: Console, playground: Optional[Playground] = None) -> None:
    """Where we are and what the steps are."""
    playground = playground or find_playground()
    current = playground.current_step_id
    console.print(f"[bold]playground[/bold]  {playground.root}")
    console.print()
    width = max(len(step["title"]) for step in STEPS) + 2
    for step in STEPS:
        marker = "[bold green]>[/bold green]" if step["id"] == current else " "
        style = "bold" if step["id"] == current else "dim"
        title = escape(step["title"]).ljust(width)
        console.print(f" {marker} [{style}]{step['id']}  {title}"
                      f"{escape(step['point'])}[/{style}]")
    console.print()
    edits = playground.demo.read_text() != playground.pristine_text(current)
    if edits:
        added, removed = adv.diff_stat(
            playground.pristine_text(current), playground.demo.read_text()
        )
        console.print(f"[yellow]demo.py has local edits (+{added} -{removed})[/yellow]")
    console.print("Run [bold]railroad tutorial watch[/bold] beside your editor.")


def cmd_init(console: Console, directory: Optional[str], force: bool) -> None:
    root = Path(directory) if directory else Path.cwd() / DEFAULT_DIRNAME
    playground = init_playground(root, force=force)
    console.print(f"[green]scaffolded[/green] {playground.root}")
    console.print(f"  {playground.demo.name} is on step {playground.current_step_id} "
                  f"({get_step(playground.current_step_id)['title']})")
    console.print()
    console.print("Open it in your editor with (global-auto-revert-mode 1), then:")
    console.print("  [bold]railroad tutorial watch[/bold]")


# -- running -----------------------------------------------------------------


def cmd_run(console: Console, playground: Optional[Playground] = None) -> RunResult:
    playground = playground or find_playground()
    result = run_demo(playground)
    if not result.ok:
        console.print(f"[red]demo.py exited {result.returncode}[/red]")
    return result


def cmd_edit(console: Console, playground: Optional[Playground] = None) -> None:
    playground = playground or find_playground()
    subprocess.run(editor_command(playground.demo))


# -- moving between steps ----------------------------------------------------


def render_patch(
    console: Console, playground: Playground, from_id: str, to_id: str
) -> None:
    """Show the canonical diff between two steps -- the unit of the talk."""
    from_step, to_step = get_step(from_id), get_step(to_id)
    diff = adv.unified(
        playground.pristine_text(from_id),
        playground.pristine_text(to_id),
        f"step {from_id} ({from_step['title']})",
        f"step {to_id} ({to_step['title']})",
    )
    added, removed = adv.diff_stat(
        playground.pristine_text(from_id), playground.pristine_text(to_id)
    )
    console.rule(f"step {to_id} · {escape(to_step['title'])}  (+{added} −{removed})")
    console.print(adv.colorize(diff))
    console.print(f"[dim]{escape(to_step['point'])}[/dim]")
    for note in to_step["notes"]:
        console.print(f"  [dim]· {escape(note)}[/dim]")
    if to_step["sweep"]:
        console.print(f"  [dim]sweep: {escape(to_step['sweep'])}[/dim]")


def cmd_peek(console: Console, playground: Optional[Playground] = None) -> None:
    playground = playground or find_playground()
    current = playground.current_step_id
    upcoming = neighbour(current, +1)
    if upcoming is None:
        console.print("[yellow]already on the last step[/yellow]")
        return
    render_patch(console, playground, current, upcoming["id"])


def cmd_goto(
    console: Console,
    target: str,
    *,
    playground: Optional[Playground] = None,
    ask: Optional[Ask] = None,
    force: bool = False,
    editor_sync: bool = True,
    show_patch: bool = True,
) -> bool:
    """Show the patch, confirm, then merge it into ``demo.py``.

    Returns whether the file actually moved. ``ask`` returning ``"force"``
    takes the target snapshot verbatim; the pre-advance file is snapshotted
    either way, so ``undo`` can always put it back.
    """
    playground = playground or find_playground()
    current = playground.current_step_id
    target_id = get_step(target)["id"]
    if target_id == current:
        console.print(f"[yellow]already on step {current}[/yellow]")
        return False

    if show_patch:
        render_patch(console, playground, current, target_id)

    if not force:
        answer = (ask or _plain_ask)("apply?")
        if answer == "no":
            console.print("[dim]not applied[/dim]")
            return False
        force = answer == "force"

    try:
        result = adv.advance(
            playground, target_id, force=force, editor_sync=editor_sync
        )
    except adv.MergeConflict as conflict:
        console.print(
            f"[red]{conflict.conflicts} conflicting hunk(s)[/red] between your edits "
            f"and this step's patch. Nothing was written."
        )
        console.print("[dim]Take the step's version instead with "
                      "'railroad tutorial goto "
                      f"{target_id} --force' (undo still works afterwards).[/dim]")
        return False
    except adv.MergeUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        return False

    moved = f"step {result.from_step} → {result.to_step}"
    if result.took_pristine:
        console.print(f"[green]{moved}[/green]  (took the step's version verbatim)")
    elif result.preserved_edits:
        console.print(f"[green]{moved}[/green]  (your local edits were merged in)")
    else:
        console.print(f"[green]{moved}[/green]")
    return True


def cmd_step(
    console: Console,
    offset: int,
    *,
    playground: Optional[Playground] = None,
    ask: Optional[Ask] = None,
    force: bool = False,
    editor_sync: bool = True,
) -> bool:
    """``next`` / ``prev``."""
    playground = playground or find_playground()
    target = neighbour(playground.current_step_id, offset)
    if target is None:
        edge = "last" if offset > 0 else "first"
        console.print(f"[yellow]already on the {edge} step[/yellow]")
        return False
    return cmd_goto(
        console, target["id"], playground=playground, ask=ask,
        force=force, editor_sync=editor_sync,
    )


def cmd_diff(
    console: Console,
    steps: Optional[Sequence[str]] = None,
    playground: Optional[Playground] = None,
) -> None:
    """Local edits against this step, or the canonical diff between two steps."""
    playground = playground or find_playground()
    if steps:
        first, second = steps
        render_patch(console, playground, get_step(first)["id"], get_step(second)["id"])
        return
    current = playground.current_step_id
    diff = adv.unified(
        playground.pristine_text(current),
        playground.demo.read_text(),
        f"step {current}",
        "your demo.py",
    )
    console.rule(f"your edits on top of step {current}")
    console.print(adv.colorize(diff))


def cmd_undo(console: Console, playground: Optional[Playground] = None) -> None:
    playground = playground or find_playground()
    try:
        entry = adv.undo(playground)
    except FileNotFoundError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return
    reason = entry.get("reason") or "snapshot"
    console.print(f"[green]restored[/green] demo.py from before: {reason}")
    console.print(f"  now on step {playground.current_step_id}")


# -- benchmarks --------------------------------------------------------------


def cmd_bench(
    console: Console,
    *,
    playground: Optional[Playground] = None,
    parallel: int = DEFAULT_PARALLEL,
    repeat_max: Optional[int] = None,
    dry_run: bool = False,
) -> RunResult:
    playground = playground or find_playground()
    step = get_step(playground.current_step_id)
    if not step["sweep"]:
        console.print(f"[yellow]step {step['id']} has no sweep[/yellow]")
        return RunResult(0, 0.0)
    console.print(f"[bold]sweep[/bold] {step['sweep']}  "
                  f"[dim]({parallel} workers → experiment {EXPERIMENT})[/dim]")
    result = run_sweep(
        playground, parallel=parallel, repeat_max=repeat_max, dry_run=dry_run
    )
    if not result.ok:
        console.print(f"[red]sweep exited {result.returncode}[/red]")
    return result


def cmd_dashboard(console: Console, playground: Optional[Playground] = None) -> None:
    playground = playground or find_playground()
    start_dashboard(playground)
    console.print(f"[green]dashboard[/green] {DASHBOARD_URL}  "
                  f"[dim](log: {playground.root / 'dashboard.log'})[/dim]")
    open_browser()


def cmd_compare(console: Console, playground: Optional[Playground] = None) -> None:
    """Most recent run of each step, with the change in plan cost."""
    playground = playground or find_playground()
    runs = playground.read_runs()
    if not runs:
        console.print("[yellow]no runs recorded yet[/yellow]")
        return

    latest = {}
    for record in runs:
        latest[record.get("step", "??")] = record

    def order(step_id: str) -> int:
        try:
            return step_index(step_id)
        except (KeyError, ValueError):
            return len(STEPS)

    table = Table(box=None, pad_edge=False)
    for column in ("step", "", "cost", "Δ", "actions", "wall", ""):
        table.add_column(column)

    previous: Optional[float] = None
    previous_problem: Optional[str] = None
    for step_id in sorted(latest, key=order):
        record = latest[step_id]
        cost = record.get("cost")
        known = order(step_id) < len(STEPS)
        step = get_step(step_id) if known else None
        problem = step["problem"] if step else None

        # A delta across a change of problem would be meaningless: step 05 moves
        # to a bigger house, so its cost has nothing to do with step 04's.
        delta = ""
        comparable = (
            previous is not None
            and problem is not None
            and problem == previous_problem
            and isinstance(cost, (int, float))
        )
        if comparable:
            change = cost - previous
            delta = (f"[green]{change:+.1f}[/green]" if change < 0
                     else f"[red]{change:+.1f}[/red]")

        wall = record.get("wall")
        table.add_row(
            step_id,
            escape(step["title"]) if step else "",
            f"{cost:.1f}s" if isinstance(cost, (int, float)) else "-",
            delta,
            str(len(record.get("actions", []))),
            f"{wall:.1f}s" if isinstance(wall, (int, float)) else "-",
            "" if record.get("goal_reached") else "[red]goal not reached[/red]",
        )
        if isinstance(cost, (int, float)):
            previous = cost
            previous_problem = problem
    console.print(table)


# -- pre-flight --------------------------------------------------------------


def cmd_doctor(console: Console) -> bool:
    """Check the things that ruin a live demo. Returns whether all passed."""
    checks: List[tuple[bool, str, str]] = []

    def record(ok: bool, label: str, detail: str = "") -> None:
        checks.append((ok, label, detail))

    cwd = Path.cwd()
    record(True, f"working directory: {cwd}",
           "mlflow.db, mlruns/ and the ProcTHOR cache all resolve from here")

    try:
        import railroad.bench  # noqa: F401
        record(True, "railroad[bench] installed")
    except ImportError as exc:
        record(False, "railroad[bench] missing", str(exc))

    try:
        from railroad.environment.procthor import is_available
        procthor_ok = is_available()
    except Exception as exc:  # pragma: no cover - defensive
        procthor_ok = False
        record(False, "railroad[procthor] check failed", str(exc))
    else:
        if procthor_ok:
            cache = cwd / "resources" / "procthor-10k" / "cache"
            seeds = sorted(p.stem.removeprefix("scene_")
                           for p in cache.glob("scene_*.pkl"))
            detail = (
                f"cached scenes: {', '.join(seeds)}" if seeds
                else "no cached scenes -- the first load starts Unity and "
                     "needs a GL context"
            )
            record(bool(seeds), "railroad[procthor] installed", detail)
        else:
            record(False, "railroad[procthor] missing",
                   "steps 06-07 need it; earlier steps do not")

    record(shutil.which("git") is not None, "git on PATH",
           "used to merge your live edits when advancing a step")
    record(shutil.which("ffmpeg") is not None, "ffmpeg on PATH",
           "needed only for --save-video")
    emacs = shutil.which("emacsclient")
    record(True, f"emacsclient: {'found' if emacs else 'not found'}",
           "optional; buffers still refresh via global-auto-revert-mode")
    record(sys.stdin.isatty() and sys.stdout.isatty(), "attached to a terminal",
           "the live dashboard and the watch pane both need one")

    try:
        playground = find_playground()
        record(True, f"playground: {playground.root}")
    except PlaygroundError:
        record(False, "no playground", "run 'railroad tutorial init'")

    all_ok = True
    for ok, label, detail in checks:
        mark = "[green]ok  [/green]" if ok else "[red]!!  [/red]"
        # Labels carry extras like "railroad[bench]", which rich would otherwise
        # read as a style tag and swallow.
        console.print(f"{mark}{escape(label)}")
        if detail:
            console.print(f"    [dim]{escape(detail)}[/dim]")
        all_ok = all_ok and ok
    return all_ok

"""The tutorial's commands, free of any CLI framework.

``railroad.cli`` is a thin click wrapper over these. Keeping the logic here
means it is testable without a terminal.

``notebook``, ``run``, ``bench`` and ``dashboard`` are short names for commands
you could type yourself, and each prints the longer one it expands to first.
That is the deal: convenient to type mid-sentence, and never pretending to be
something other than the ordinary CLI. Everything else here does the job a
script cannot do for itself -- saying what to type, and moving ``demo.py``
between steps without losing what you typed.
"""

from __future__ import annotations

import shutil
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import _advance as adv
from . import _launch as launch
from ._playground import (
    DEFAULT_DIRNAME,
    Playground,
    PlaygroundError,
    find_playground,
    init_playground,
)
from ._steps import (
    NOTEBOOK,
    STEPS,
    command_lines,
    get_step,
    neighbour,
    step_index,
)

Ask = Callable[[str], str]
"""Prompt the presenter; return ``"yes"``, ``"no"`` or ``"force"``."""


def _plain_ask(prompt: str) -> str:
    reply = input(f"{prompt} [y/N/f] ").strip().lower()
    return {"y": "yes", "yes": "yes", "f": "force"}.get(reply, "no")


def _local_edits(playground: Playground) -> Optional[tuple[int, int]]:
    """``(added, removed)`` against this step's snapshot, or ``None`` if clean."""
    pristine = playground.pristine_text(playground.current_step_id)
    current = playground.demo.read_text()
    if current == pristine:
        return None
    return adv.diff_stat(pristine, current)


def _latest_runs(playground: Playground) -> Dict[str, dict]:
    """The most recent recorded run of each step."""
    latest: Dict[str, dict] = {}
    for record in playground.read_runs():
        latest[record.get("step", "??")] = record
    return latest


# -- the card ----------------------------------------------------------------


def cmd_card(console: Console, playground: Optional[Playground] = None) -> None:
    """Where you are, and what to type. The default command.

    Deliberately how-to and nothing else: no talking points, no rationale.
    Those are ``notes``, which is a different question and a different command.
    """
    playground = playground or find_playground()
    step = get_step(playground.current_step_id)
    position = step_index(step["id"]) + 1

    console.print(f"[bold]step {step['id']} of {len(STEPS):02d}[/bold] · "
                  f"{escape(step['title'])}")
    console.print(f"  [dim]{escape(step['point'])}[/dim]")
    if Path.cwd().resolve() != playground.root:
        console.print(f"  [yellow]cd {playground.root}[/yellow]  "
                      f"[dim]these commands assume you are in there[/dim]")
    console.print()

    for label, command, comment in command_lines(step):
        # A trailing '#' comment is both an annotation and still a line you can
        # select and run. soft_wrap keeps rich from inserting newlines into it.
        note = f"  [dim]# {escape(comment)}[/dim]" if comment else ""
        console.print(f"  [bold]{label:<10}[/bold]{escape(command)}{note}",
                      soft_wrap=True)

    console.print()
    edits = _local_edits(playground)
    if edits:
        console.print(f"  [yellow]demo.py has local edits "
                      f"(+{edits[0]} -{edits[1]})[/yellow]  "
                      f"[dim]they survive 'next'; 'diff' shows them[/dim]")
    stale = _stale_snapshots(playground)
    if stale:
        console.print(
            f"  [yellow]{len(stale)} snapshot(s) differ from the installed "
            f"package ({', '.join(stale)})[/yellow]"
        )
        console.print("  [dim]The playground is frozen on purpose, so a talk "
                      "cannot change under you. 'uv run railroad tutorial init "
                      "--force' takes the new ones.[/dim]")
    coming = "peek" if position < len(STEPS) else "steps"
    console.print(f"  [dim]why: uv run railroad tutorial notes"
                  f"    next: uv run railroad tutorial {coming}[/dim]")


def _stale_snapshots(playground: Playground) -> List[str]:
    """Step ids whose playground copy no longer matches the installed package."""
    shipped = Path(__file__).parent / "steps"
    stale = []
    for step in STEPS:
        packaged = shipped / step["filename"]
        local = playground.steps_dir / step["filename"]
        if not packaged.exists() or not local.exists():
            continue
        if packaged.read_bytes() != local.read_bytes():
            stale.append(step["id"])
    return stale


def cmd_init(console: Console, directory: Optional[str], force: bool) -> None:
    root = Path(directory) if directory else Path.cwd() / DEFAULT_DIRNAME
    playground = init_playground(root, force=force)
    try:
        where = playground.root.relative_to(Path.cwd())
    except ValueError:
        where = playground.root
    console.print(f"[green]scaffolded[/green] {playground.root}")
    if playground.resources_dir.is_symlink():
        console.print("  [dim]resources/ linked, so the ProcTHOR scenes are "
                      "found from in there[/dim]")
    console.print()
    console.print(f"  [bold]cd {where}[/bold]   "
                  "[dim]everything runs from inside the playground[/dim]")
    console.print("  [bold]uv run railroad tutorial notebook[/bold]   "
                  "[dim]the language itself, before any of the arc[/dim]")
    console.print("  [bold]uv run railroad tutorial[/bold]   "
                  "[dim]what step you are on, and what to type[/dim]")


# -- running things ----------------------------------------------------------


def _echo(console: Console, argv: Sequence[str]) -> None:
    """Show the command before running it.

    ``soft_wrap`` leaves the wrapping to the terminal: rich would otherwise
    break a long command across lines by inserting newlines into it, and the
    point is that you can select the line and run it yourself.
    """
    console.print(f"[dim]$ {escape(launch.pretty(argv))}[/dim]", soft_wrap=True)


def cmd_notebook(
    console: Console,
    extra: Sequence[str] = (),
    playground: Optional[Playground] = None,
) -> launch.RunResult:
    """Open the language primer in Jupyter. Arguments pass through to it.

    The notebook comes before the arc: fluents, timed effects and transitions
    are things you want to poke at one at a time, which is the one thing a
    script is bad at. It blocks until you stop Jupyter -- that is Jupyter's
    shape, not ours.
    """
    playground = playground or find_playground()
    if not playground.notebook.exists():
        console.print(f"[yellow]{NOTEBOOK} is not in this playground[/yellow]  "
                      "[dim]uv run railroad tutorial init --force puts it back[/dim]")
        return launch.RunResult(1)
    if find_spec("jupyterlab") is None:
        console.print("[red]jupyterlab is not installed[/red]  "
                      "[dim]it comes with railroad\\[tutorial][/dim]")
        console.print("[dim]or, without installing anything: "
                      f"uv run --with jupyterlab jupyter lab {NOTEBOOK}[/dim]")
        return launch.RunResult(1)

    argv = launch.notebook_argv(extra)
    _echo(console, argv)
    result = launch.run(playground, argv)
    if not result.ok:
        console.print(f"[red]jupyter exited {result.returncode}[/red]")
    return result


def cmd_run(
    console: Console,
    extra: Sequence[str] = (),
    playground: Optional[Playground] = None,
) -> launch.RunResult:
    """Run ``demo.py``. Arguments are passed straight through to it."""
    playground = playground or find_playground()
    argv = launch.demo_argv(extra)
    _echo(console, argv)
    result = launch.run(playground, argv)
    if not result.ok:
        console.print(f"[red]demo.py exited {result.returncode}[/red]")
    return result


def cmd_bench(
    console: Console,
    extra: Sequence[str] = (),
    playground: Optional[Playground] = None,
) -> launch.RunResult:
    """Sweep this step, with the include/tag/experiment arguments filled in."""
    playground = playground or find_playground()
    step = get_step(playground.current_step_id)
    console.print(f"[bold]sweep[/bold] {escape(step['sweep'])}")
    argv = launch.bench_argv(extra)
    _echo(console, argv)
    result = launch.run(playground, argv)
    if not result.ok:
        console.print(f"[red]sweep exited {result.returncode}[/red]")
    return result


def cmd_dashboard(
    console: Console,
    *,
    playground: Optional[Playground] = None,
    port: int = launch.DEFAULT_PORT,
    host: str = "auto",
    status: bool = False,
    stop: bool = False,
) -> bool:
    """Start the dashboard, ask after it, or tear it down.

    It outlives the command that starts it, which is the whole point -- you
    leave it open in a browser tab and refresh as sweeps land. Its pid is
    recorded in the playground so the other two verbs have something to act on.
    """
    playground = playground or find_playground()
    running = launch.recorded(playground)

    if stop:
        pid = launch.stop(playground)
        if pid is None:
            console.print("[yellow]no dashboard running[/yellow]")
            return False
        console.print(f"[green]stopped[/green] dashboard (pid {pid})")
        return True

    if status:
        if running is None:
            console.print("[yellow]no dashboard running[/yellow]")
            tail = launch.log_tail(playground)
            if tail:
                console.print("[dim]last dashboard log:[/dim]")
            for line in tail:
                console.print(f"  [dim]{escape(line)}[/dim]")
            return False
        console.print(f"[green]running[/green] pid {running.pid}, port {running.port}")
        for line in launch.urls(running.port, running.host):
            console.print(f"[dim]{escape(line)}[/dim]")
        return True

    if running is not None:
        console.print(f"[yellow]already running[/yellow] (pid {running.pid}, "
                      f"port {running.port})")
        for line in launch.urls(running.port, running.host):
            console.print(f"[dim]{escape(line)}[/dim]")
        return True

    _echo(console, launch.dashboard_argv(host, port))
    started = launch.start(playground, host=host, port=port)
    console.print(f"[green]dashboard[/green] pid {started.pid}  "
                  f"[dim](this playground's results only)[/dim]")
    for line in launch.urls(port, host):
        console.print(f"[dim]{escape(line)}[/dim]")
    console.print(f"[dim]log: {launch.log_path(playground)}   "
                  f"stop it with: railroad tutorial dashboard --stop[/dim]")
    return True


# -- reference and explanation -----------------------------------------------


def cmd_steps(console: Console, playground: Optional[Playground] = None) -> None:
    """The whole arc, with the last recorded cost of each step."""
    playground = playground or find_playground()
    current = playground.current_step_id
    latest = _latest_runs(playground)

    table = Table(box=None, pad_edge=False)
    for column in ("", "step", "", "cost", "Δ", "sweep"):
        table.add_column(column, overflow="fold")

    previous: Optional[float] = None
    previous_problem: Optional[str] = None
    for step in STEPS:
        record = latest.get(step["id"], {})
        cost = record.get("cost")
        # A delta across a change of problem would be meaningless: step 05 moves
        # to a bigger house, so its cost has nothing to do with step 04's.
        delta = ""
        if (previous is not None and isinstance(cost, (int, float))
                and step["problem"] == previous_problem):
            change = cost - previous
            delta = (f"[green]{change:+.1f}[/green]" if change < 0
                     else f"[red]{change:+.1f}[/red]")
        if isinstance(cost, (int, float)):
            previous, previous_problem = cost, step["problem"]

        style = "bold" if step["id"] == current else "dim"
        needs = "  [procthor]" if step["requires"] else ""
        table.add_row(
            "[bold green]>[/bold green]" if step["id"] == current else " ",
            f"[{style}]{step['id']}[/{style}]",
            f"[{style}]{escape(step['title'] + needs)}[/{style}]",
            f"{cost:.1f}s" if isinstance(cost, (int, float)) else "",
            delta,
            f"[dim]{escape(step['sweep'])}[/dim]",
        )
    console.print(table)


def cmd_notes(
    console: Console,
    step_id: Optional[str] = None,
    playground: Optional[Playground] = None,
) -> None:
    """Why this step exists -- the talking points, so nothing is memorised."""
    playground = playground or find_playground()
    step = get_step(step_id or playground.current_step_id)
    console.print(f"[bold]{step['id']} · {escape(step['title'])}[/bold]")
    console.print(f"  {escape(step['point'])}")
    console.print()
    # A table rather than print(f"· {note}") purely for the hanging indent: a
    # wrapped talking point that starts again at column 0 is hard to scan.
    bullets = Table(box=None, pad_edge=False, show_header=False)
    bullets.add_column(width=3)
    bullets.add_column(overflow="fold")
    for note in step["notes"]:
        bullets.add_row("  ·", escape(note))
    console.print(bullets)
    if step["sweep"]:
        console.print()
        console.print(f"  [dim]sweep: {escape(step['sweep'])}[/dim]")


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


def _warn_about_extras(console: Console, step_id: str) -> None:
    """Say so before the step is applied, not when the run fails."""
    if get_step(step_id)["requires"] != "procthor":
        return
    try:
        from railroad.environment.procthor import is_available
        available = is_available()
    except ImportError:
        available = False
    if not available:
        console.print(
            "[yellow]this step needs railroad\\[procthor]; it will apply, but "
            "the run will fail until that extra is installed[/yellow]"
        )


def cmd_peek(console: Console, playground: Optional[Playground] = None) -> None:
    """The next patch and why it matters, without applying it."""
    playground = playground or find_playground()
    current = playground.current_step_id
    upcoming = neighbour(current, +1)
    if upcoming is None:
        console.print("[yellow]already on the last step[/yellow]")
        return
    render_patch(console, playground, current, upcoming["id"])
    console.print()
    cmd_notes(console, upcoming["id"], playground)


def cmd_goto(
    console: Console,
    target: str,
    *,
    playground: Optional[Playground] = None,
    ask: Optional[Ask] = None,
    force: bool = False,
    editor_sync: bool = True,
    show_patch: bool = True,
    show_card: bool = True,
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
    _warn_about_extras(console, target_id)

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
                      "'uv run railroad tutorial goto "
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
    if show_card:
        console.print()
        cmd_card(console, playground)
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


# -- pre-flight --------------------------------------------------------------


def cmd_doctor(console: Console) -> bool:
    """Check the things that ruin a live demo. Returns whether all passed."""
    checks: List[tuple[bool, str, str]] = []

    def record(ok: bool, label: str, detail: str = "") -> None:
        checks.append((ok, label, detail))

    cwd = Path.cwd()
    try:
        playground = find_playground()
    except PlaygroundError:
        playground = None

    resources = playground.resources_dir if playground else cwd / "resources"
    record(True, f"working directory: {cwd}",
           "run everything from inside the playground: mlflow.db, mlruns/ and "
           "media/ are all resolved from here")

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
            cache = resources / "procthor-10k" / "cache"
            seeds = sorted(p.stem.removeprefix("scene_")
                           for p in cache.glob("scene_*.pkl"))
            detail = (
                f"cached scenes: {', '.join(seeds)}" if seeds
                else f"no cached scenes under {cache} -- the first load starts "
                     "Unity and needs a GL context"
            )
            record(bool(seeds), "railroad[procthor] installed", detail)
        else:
            record(False, "railroad[procthor] missing",
                   "steps 06-07 need it; earlier steps do not")

    record(find_spec("jupyterlab") is not None, "jupyterlab installed",
           f"needed for 'tutorial notebook', which opens {NOTEBOOK}")
    record(shutil.which("git") is not None, "git on PATH",
           "used to merge your live edits when advancing a step")
    record(shutil.which("ffmpeg") is not None, "ffmpeg on PATH",
           "needed only for --save-video")
    emacs = shutil.which("emacsclient")
    record(True, f"emacsclient: {'found' if emacs else 'not found'}",
           "optional; buffers still refresh via global-auto-revert-mode")
    record(sys.stdin.isatty() and sys.stdout.isatty(), "attached to a terminal",
           "the live planner dashboard needs one")

    if playground is not None:
        record(True, f"playground: {playground.root}",
               "its own mlflow.db, mlruns/ and media/ live here")
    else:
        record(False, "no playground", "run 'uv run railroad tutorial init'")

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

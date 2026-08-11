"""Moving between steps without losing what you typed.

Advancing is two beats. First the canonical patch between the current step's
snapshot and the next one is rendered -- that diff is the thing you show the
audience. Then, on confirmation, it is three-way merged into the file as it
actually stands, so a constant you tuned mid-sentence survives the move.

The merge is ``git merge-file``, which happily operates on loose files outside
any repository. When git is unavailable the fallback is honest rather than
clever: if the working file is untouched, take the target snapshot verbatim; if
it has edits, refuse and say why.

Every write is preceded by a snapshot into ``.history/``, so ``undo`` always has
somewhere to go back to.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ._playground import Playground
from ._steps import get_step

HISTORY_INDEX = "index.jsonl"


class MergeConflict(Exception):
    """The presenter's edits and the step's patch touch the same lines."""

    def __init__(self, conflicts: int, merged_text: str) -> None:
        super().__init__(
            f"{conflicts} conflicting hunk(s) between your edits and the step patch"
        )
        self.conflicts = conflicts
        self.merged_text = merged_text


class MergeUnavailable(Exception):
    """No ``git`` to merge with, and the working file has edits to preserve."""


# -- history / undo ----------------------------------------------------------


def _index_path(playground: Playground) -> Path:
    return playground.history_dir / HISTORY_INDEX


def _read_index(playground: Playground) -> List[dict]:
    path = _index_path(playground)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _write_index(playground: Playground, entries: List[dict]) -> None:
    _index_path(playground).write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )


def snapshot_demo(playground: Playground, *, reason: str = "") -> Optional[Path]:
    """Copy ``demo.py`` into ``.history/`` and record which step it was on.

    Returns the snapshot path, or ``None`` when there is nothing to snapshot.
    """
    if not playground.demo.exists():
        return None
    playground.history_dir.mkdir(parents=True, exist_ok=True)
    entries = _read_index(playground)
    name = f"{len(entries):04d}.py"
    target = playground.history_dir / name
    shutil.copyfile(playground.demo, target)
    try:
        step = playground.current_step_id
    except Exception:
        step = ""
    entries.append(
        {"file": name, "step": step, "reason": reason, "ts": time.time()}
    )
    _write_index(playground, entries)
    return target


def undo(playground: Playground) -> dict:
    """Restore the most recent snapshot of ``demo.py`` and its step.

    Pops the entry, so repeated undos walk back through the history. The
    snapshot files themselves are kept.
    """
    entries = _read_index(playground)
    if not entries:
        raise FileNotFoundError("nothing to undo -- no snapshots recorded yet")
    entry = entries.pop()
    source = playground.history_dir / entry["file"]
    if not source.exists():
        raise FileNotFoundError(f"snapshot {source} is missing")
    shutil.copyfile(source, playground.demo)
    if entry.get("step"):
        playground.set_current_step(entry["step"])
    _write_index(playground, entries)
    return entry


# -- diffs -------------------------------------------------------------------


def unified(from_text: str, to_text: str, from_label: str, to_label: str) -> str:
    """A unified diff, empty when the two texts agree."""
    return "".join(
        difflib.unified_diff(
            from_text.splitlines(keepends=True),
            to_text.splitlines(keepends=True),
            fromfile=from_label,
            tofile=to_label,
        )
    )


def colorize(diff_text: str) -> Text:
    """Render a unified diff for the terminal."""
    if not diff_text:
        return Text("(no changes)", style="dim")
    out = Text()
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            style = "bold"
        elif line.startswith("@@"):
            style = "cyan"
        elif line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        else:
            style = "dim"
        out.append(line + "\n", style=style)
    return out


MIN_SIDE_BY_SIDE_WIDTH = 120
"""Below this each column is under ~52 characters, which is narrower than the
code it has to hold. A unified diff is the better answer down there."""

SYNTAX_ENV = "RAILROAD_TUTORIAL_SYNTAX"
"""Set to ``off`` for plain text, if a terminal makes a mess of the colours."""

SYNTAX_THEME = "ansi_dark"
"""Named ANSI colours only, so the diff comes out in the terminal's own palette
rather than a theme's idea of it. A hardcoded #rrggbb scheme looks right on the
laptop it was chosen on and wrong over ssh."""

REMOVED, ADDED = "on #5c151d", "on #14431f"
"""Backgrounds for the changed lines: a dark red and a dark green, tuned to sit
under syntax-coloured text without competing with it."""

REMOVED_BASIC, ADDED_BASIC = "on red", "on green"
"""The same idea in the eight ANSI colours. Louder than anyone would choose,
but the terminal's own red and green, so they land wherever the palette puts
them -- and, crucially, they are still *backgrounds*."""

DEEP_COLOUR = {"truecolor", "256"}
"""Colour systems that can render a dark tint as a dark tint. Below these, rich
rounds the tints above to the nearest of eight, which is black."""


def marking(color_system: Optional[str]) -> Tuple[str, str, str]:
    """``(removed, added, context)`` styles for the colour depth we have.

    Always a background, at every depth. The alternative -- colouring changed
    lines red and green -- throws the syntax highlighting away at exactly the
    point the eye is being sent, which is the one place it was worth having.
    Only the shade changes with the depth available.

    Context is never dimmed. Dimming it makes the common case of the screen
    (most lines are unchanged) read as fog, and the background already does
    all the work of saying which lines are the interesting ones.
    """
    if color_system in DEEP_COLOUR:
        return REMOVED, ADDED, ""
    return REMOVED_BASIC, ADDED_BASIC, ""


@lru_cache(maxsize=8)
def _highlight(text: str) -> Tuple[Text, ...]:
    """Lex *text* once. Cached because the result does not depend on width.

    The viewer re-renders on every resize, and re-lexing two files each time
    is most of what that costs. Callers copy before styling, so handing the
    same objects back repeatedly is safe.
    """
    lines = text.splitlines()
    try:
        syntax = Syntax(text, "python", theme=SYNTAX_THEME,
                        background_color="default")
        highlighted = list(syntax.highlight(text).split("\n"))
    except Exception:  # pragma: no cover - a lexer that will not
        return tuple(Text(line) for line in lines)
    # Never let the highlighter disagree with the diff about how many lines
    # there are: the alignment is computed from the plain text.
    highlighted = highlighted[:len(lines)]
    highlighted += [Text("") for _ in range(len(lines) - len(highlighted))]
    return tuple(highlighted)


def _highlighted(text: str) -> List[Text]:
    """The file, syntax-coloured, one :class:`Text` per line.

    Pygments comes with rich, so this costs no dependency. Highlighting the
    file whole rather than line by line is what lets the lexer see a docstring
    or a bracket that spans lines.
    """
    if os.environ.get(SYNTAX_ENV, "").strip().lower() in {"off", "0", "none"}:
        return [Text(line) for line in text.splitlines()]
    return list(_highlight(text))


def _cell(lines: List[Text], index: Optional[int], changed: bool,
          changed_style: str, context_style: str, width: int) -> Text:
    """One side of one row: the highlighted line, marked for its role."""
    if index is None:
        # No counterpart on this side. Fill the cell so the gap reads as a
        # block rather than as whitespace that might just be a short line --
        # but only when the marking is a background, or there is nothing there
        # for a foreground colour to colour.
        return (Text(" " * width, style=changed_style)
                if changed and changed_style.startswith("on ") else Text(""))
    line = lines[index].copy()
    style = changed_style if changed else context_style
    if style:
        if style.startswith("on "):
            # Pad out to the end of the cell before styling. Rich colours a
            # cell's padding from the *column's* style, not the cell's, so a
            # background otherwise stops dead at the last character and the
            # changed line reads as a ragged smear rather than a band. Padding
            # to a whole number of rows covers folded lines too.
            length = len(line.plain)
            rows = max(1, -(-length // width))  # ceil, without importing math
            line.pad_right(rows * width - length)
        # Stacks on top of the token colours rather than replacing them: a
        # background sets no foreground, so the syntax survives underneath.
        line.stylize(style)
    return line


def _aligned_rows(
    from_lines: List[str], to_lines: List[str]
) -> List[tuple[str, Optional[int], Optional[int]]]:
    """Pair the two files line for line as ``(tag, left_index, right_index)``.

    Unchanged lines pair up one to one. Inside a changed run the two sides are
    zipped as far as they go and the shorter one is padded with ``None``, which
    is what keeps the rest of the file level across the divider.
    """
    matcher = difflib.SequenceMatcher(None, from_lines, to_lines, autojunk=False)
    rows: List[tuple[str, Optional[int], Optional[int]]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        left = list(range(i1, i2))
        right = list(range(j1, j2))
        for offset in range(max(len(left), len(right))):
            rows.append((
                tag,
                left[offset] if offset < len(left) else None,
                right[offset] if offset < len(right) else None,
            ))
    return rows


def side_by_side(
    from_text: str,
    to_text: str,
    from_label: str,
    to_label: str,
    *,
    width: int,
    color_system: Optional[str] = "truecolor",
) -> Table | Text:
    """Both versions of the whole file, aligned, changes coloured.

    A unified diff answers "what changed"; this answers "what changed, and
    where does it sit in the file" -- which is the question an audience
    watching one file evolve is actually asking. The whole file is shown, not
    just hunks with context, so the shape of the step stays visible.

    Falls back to the unified diff on a terminal too narrow to hold two columns
    of code, because half-width Python is worse than a normal diff.
    """
    if width < MIN_SIDE_BY_SIDE_WIDTH:
        return colorize(unified(from_text, to_text, from_label, to_label))

    from_lines = from_text.splitlines()
    to_lines = to_text.splitlines()
    from_rich, to_rich = _highlighted(from_text), _highlighted(to_text)
    number_width = max(len(str(len(from_lines))), len(str(len(to_lines))), 2)
    # Two line-number columns, the divider, and the two spaces rich puts in
    # each of the four gaps between the five columns.
    column_width = max((width - 2 * number_width - 1 - 8) // 2, 20)

    table = Table(box=None, pad_edge=False, header_style="bold")
    # The line-number columns carry no blanket style: _number sets dim for
    # context and red/green for a change, and a column style would stack on
    # top and dim the marking back down again.
    table.add_column("", justify="right", width=number_width)
    table.add_column(Text(from_label), width=column_width, overflow="fold")
    table.add_column("", width=1, style="dim")
    table.add_column("", justify="right", width=number_width)
    table.add_column(Text(to_label), width=column_width, overflow="fold")

    removed, added, context = marking(color_system)
    for tag, left, right in _aligned_rows(from_lines, to_lines):
        changed = tag != "equal"
        table.add_row(
            _number(left, changed, "red"),
            _cell(from_rich, left, changed, removed, context, column_width),
            "│",
            _number(right, changed, "green"),
            _cell(to_rich, right, changed, added, context, column_width),
        )
    return table


def _number(index: Optional[int], changed: bool, style: str) -> Text:
    """The gutter. Carries the red/green the lines themselves gave up."""
    if index is None:
        return Text("")
    return Text(str(index + 1), style=style if changed else "dim")


def first_changed_line(from_text: str, to_text: str) -> int:
    """1-indexed line in *to_text* where it first departs from *from_text*."""
    before = from_text.splitlines()
    after = to_text.splitlines()
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for tag, _i1, _i2, j1, _j2 in matcher.get_opcodes():
        if tag != "equal":
            return j1 + 1
    return 1


def diff_stat(from_text: str, to_text: str) -> tuple[int, int]:
    """``(added, removed)`` line counts between two texts."""
    added = removed = 0
    for line in unified(from_text, to_text, "a", "b").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


# -- the merge ---------------------------------------------------------------


def merge_three_way(
    current: str, base: str, other: str, *, labels: tuple[str, str, str]
) -> tuple[str, int]:
    """Merge *base* -> *other* into *current*; return ``(text, conflicts)``.

    Raises :class:`MergeUnavailable` when git is absent and *current* has
    diverged from *base* (with no divergence there is nothing to preserve, so
    the caller can simply take *other*).
    """
    git = shutil.which("git")
    if git is None:
        if current == base:
            return other, 0
        raise MergeUnavailable(
            "git is not on PATH, so your edits cannot be merged with the step patch"
        )

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = {}
        for name, text in (("current", current), ("base", base), ("other", other)):
            paths[name] = root / name
            paths[name].write_text(text, encoding="utf-8")
        result = subprocess.run(
            [
                git, "merge-file", "-p",
                "-L", labels[0], "-L", labels[1], "-L", labels[2],
                str(paths["current"]), str(paths["base"]), str(paths["other"]),
            ],
            capture_output=True,
            text=True,
        )
    # git merge-file exits with the number of conflicts, or <0 on error.
    if result.returncode < 0:
        raise MergeUnavailable(f"git merge-file failed: {result.stderr.strip()}")
    return result.stdout, result.returncode


@dataclass(frozen=True)
class AdvanceResult:
    """What :func:`advance` did, for the caller to narrate."""

    from_step: str
    to_step: str
    added: int
    removed: int
    changed_line: int
    took_pristine: bool
    preserved_edits: bool


def advance(
    playground: Playground,
    target_step_id: str,
    *,
    force: bool = False,
    editor_sync: bool = True,
) -> AdvanceResult:
    """Move ``demo.py`` to *target_step_id*, keeping live edits where possible.

    ``force`` takes the target snapshot verbatim, discarding local edits (still
    recoverable via :func:`undo`). Without it, a conflicting merge raises
    :class:`MergeConflict` rather than writing conflict markers into the file
    the presenter is about to run.
    """
    from_step = playground.current_step_id
    to_step = get_step(target_step_id)["id"]

    current = playground.demo.read_text(encoding="utf-8")
    base = playground.pristine_text(from_step)
    other = playground.pristine_text(to_step)
    has_edits = current != base

    if force:
        merged, took_pristine = other, True
    else:
        merged, conflicts = merge_three_way(
            current, base, other,
            labels=(f"your demo.py (step {from_step})",
                    f"step {from_step}",
                    f"step {to_step}"),
        )
        if conflicts:
            raise MergeConflict(conflicts, merged)
        took_pristine = False

    snapshot_demo(playground, reason=f"advance {from_step}->{to_step}")
    playground.demo.write_text(merged, encoding="utf-8")
    playground.set_current_step(to_step)

    added, removed = diff_stat(base, other)
    line = first_changed_line(current, merged)
    if editor_sync:
        sync_editor(playground.demo, line)
    return AdvanceResult(
        from_step=from_step,
        to_step=to_step,
        added=added,
        removed=removed,
        changed_line=line,
        took_pristine=took_pristine,
        preserved_edits=has_edits and not took_pristine,
    )


# -- editor ------------------------------------------------------------------


_ELISP = """\
(let ((b (get-file-buffer {path})))
  (when b
    (with-current-buffer b (revert-buffer t t t))
    (let ((w (get-buffer-window b t)))
      (when w
        (with-selected-window w
          (goto-char (point-min))
          (forward-line {line})
          (recenter))))))
"""


def sync_editor(path: Path, line: int) -> bool:
    """Nudge a running emacs to reload *path* and put point on *line*.

    Best-effort and silent: no server, no emacsclient, or any error at all just
    means the buffer refreshes on its own via ``global-auto-revert-mode``.
    Never starts a daemon.
    """
    exe = shutil.which("emacsclient")
    if exe is None:
        return False
    elisp = _ELISP.format(path=json.dumps(str(path)), line=max(line - 1, 0))
    try:
        result = subprocess.run(
            [exe, "--no-wait", "--eval", elisp],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0

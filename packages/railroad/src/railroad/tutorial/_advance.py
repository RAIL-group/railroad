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
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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

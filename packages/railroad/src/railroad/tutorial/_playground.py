"""The playground directory: one editable file plus the state around it.

Layout::

    railroad-tutorial/
      demo.py        the only file the presenter opens
      README.md      the key map, for the morning of
      .steps/        pristine snapshot per step, copied at init
      .history/      snapshot of demo.py before every advance (undo)
      .state.json    which step demo.py is currently on
      runs.jsonl     one record per run, for `compare`

Snapshots are *copied* into ``.steps/`` rather than read from the installed
package so the playground is frozen for the duration of a talk: upgrading
railroad mid-week cannot change the diff you rehearsed.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._steps import STEPS, get_step

ENV_DIR = "RAILROAD_TUTORIAL_DIR"
DEFAULT_DIRNAME = "railroad-tutorial"
STATE_FILE = ".state.json"
DEMO_FILE = "demo.py"


class PlaygroundError(RuntimeError):
    """Raised when a playground cannot be found or is malformed."""


def _shipped_steps_dir() -> Path:
    return Path(__file__).parent / "steps"


@dataclass(frozen=True)
class Playground:
    """A scaffolded tutorial directory."""

    root: Path

    @property
    def demo(self) -> Path:
        return self.root / DEMO_FILE

    @property
    def steps_dir(self) -> Path:
        return self.root / ".steps"

    @property
    def history_dir(self) -> Path:
        return self.root / ".history"

    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILE

    @property
    def runs_path(self) -> Path:
        return self.root / "runs.jsonl"

    # -- state ---------------------------------------------------------------

    def read_state(self) -> Dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError) as exc:
            raise PlaygroundError(f"unreadable {self.state_path}: {exc}") from exc

    def write_state(self, state: Dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, indent=2) + "\n")

    @property
    def current_step_id(self) -> str:
        step_id = self.read_state().get("step")
        if not isinstance(step_id, str):
            raise PlaygroundError(f"no current step recorded in {self.state_path}")
        return step_id

    def set_current_step(self, step_id: str) -> None:
        state = self.read_state()
        state["step"] = get_step(step_id)["id"]
        self.write_state(state)

    # -- snapshots -----------------------------------------------------------

    def pristine_path(self, step_id: str) -> Path:
        path = self.steps_dir / get_step(step_id)["filename"]
        if not path.exists():
            raise PlaygroundError(
                f"missing snapshot {path}; re-run 'railroad tutorial init --force'"
            )
        return path

    def pristine_text(self, step_id: str) -> str:
        return self.pristine_path(step_id).read_text()

    # -- runs ----------------------------------------------------------------

    def append_run(self, record: Dict[str, Any]) -> None:
        with self.runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def read_runs(self) -> List[Dict[str, Any]]:
        if not self.runs_path.exists():
            return []
        records = []
        for line in self.runs_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue  # a half-written line from an interrupted run
        return records


def is_playground(path: Path) -> bool:
    """True when *path* looks like a scaffolded playground."""
    return (path / STATE_FILE).is_file() and (path / DEMO_FILE).is_file()


def _candidate_roots(explicit: Optional[Path]) -> List[Path]:
    if explicit is not None:
        return [explicit]
    candidates: List[Path] = []
    from_env = os.environ.get(ENV_DIR)
    if from_env:
        candidates.append(Path(from_env))
    cwd = Path.cwd()
    candidates.append(cwd)
    candidates.append(cwd / DEFAULT_DIRNAME)
    # Running `python .../demo.py` directly from anywhere.
    if sys.argv and sys.argv[0]:
        candidates.append(Path(sys.argv[0]).resolve().parent)
    return candidates


def find_playground(explicit: Optional[Path] = None) -> Playground:
    """Locate the playground, or explain how to make one.

    Checks, in order: an explicit path, ``$RAILROAD_TUTORIAL_DIR``, the current
    directory, ``./railroad-tutorial``, and the directory holding the script
    being run (so ``python tutorial/demo.py`` works from anywhere).
    """
    for candidate in _candidate_roots(explicit):
        resolved = candidate.expanduser()
        if is_playground(resolved):
            return Playground(resolved.resolve())
    raise PlaygroundError(
        "no tutorial playground found. Run 'railroad tutorial init' from the "
        "repository root (the benchmark database and the ProcTHOR scene cache "
        "are both resolved relative to the working directory), or point "
        f"${ENV_DIR} at an existing one."
    )


README = """\
# railroad tutorial

`demo.py` is the only file you edit. Keep it open in your editor with
`(global-auto-revert-mode 1)` so it refreshes when a step is applied.

Run the pane next to it:

    railroad tutorial watch

    n  next step (shows the patch first, then merges your edits)
    p  previous step
    k  peek at the next patch without applying it
    d  diff: what you have changed since this step's snapshot
    b  run this step's benchmark sweep
    o  open the benchmark dashboard
    c  compare every run so far
    u  undo the last advance
    q  quit

Saving `demo.py` re-runs it. Everything is recoverable: `u` restores the file
as it was before the last advance, and `railroad tutorial goto <n> --force`
takes the pristine snapshot of any step.
"""


def init_playground(root: Path, *, force: bool = False) -> Playground:
    """Scaffold a playground at *root* and put it on step 00."""
    root = root.expanduser()
    if root.exists() and not root.is_dir():
        raise PlaygroundError(f"{root} exists and is not a directory")
    if is_playground(root) and not force:
        raise PlaygroundError(
            f"{root} is already a tutorial playground; pass --force to reset it "
            "(this overwrites demo.py -- your edits are snapshotted to .history first)"
        )

    playground = Playground(root.resolve())
    playground.steps_dir.mkdir(parents=True, exist_ok=True)
    playground.history_dir.mkdir(parents=True, exist_ok=True)

    shipped = _shipped_steps_dir()
    for step in STEPS:
        source = shipped / step["filename"]
        if not source.exists():
            raise PlaygroundError(f"packaged step snapshot missing: {source}")
        shutil.copyfile(source, playground.steps_dir / step["filename"])

    if playground.demo.exists():
        # Never destroy work without a way back, even on --force.
        from ._advance import snapshot_demo

        snapshot_demo(playground, reason="init-force")

    first = STEPS[0]["id"]
    shutil.copyfile(playground.pristine_path(first), playground.demo)
    playground.write_state({"step": first})
    (root / "README.md").write_text(README)
    playground.runs_path.touch()
    return playground

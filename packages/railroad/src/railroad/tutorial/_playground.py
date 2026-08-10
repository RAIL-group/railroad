"""The playground directory: one editable file plus the state around it.

Layout::

    railroad-tutorial/
      demo.py        the only file you open
      README.md      what to type, for the morning of
      media/         plots and videos; the dashboard serves this
      resources ->   symlink to the ProcTHOR scenes, so nothing is copied
      .steps/        pristine snapshot per step, copied at init
      .history/      snapshot of demo.py before every advance (undo)
      .state.json    which step demo.py is currently on
      runs.jsonl     one record per run, for the step list

You work *inside* this directory, and that is the whole isolation story:
``mlflow.db``, ``mlruns/``, ``.benchmark_cache/`` and ``media/`` are all
resolved relative to the working directory, so the tutorial gets its own of
each and cannot disturb results you already had.

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

from ._steps import DEMO_FILE, MEDIA_DIR, STEPS, get_step

ENV_DIR = "RAILROAD_TUTORIAL_DIR"
DEFAULT_DIRNAME = "railroad-tutorial"
STATE_FILE = ".state.json"


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

    @property
    def media_dir(self) -> Path:
        """Where plots and videos go; also what the dashboard serves."""
        return self.root / MEDIA_DIR

    @property
    def resources_dir(self) -> Path:
        """The ProcTHOR scenes -- a symlink, so nothing is copied."""
        return self.root / "resources"

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
        "repository root -- that is where the ProcTHOR resources are linked "
        f"from -- or point ${ENV_DIR} at an existing one."
    )


README = """\
# railroad tutorial

Work from inside this directory. `demo.py` is the only file you edit; keep it
open with `(global-auto-revert-mode 1)` so it refreshes when a step is applied.

    railroad tutorial            what step you are on, and what to type

Everything you run is an ordinary command. Nothing here is a special mode:

    python demo.py               run it, live
    python demo.py --list        the parameter cases this step sweeps
    python demo.py --case 4      run one of them by hand, live
    python demo.py --video x.mp4 ...and record it, into media/

    railroad benchmarks run -i demo.py --tags tutorial \\
        --experiment railroad-tutorial --parallel 12
    railroad benchmarks dashboard

Moving through the tutorial is the part a script does for you:

    railroad tutorial peek       the next patch, and why it matters
    railroad tutorial next       show that patch, then merge it into demo.py
    railroad tutorial notes      the talking points for this step
    railroad tutorial diff       what you have changed since the snapshot
    railroad tutorial undo       put demo.py back the way it was

Advancing three-way merges the step's patch into the file as it actually
stands, so a constant you tuned mid-sentence survives the move. On a genuine
conflict it refuses rather than writing markers into a file you are about to
run; `railroad tutorial goto <n> --force` takes the snapshot verbatim, and
`undo` still works afterwards.

## This directory is its own world

Because you run from here, `mlflow.db`, `mlruns/`, `.benchmark_cache/` and
`media/` are all created *inside* it. Sweeps cannot disturb results you already
had, and the dashboard shows this tutorial and nothing else. Delete the
directory and it is all gone.

`resources` is a symlink rather than a copy -- the ProcTHOR scenes are a
gigabyte and nobody wants two of them. Steps 06 and 07 need
`railroad[procthor]`; run `railroad tutorial doctor` the morning of.
"""


def _link_resources(root: Path, source: Path) -> bool:
    """Point the playground at the ProcTHOR scenes without copying them.

    A symlink rather than an environment variable, so that a command typed by
    hand from this directory finds the cached scenes exactly as one run from
    the repository root would.
    """
    link = root / "resources"
    if link.exists() or link.is_symlink() or not source.is_dir():
        return link.is_symlink() or link.is_dir()
    try:
        link.symlink_to(source.resolve(), target_is_directory=True)
    except OSError:
        return False
    return True


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
    playground.media_dir.mkdir(exist_ok=True)
    _link_resources(playground.root, Path.cwd() / "resources")
    (root / "README.md").write_text(README)
    playground.runs_path.touch()
    return playground

"""A guided, terminal-only tour of railroad, driven from one editable file.

``railroad tutorial init`` scaffolds a playground whose ``demo.py`` is the only
file you open. ``railroad tutorial watch`` runs beside your editor: it re-runs
``demo.py`` whenever you save it, and single keypresses advance through the
steps, fire the current step's benchmark sweep, and open the dashboard.

Advancing shows the canonical diff first -- that diff is the unit of the talk --
and then three-way merges it into whatever you currently have on disk, so a
value you tuned live survives moving on. See :mod:`railroad.tutorial._advance`.

Step snapshots live in ``steps/`` and are plain, runnable Python; nothing in
them depends on the tutorial machinery except the closing :func:`report` call.
"""

from ._playground import (
    ENV_DIR,
    DEFAULT_DIRNAME,
    Playground,
    PlaygroundError,
    find_playground,
    init_playground,
    is_playground,
)
from ._media import media_args
from ._report import report
from ._steps import STEPS, StepInfo, get_step, neighbour, step_ids, step_index

__all__ = [
    "DEFAULT_DIRNAME",
    "ENV_DIR",
    "Playground",
    "PlaygroundError",
    "STEPS",
    "StepInfo",
    "find_playground",
    "get_step",
    "init_playground",
    "is_playground",
    "media_args",
    "neighbour",
    "report",
    "step_ids",
    "step_index",
]

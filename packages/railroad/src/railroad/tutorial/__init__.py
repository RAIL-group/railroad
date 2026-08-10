"""A guided, terminal-only tour of railroad, driven from one editable file.

``uv run railroad tutorial init`` scaffolds a playground; you ``cd`` into it
and work there. ``demo.py`` is the only file you open, and everything you run
is an ordinary command -- ``uv run python demo.py``, ``uv run railroad
benchmarks run``, ``uv run railroad benchmarks dashboard`` -- printed for you
by ``uv run railroad tutorial`` rather than hidden behind a wrapper.

The tutorial itself does the one thing a script cannot do for you: advancing.
``uv run railroad tutorial next`` shows the canonical diff between two steps --
that diff is the unit of the talk -- and then three-way merges it into whatever
you currently have on disk, so a value you tuned live survives moving on. See
:mod:`railroad.tutorial._advance`.

Step snapshots in ``steps/`` are plain, runnable Python. What they import from
here is plumbing only (:mod:`railroad.tutorial._harness`): a dashboard that
knows whether anyone is watching, and a ``main`` that runs one case of the
step's own benchmark sweep by hand.
"""

from ._harness import dashboard, main, result, show_plots
from ._playground import (
    ENV_DIR,
    DEFAULT_DIRNAME,
    MEDIA_DIR,
    Playground,
    PlaygroundError,
    find_playground,
    init_playground,
    is_playground,
)
from ._steps import (
    EXPERIMENT,
    RUNNER,
    STEPS,
    Command,
    StepInfo,
    command_lines,
    get_step,
    neighbour,
    step_ids,
    step_index,
)

__all__ = [
    "Command",
    "DEFAULT_DIRNAME",
    "ENV_DIR",
    "EXPERIMENT",
    "MEDIA_DIR",
    "Playground",
    "PlaygroundError",
    "RUNNER",
    "STEPS",
    "StepInfo",
    "command_lines",
    "dashboard",
    "find_playground",
    "get_step",
    "init_playground",
    "is_playground",
    "main",
    "neighbour",
    "result",
    "show_plots",
    "step_ids",
    "step_index",
]

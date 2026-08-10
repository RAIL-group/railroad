"""The plumbing a step file imports, so the rest of it can be language.

Every step is one ``@benchmark`` function serving two callers. Run
``uv run python demo.py`` and it plans once, live, with the terminal dashboard. Point
``uv run railroad benchmarks run`` at the same file and it plans every case in the
sweep, many times over, in worker processes with no terminal at all. The three
helpers here are what differ between those two worlds:

- :func:`dashboard` picks a live view or a recording console;
- :func:`show_plots` draws the trajectory only when somebody is watching;
- :func:`result` returns the metrics dict both callers want.

:func:`main` is the ``if __name__ == "__main__"`` half: it builds a
:class:`BenchmarkCase` from the sweep's own case list, so ``--case 4`` runs one
point of the sweep by hand, with the live view, at full speed.
"""

from __future__ import annotations

import argparse
import re
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ._playground import PlaygroundError, find_playground
from ._steps import MEDIA_DIR

_STEP_IN_NAME = re.compile(r"\bs(\d{2})_")


# -- the two worlds ----------------------------------------------------------


@contextmanager
def dashboard(case: Any, goal: Any, env: Any, *, fluent_filter=None) -> Iterator[Any]:
    """A :class:`PlannerDashboard` suited to whoever is running this case.

    Live, the dashboard takes the terminal: goal tree, timeline, MCTS trace.
    Under the benchmark runner it writes to a recording console instead, which
    becomes the ``log_html`` artifact the results dashboard shows. The
    ``force_interactive=False`` is load-bearing there -- ``force_terminal`` is
    what makes the recording come out with colour, and it would otherwise
    convince rich to start a live view inside a worker process.
    """
    from rich.console import Console

    from railroad.dashboard import PlannerDashboard

    live = bool(getattr(case, "live", False))
    if live:
        view = PlannerDashboard(goal, env, fluent_filter=fluent_filter)
    else:
        view = PlannerDashboard(
            goal, env,
            fluent_filter=fluent_filter,
            console=Console(record=True, force_terminal=True, width=120),
            print_on_exit=False,
            force_interactive=False,
        )
    with view:
        yield view
    if not live:
        view.print_history()


def show_plots(view: Any, case: Any, **kwargs: Any) -> None:
    """Draw whatever the command line asked for -- for a live run only.

    A sweep runs dozens of these at once in processes with no display, and
    every one of them would be writing over the same file, so it does nothing
    there. Extra keywords go to :meth:`PlannerDashboard.show_plots`;
    ``location_coords`` is the useful one, for environments that do not carry
    coordinates of their own.
    """
    if not getattr(case, "live", False):
        return
    view.show_plots(**{**getattr(case, "media", {}), **kwargs})


def result(view: Any, **extra: Any) -> Dict[str, Any]:
    """The metrics both callers want, from the dashboard's own record.

    ``plan_cost`` is the environment's clock -- the makespan, not the wall
    clock -- which is the number the whole tutorial is about. Anything a step
    wants to add (a count of searches, say) comes through ``extra``.
    """
    actions = [name for name, _ in view.actions_taken]
    state = view._env.state
    return {
        "success": bool(view.goal.evaluate(state.fluents)),
        "plan_cost": float(state.time),
        "wall_time": perf_counter() - view._start_time,
        "actions_count": len(actions),
        "searches": len([name for name in actions if name.startswith("search")]),
        "log_html": view.console.export_html(inline_styles=True),
        **extra,
    }


# -- running one case by hand ------------------------------------------------


def _media_path(name: str) -> str:
    """A bare filename goes where the dashboard serves it; a path is left alone."""
    if Path(name).parent != Path("."):
        return name
    Path(MEDIA_DIR).mkdir(exist_ok=True)
    return str(Path(MEDIA_DIR) / name)


def _describe(params: Dict[str, Any]) -> str:
    return "  ".join(f"{key}={value}" for key, value in params.items()) or "(no parameters)"


def _step_id(bench: Any) -> str:
    """``demo::s03_hidden_objects`` -> ``03``, falling back to the playground."""
    match = _STEP_IN_NAME.search(getattr(bench, "name", "") or "")
    if match:
        return match.group(1)
    try:
        return find_playground().current_step_id
    except PlaygroundError:
        return "??"


def _announce(step: str, outcome: Dict[str, Any]) -> None:
    """The headline, and a line in ``runs.jsonl`` for the step list."""
    status = "goal reached" if outcome.get("success") else "goal NOT reached"
    print(
        f"[tutorial] step {step} · cost {outcome.get('plan_cost', float('nan')):.1f}s"
        f" · {outcome.get('actions_count', 0)} actions"
        f" · {outcome.get('wall_time', 0.0):.1f}s wall · {status}"
    )
    try:
        playground = find_playground()
    except PlaygroundError:
        return  # run from outside a playground: print, but nowhere to file it
    playground.append_run({
        "step": step,
        "cost": outcome.get("plan_cost"),
        "wall": outcome.get("wall_time"),
        "actions": outcome.get("actions_count"),
        "goal_reached": bool(outcome.get("success")),
    })


def _parse(argv: Optional[Sequence[str]], cases: List[Dict[str, Any]]) -> argparse.Namespace:
    """This step's own command line: two flags of its own, plus the shared ones.

    The plot and video flags are the ones ``railroad example`` has always had
    -- ``--save-plot``, ``--save-video``, ``--video-dpi`` and the rest --
    declared once in :mod:`railroad.dashboard` and borrowed here, so there is
    one spelling to remember across the whole tool. Bare filenames land in
    ``media/``, which is what the results dashboard serves.
    """
    from railroad.dashboard import add_to_argparse

    parser = argparse.ArgumentParser(
        description="Run one case of this step live. The sweep over every case "
                    "is 'uv run railroad tutorial bench'.",
    )
    parser.add_argument("--case", type=int, default=0, metavar="N",
                        help=f"which case to run (0-{max(len(cases) - 1, 0)})")
    parser.add_argument("--list", action="store_true",
                        help="print the cases and exit")
    add_to_argparse(parser)
    return parser.parse_args(None if argv is None else list(argv))


def main(bench: Any, argv: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    """Run one case of *bench* live; return its metrics.

    *bench* is what ``@benchmark`` handed back: a ``Benchmark`` holding the
    function and its cases. Running a case rather than a hardcoded
    configuration means the thing you demonstrate and the thing you sweep can
    never drift apart -- and that ``--case`` gives you every point of the sweep
    as something you can watch.
    """
    from railroad.bench import BenchmarkCase

    cases: List[Dict[str, Any]] = list(getattr(bench, "cases", None) or [{}])
    args = _parse(argv, cases)

    if args.list:
        print(f"{bench.name}: {len(cases)} case(s)")
        for index, params in enumerate(cases):
            print(f"  {index:>2}  {_describe(params)}")
        return None

    if not 0 <= args.case < len(cases):
        raise SystemExit(f"--case must be between 0 and {len(cases) - 1}")

    from railroad.dashboard import media_kwargs

    params = dict(cases[args.case])
    case = BenchmarkCase(bench.name, args.case, 0, params)
    case.live = True
    case.media = media_kwargs(vars(args), relocate=_media_path)

    print(f"case {args.case}: {_describe(params)}")
    outcome = bench.fn(case)
    _announce(_step_id(bench), outcome)
    return outcome

"""The one line every ``demo.py`` ends with.

``report(dashboard)`` prints the headline number and appends a record to
``runs.jsonl`` so ``railroad tutorial compare`` can put successive steps beside
each other. It is deliberately forgiving: a demo that is run outside a
playground still prints, it just has nowhere to file the result.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Optional

from ._playground import PlaygroundError, find_playground


def _final_state(dashboard: Any) -> Any:
    """The state whose ``time`` the dashboard prints as 'Total cost'.

    Reads the dashboard's own environment handle so the number filed here is
    the number on screen. Falls back to the last history entry when a caller
    passes something dashboard-shaped that has no environment.
    """
    env = getattr(dashboard, "_env", None)
    if env is not None:
        return env.state
    return None


def report(dashboard: Any, step: Optional[str] = None) -> None:
    """Print this run's headline and file it under ``runs.jsonl``.

    Call after the ``with PlannerDashboard(...)`` block, so the dashboard has
    already printed its summary and the live view has been torn down.
    """
    state = _final_state(dashboard)
    history = getattr(dashboard, "history", []) or []

    if state is not None:
        cost = float(state.time)
        reached = bool(dashboard.goal.evaluate(state.fluents))
    elif history:
        cost = float(history[-1]["time"])
        reached = bool(history[-1].get("goal_satisfied", False))
    else:
        cost, reached = float("nan"), False

    actions = [name for name, _ in getattr(dashboard, "actions_taken", [])]
    started = getattr(dashboard, "_start_time", None)
    wall = perf_counter() - started if started is not None else None

    try:
        playground = find_playground()
        step_id = step or playground.current_step_id
    except PlaygroundError:
        playground, step_id = None, (step or "??")

    status = "goal reached" if reached else "goal NOT reached"
    wall_text = f" · {wall:.1f}s wall" if wall is not None else ""
    print(
        f"[tutorial] step {step_id} · cost {cost:.1f}s · "
        f"{len(actions)} actions{wall_text} · {status}"
    )

    if playground is None:
        return
    playground.append_run(
        {
            "step": step_id,
            "cost": cost,
            "wall": wall,
            "actions": actions,
            "goal_reached": reached,
        }
    )

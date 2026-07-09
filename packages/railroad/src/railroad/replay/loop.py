"""The shared plan->act loop, silent or dashboard-driven.

Deployment and replay run the *same* loop so the counterfactual holds the
planner fixed and varies only the policy. :func:`plan_act_loop` is the headless
core (no printing, no plots); :func:`run_dashboard_loop` wraps it in the standard
:class:`~railroad.dashboard.PlannerDashboard` for the rendered videos. The MCTS
knobs live in one :class:`MctsConfig` that both phases share, so their planners
cannot drift apart by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from railroad.core import get_action_by_name

# (env, actions, goal) -> action name (or "NONE"/"" to stop). The seam that lets
# tests inject a deterministic selector in place of production MCTS.
ActionSelector = Callable[[Any, list, "Any"], str]


@dataclass(frozen=True)
class MctsConfig:
    """MCTS knobs for :func:`mcts_selector`.

    One config is shared by a deployment and every replay of it, so the planner
    is held fixed while only the policy varies (the whole point of replay). Pass
    the same instance to the deployment loop and to :func:`run_replay`; override
    per candidate only when a study deliberately varies the planner too.
    """

    iterations: int = 4000
    c: float = 10.0
    max_depth: int = 20
    heuristic_multiplier: float = 5.0


def mcts_selector(config: MctsConfig) -> ActionSelector:
    """The default action selector: production MCTS with *config*.

    The returned callable stashes the planner it just built on its own
    ``last_planner`` attribute so :func:`run_dashboard_loop` can hand it to the
    dashboard (which reads the search trace and heuristic from it).
    """
    from railroad.planner import MCTSPlanner

    def select(env: Any, actions: list, goal: Any) -> str:
        planner = MCTSPlanner(actions)
        name = planner(
            env.state,
            goal,
            max_iterations=config.iterations,
            c=config.c,
            max_depth=config.max_depth,
            heuristic_multiplier=config.heuristic_multiplier,
        )
        select.last_planner = planner  # type: ignore[attr-defined]
        return name

    select.last_planner = None  # type: ignore[attr-defined]
    return select


def plan_act_loop(
    env: Any,
    goal: Any,
    *,
    select: ActionSelector,
    max_iterations: int,
    dashboard: Optional[Any] = None,
) -> str:
    """Drive the plan->act loop until the goal or a dead end; return why it stopped.

    Check the goal, get applicable actions, ask *select* for one, act. When
    *dashboard* is given, echo progress to its console, feed ``env.act`` the
    dashboard's step callback, and update it after each action (the dashboard
    reads the planner from ``select.last_planner``). Terminations:
    ``goal_reached`` / ``no_actions`` / ``planner_none`` / ``max_iterations``.
    """
    act_callback = dashboard.make_act_callback() if dashboard is not None else None

    def announce(markup: str) -> None:
        if dashboard is not None:
            dashboard.console.print(markup)

    termination = "max_iterations"
    for _ in range(max_iterations):
        if goal.evaluate(env.state.fluents):
            termination = "goal_reached"
            announce("[green]Goal reached![/green]")
            break
        actions = env.get_actions()
        if not actions:
            termination = "no_actions"
            announce("[red]No actions available — stuck.[/red]")
            break
        name = select(env, actions, goal)
        if name in ("NONE", "", None):
            termination = "planner_none"
            announce("[yellow]Planner returned NONE — stopping.[/yellow]")
            break
        env.act(get_action_by_name(actions, name), loop_callback_fn=act_callback)
        if dashboard is not None:
            planner = getattr(select, "last_planner", None)
            if planner is not None:
                dashboard.update(planner, name)
    if termination == "max_iterations" and goal.evaluate(env.state.fluents):
        termination = "goal_reached"
    return termination


def run_dashboard_loop(
    env: Any,
    goal: Any,
    *,
    select: ActionSelector,
    max_iterations: int,
    fluent_keywords: Sequence[str],
    scene: Optional[Any] = None,
    save_video: Optional[str] = None,
    label: str = "",
) -> str:
    """Run :func:`plan_act_loop` inside the standard :class:`PlannerDashboard`.

    Builds the dashboard with a fluent filter derived from *fluent_keywords*,
    exposes *scene* to it for the overhead map (if given), and renders a video to
    *save_video* on exit. Returns the loop's termination reason. Imported lazily
    so the headless path stays GL/plot free.
    """
    from railroad.dashboard import PlannerDashboard

    if scene is not None:
        env.scene = scene  # exposed to the dashboard for the overhead map

    def fluent_filter(fluent: Any) -> bool:
        return any(keyword in fluent.name for keyword in fluent_keywords)

    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        if label:
            dashboard.console.print(f"[bold]{label}[/bold]")
        termination = plan_act_loop(
            env,
            goal,
            select=select,
            max_iterations=max_iterations,
            dashboard=dashboard,
        )

    dashboard.show_plots(save_video=save_video, video_fps=10, video_dpi=130)
    return termination

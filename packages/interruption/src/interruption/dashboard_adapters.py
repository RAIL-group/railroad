from collections.abc import Callable
from typing import Union

from railroad.core import Action, Goal, State, Fluent, LiteralGoal


class AstarDashboardPlanner:
    """
    Adapts astar_search (a one-shot planner) to the DashboardPlanner protocol
    expected by railroad.dashboard.PlannerDashboard, which is otherwise built
    around incremental, per-step planners like MCTSPlanner.
    """

    def __init__(
        self,
        actions: list[Action],
        heuristic_fn: float | Callable[[State, Goal, list[Action]], float] = 0,
    ):
        self._actions = actions
        self._heuristic_fn = heuristic_fn

    def heuristic(self, state: State, goal: Union[Goal, Fluent]) -> float:
        """
        Returns the value of the non-discounted heuristic function used as part
        of the Astar planning algorithm for a given state and the current task
        in a real-time task stream setting.
        """
        if isinstance(self._heuristic_fn, (int, float)):
            return float(self._heuristic_fn)
        return self._heuristic_fn(
            state,
            goal if isinstance(goal, Goal) else LiteralGoal(goal),
            self._actions
        )

    def get_trace_from_last_mcts_tree(self) -> str:
        """
        A placeholder function since astar_search has no tree to trace;
        nothing meaningful to report.
        """
        return ""

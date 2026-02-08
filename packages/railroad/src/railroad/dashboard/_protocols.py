from typing import Union, runtime_checkable, Protocol

from railroad._bindings import (
    Action,
    Fluent,
    Goal,
    State,
)


@runtime_checkable
class DashboardEnvironment(Protocol):
    @property
    def state(self) -> State: ...
    def get_actions(self) -> list[Action]: ...


class DashboardPlanner(Protocol):
    def heuristic(self, state: State, goal: Union[Goal, Fluent]) -> float: ...
    def get_trace_from_last_mcts_tree(self) -> str: ...

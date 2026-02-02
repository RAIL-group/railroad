"""Active skill protocol and base implementations."""

from typing import TYPE_CHECKING, List, Protocol, Tuple, runtime_checkable

from railroad._bindings import GroundedEffect

if TYPE_CHECKING:
    from railroad.environment.base import AbstractEnvironment


@runtime_checkable
class ActiveSkill(Protocol):
    """Protocol for tracking execution of a single action.

    ActiveSkills unify symbolic and physical execution:
    - Symbolic mode: Skills step through effects immediately when asked
    - Physical mode: Skills wrap running processes that report progress asynchronously
    """

    @property
    def robot(self) -> str:
        """Which robot is executing this skill."""
        ...

    @property
    def is_done(self) -> bool:
        """Whether the skill has completed."""
        ...

    @property
    def is_interruptible(self) -> bool:
        """Whether this skill can be interrupted mid-execution."""
        ...

    @property
    def upcoming_effects(self) -> List[Tuple[float, GroundedEffect]]:
        """Effects still to occur, with expected times."""
        ...

    @property
    def time_to_next_event(self) -> float:
        """Time until next effect. May block in physical mode."""
        ...

    def advance(self, time: float, env: "AbstractEnvironment") -> None:
        """Advance to given time, apply due effects to environment."""
        ...

    def interrupt(self, env: "AbstractEnvironment") -> None:
        """Interrupt this skill, applying partial effects to environment."""
        ...

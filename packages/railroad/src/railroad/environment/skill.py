"""Active skill protocol and base implementations."""

from typing import TYPE_CHECKING, Dict, List, Protocol, Set, Tuple, runtime_checkable

from railroad._bindings import GroundedEffect
from railroad.core import Fluent

if TYPE_CHECKING:
    from railroad._bindings import Action


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

    def advance(self, time: float, env: "Environment") -> None:
        """Advance to given time, apply due effects to environment."""
        ...

    def interrupt(self, env: "Environment") -> None:
        """Interrupt this skill, applying partial effects to environment."""
        ...


@runtime_checkable
class Environment(Protocol):
    """Protocol for environment that owns world state.

    The Environment is the single source of truth for the world state. It:
    - Holds current fluents (ground truth)
    - Holds objects_by_type (all known objects)
    - Creates ActiveSkills via factory method
    - Applies effects (handling adds, removes, and perception)
    - Resolves probabilistic effect branches
    """

    @property
    def fluents(self) -> Set[Fluent]:
        """Current ground truth fluents."""
        ...

    @property
    def objects_by_type(self) -> Dict[str, Set[str]]:
        """All known objects, organized by type."""
        ...

    def create_skill(self, action: "Action", time: float) -> ActiveSkill:
        """Create an ActiveSkill appropriate for this environment."""
        ...

    def apply_effect(self, effect: GroundedEffect) -> None:
        """Apply an effect, handling adds, removes, and perception."""
        ...

    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Resolve which branch of a probabilistic effect occurs."""
        ...

    def get_objects_at_location(self, location: str) -> Dict[str, Set[str]]:
        """Get objects at a location (ground truth for search resolution)."""
        ...

    def remove_object_from_location(self, obj: str, location: str) -> None:
        """Update ground truth when object picked."""
        ...

    def add_object_at_location(self, obj: str, location: str) -> None:
        """Update ground truth when object placed."""
        ...


class SymbolicSkill:
    """Base class for symbolic (non-physical) skill execution.

    Implements ActiveSkill protocol for symbolic mode where skills
    step through effects immediately when asked.
    """

    def __init__(
        self,
        action: "Action",
        start_time: float,
        robot: str,
        is_interruptible: bool = False,
    ) -> None:
        """Initialize a symbolic skill.

        Args:
            action: The action being executed.
            start_time: Start time of the action.
            robot: Name of the robot executing this skill.
            is_interruptible: Whether this skill can be interrupted.
        """
        self._action = action
        self._start_time = start_time
        self._robot = robot
        self._is_interruptible = is_interruptible
        self._upcoming_effects: List[Tuple[float, GroundedEffect]] = sorted(
            [(start_time + eff.time, eff) for eff in action.effects],
            key=lambda el: el[0]
        )

    @property
    def robot(self) -> str:
        return self._robot

    @property
    def is_done(self) -> bool:
        return len(self._upcoming_effects) == 0

    @property
    def is_interruptible(self) -> bool:
        return self._is_interruptible

    @property
    def upcoming_effects(self) -> List[Tuple[float, GroundedEffect]]:
        return self._upcoming_effects

    @property
    def time_to_next_event(self) -> float:
        if self._upcoming_effects:
            return self._upcoming_effects[0][0]
        return float("inf")

    def advance(self, time: float, env: Environment) -> None:
        """Advance to given time, apply due effects to environment."""
        due_effects = [
            (t, eff) for t, eff in self._upcoming_effects
            if t <= time + 1e-9
        ]
        self._upcoming_effects = self._upcoming_effects[len(due_effects):]

        for _, effect in due_effects:
            env.apply_effect(effect)

    def interrupt(self, env: Environment) -> None:
        """Interrupt this skill. Default: no-op (non-interruptible skills)."""
        pass


class SymbolicMoveSkill(SymbolicSkill):
    """Symbolic skill for move actions with interruption support.

    Move actions can be interrupted mid-execution, creating an
    intermediate location (robot_loc) at the interrupt point.
    """

    def __init__(
        self,
        action: "Action",
        start_time: float,
        robot: str,
        start: str,
        end: str,
    ) -> None:
        """Initialize a symbolic move skill.

        Args:
            action: The move action being executed.
            start_time: Start time of the move.
            robot: Name of the robot executing this skill.
            start: Starting location name.
            end: Destination location name.
        """
        super().__init__(action, start_time, robot, is_interruptible=True)
        self.start = start
        self.end = end
        self._current_time = start_time

    def advance(self, time: float, env: Environment) -> None:
        """Advance the move, tracking current time for interruption."""
        self._current_time = time
        super().advance(time, env)

    def interrupt(self, env: Environment) -> None:
        """Interrupt move and create intermediate location.

        Creates a robot_loc intermediate location and rewrites
        destination fluents to point there instead of the original end.
        """
        if self._current_time <= self._start_time:
            return  # Cannot interrupt before start

        if self.is_done:
            return  # Already complete

        robot = self._robot
        old_target = self.end
        new_target = f"{robot}_loc"

        # Collect fluents from remaining effects, rewriting destination
        new_fluents: Set[Fluent] = set()
        for _, eff in self._upcoming_effects:
            if eff.is_probabilistic:
                raise ValueError("Probabilistic effects cannot be interrupted.")
            for fluent in eff.resulting_fluents:
                # Remove conflicting negation if present
                if (~fluent) in new_fluents:
                    new_fluents.remove(~fluent)
                # Rewrite fluent args, replacing old target with new
                new_args = [
                    arg if arg != old_target else new_target
                    for arg in fluent.args
                ]
                new_fluents.add(
                    Fluent(" ".join([fluent.name] + new_args), negated=fluent.negated)
                )

        # Apply the rewritten fluents directly to environment
        for fluent in new_fluents:
            if fluent.negated:
                env.fluents.discard(~fluent)
            else:
                env.fluents.add(fluent)

        # Clear remaining effects
        self._upcoming_effects = []


class SymbolicSearchSkill(SymbolicSkill):
    """Symbolic skill for search actions with probabilistic resolution.

    Search actions have probabilistic outcomes resolved by the environment
    based on ground truth object locations. This skill is non-interruptible.
    """

    def __init__(
        self,
        action: "Action",
        start_time: float,
        robot: str,
        location: str,
        target_object: str,
    ) -> None:
        """Initialize a symbolic search skill.

        Args:
            action: The search action being executed.
            start_time: Start time of the search.
            robot: Name of the robot executing this skill.
            location: Location being searched.
            target_object: Object being searched for.
        """
        super().__init__(action, start_time, robot, is_interruptible=False)
        self.location = location
        self.target_object = target_object

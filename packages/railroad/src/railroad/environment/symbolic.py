"""Symbolic environment implementation."""

import inspect
import random
from typing import Any, Callable, Collection, Dict, List, Set, Tuple, Type

import numpy as np

from railroad._bindings import (
    Action,
    Fluent,
    GroundedEffect,
    State,
    apply_effect_deterministic,
)
from railroad.core import Operator

from .environment import ActiveSkill, Environment


class LocationRegistry:
    """Shared location coordinate registry for interruptible moves.

    Connects operator time functions with the environment so that
    intermediate locations (created during move interrupts) are
    automatically available for subsequent action instantiation.

    Example:
        # Create registry with known locations
        registry = LocationRegistry({
            "kitchen": np.array([0, 0]),
            "bedroom": np.array([10, 0]),
        })

        # Use registry's move_time helper for operator construction
        move_op = operators.construct_move_operator_blocking(
            registry.move_time_fn(velocity=1.0)
        )

        # Pass same registry to environment
        env = SymbolicEnvironment(
            ...,
            location_registry=registry,
            skill_overrides={"move": InterruptibleNavigationMoveSkill},
        )
    """

    def __init__(self, locations: Dict[str, Any]) -> None:
        """Initialize registry with location coordinates.

        Args:
            locations: Mapping from location names to coordinates.
                Coordinates can be any array-like type (numpy array, tuple, etc.)
                that supports arithmetic operations.
        """
        self._coords: Dict[str, Any] = dict(locations)

    def get(self, key: str) -> Any | None:
        """Get coordinates for a location, or None if not found."""
        return self._coords.get(key)

    def register(self, key: str, coords: Any) -> None:
        """Register coordinates for a new location."""
        self._coords[key] = coords

    def __contains__(self, key: str) -> bool:
        """Check if a location is registered."""
        return key in self._coords

    def move_time_fn(self, velocity: float) -> Callable[[str, str, str], float]:
        """Create a move_time function bound to this registry.

        The returned function computes move duration based on Euclidean
        distance between locations. Unknown locations return infinity,
        causing those actions to be filtered out.

        Args:
            velocity: Robot movement speed (distance per time unit).

        Returns:
            A function suitable for construct_move_operator_blocking.
        """
        def move_time(robot: str, loc_from: str, loc_to: str) -> float:
            start = self._coords.get(loc_from)
            end = self._coords.get(loc_to)
            if start is None or end is None:
                return float("inf")  # Will be filtered by _is_valid_action
            # Use numpy-compatible distance calculation
            diff = end - start
            distance = float((diff @ diff) ** 0.5)
            return distance / velocity
        return move_time


class SymbolicSkill(ActiveSkill):
    """Symbolic skill execution driven entirely by action effects.

    Implements ActiveSkill protocol for symbolic mode where skills
    step through effects immediately when asked. Not interruptible
    by default - use InterruptibleNavigationMoveSkill for moves that
    can be interrupted when another robot becomes free.
    """

    def __init__(
        self,
        action: Action,
        start_time: float,
    ) -> None:
        """Initialize a symbolic skill from an action.

        Args:
            action: The action being executed (contains all effect info).
            start_time: Start time of the action.
        """
        self._action = action
        self._start_time = start_time
        self._current_time = start_time
        self._upcoming_effects: List[Tuple[float, GroundedEffect]] = sorted(
            [(start_time + eff.time, eff) for eff in action.effects],
            key=lambda el: el[0]
        )
        self._is_interruptible = False

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

    def advance(self, time: float, env: "Environment") -> None:
        """Advance to given time, apply due effects to environment."""
        self._current_time = time
        due_effects = [
            (t, eff) for t, eff in self._upcoming_effects
            if t <= time + 1e-9
        ]
        self._upcoming_effects = self._upcoming_effects[len(due_effects):]

        for effect_time, effect in due_effects:
            # Apply effect and get any delayed effects that need scheduling
            delayed = env.apply_effect(effect)
            for relative_time, delayed_effect in delayed:
                # Schedule delayed effect relative to when parent effect fired
                abs_time = effect_time + relative_time
                self._upcoming_effects.append((abs_time, delayed_effect))

        # Re-sort if we added delayed effects
        if due_effects:
            self._upcoming_effects.sort(key=lambda el: el[0])

    def interrupt(self, env: "Environment") -> None:
        """Interrupt this skill. No-op for base SymbolicSkill."""
        pass


class SymbolicEnvironment(Environment):
    """Environment for symbolic (non-physical) execution.

    Suitable for planning simulations and unit tests where:
    - Fluents are tracked in-memory
    - Skills execute by stepping through effects
    - Probabilistic effects are sampled with a seedable RNG

    This class is deliberately domain-agnostic: no fluent or action name has
    special meaning here. For railroad's object-search domains (ground-truth
    search resolution, revelation of found objects, robot intermediate
    locations), use :class:`ObjectSearchEnvironment`.
    """

    def __init__(
        self,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator] | None = None,
        skill_overrides: Dict[str, Type[ActiveSkill]] | None = None,
        location_registry: LocationRegistry | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize a symbolic environment.

        Args:
            state: Initial state (fluents, time, and optional upcoming effects).
            objects_by_type: Objects organized by type.
            operators: Optional explicit operators for action instantiation.
                If None, Environment.define_operators() is used.
            skill_overrides: Optional mapping from action type prefix to skill
                class. E.g., {"move": InterruptibleNavigationMoveSkill}
            location_registry: Optional LocationRegistry for interruptible moves.
                Required by movement-path-aware skills such as
                InterruptibleNavigationMoveSkill.
            seed: Seed for the RNG used to sample probabilistic effect
                branches. None seeds from entropy.
        """
        # Initialize subclass state before super().__init__
        # because _create_initial_effects_skill may need these
        self._fluents: Set[Fluent] = set(state.fluents)
        self._objects_by_type = {k: set(v) for k, v in objects_by_type.items()}
        self._skill_overrides: Dict[str, Callable[..., ActiveSkill]] = dict(
            skill_overrides or {}
        )
        self._location_registry = location_registry
        self._rng = random.Random(seed)

        super().__init__(state=state, operators=operators)

    @property
    def fluents(self) -> Set[Fluent]:
        return self._fluents

    @property
    def objects_by_type(self) -> Dict[str, Set[str]]:
        return self._objects_by_type

    @property
    def location_registry(self) -> LocationRegistry | None:
        """Location coordinate registry for interruptible moves."""
        return self._location_registry

    def compute_path_between_points(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> np.ndarray:
        """Compute a straight-line 2-point path between world-coordinate points."""
        return np.array(
            [[start[0], end[0]], [start[1], end[1]]],
            dtype=float,
        )

    def compute_move_path(
        self,
        loc_from: str,
        loc_to: str,
        robot: str | None = None,
    ) -> np.ndarray:
        """Compute a straight-line 2-point path from symbolic coordinates."""
        del robot
        if self._location_registry is None:
            raise TypeError(
                "SymbolicEnvironment.compute_move_path requires location_registry."
            )
        start = self._location_registry.get(loc_from)
        end = self._location_registry.get(loc_to)
        if start is None or end is None:
            return np.empty((2, 0), dtype=float)

        start_xy = np.asarray(start, dtype=float).reshape(-1)
        end_xy = np.asarray(end, dtype=float).reshape(-1)
        if start_xy.size < 2 or end_xy.size < 2:
            raise TypeError(
                "LocationRegistry coordinates must provide at least x/y components."
            )

        return self.compute_path_between_points(
            (float(start_xy[0]), float(start_xy[1])),
            (float(end_xy[0]), float(end_xy[1])),
        )

    def _create_initial_effects_skill(
        self,
        start_time: float,
        upcoming_effects: List[Tuple[float, GroundedEffect]],
    ) -> SymbolicSkill:
        """Create a SymbolicSkill from initial upcoming effects.

        Only the top-level time is rebased onto the skill's start time; branch
        sub-effect times are already relative to their parent, so they carry
        over untouched.
        """
        relative_effects = [
            GroundedEffect(
                abs_time - start_time,
                effect.resulting_fluents,
                prob_effects=[
                    (branch.prob, branch.effects)
                    for branch in effect.prob_effects
                ],
                cond_effects=[
                    (branch.conditions, branch.effects)
                    for branch in effect.cond_effects
                ],
            )
            for abs_time, effect in upcoming_effects
        ]
        action = Action(set(), relative_effects, name="_initial_effects")
        return SymbolicSkill(action=action, start_time=start_time)

    def create_skill(self, action: Action, time: float) -> ActiveSkill:
        """Create an ActiveSkill from the action.

        Uses SymbolicSkill by default. Use skill_overrides to customize
        skill classes for specific action types (e.g., interruptible moves).
        """
        parts = action.name.split()
        action_type = parts[0] if parts else ""

        if action_type in self._skill_overrides:
            skill_class = self._skill_overrides[action_type]
            if "env" in inspect.signature(skill_class.__init__).parameters:
                return skill_class(action=action, start_time=time, env=self)
            return skill_class(action=action, start_time=time)

        return SymbolicSkill(action=action, start_time=time)

    def apply_effect(
        self, effect: GroundedEffect
    ) -> List[Tuple[float, GroundedEffect]]:
        """Apply an effect, handling adds, removes, and probabilistic branches.

        Returns:
            List of (relative_time, effect) tuples for effects that should be
            scheduled later (nested effects with time > 0).
        """
        delayed_effects: List[Tuple[float, GroundedEffect]] = []

        # The deterministic part (conditional-branch evaluation against the
        # pre-effect state, then deletes-before-adds) is delegated to the C++
        # core, so execution semantics cannot drift from the planner's.
        self._fluents, triggered = apply_effect_deterministic(
            self._fluents, effect
        )

        for sub_effect in triggered:
            if sub_effect.time > 1e-9:
                # Schedule for later - time is relative to when parent fired
                delayed_effects.append((sub_effect.time, sub_effect))
            else:
                delayed_effects.extend(self.apply_effect(sub_effect))

        # Handle probabilistic branches if present
        if effect.is_probabilistic:
            nested_effects, _ = self.resolve_probabilistic_effect(
                effect, self._fluents
            )
            for nested in nested_effects:
                if nested.time > 1e-9:
                    # Schedule for later - time is relative to when parent fired
                    delayed_effects.append((nested.time, nested))
                else:
                    # Apply immediately and collect any further delayed effects
                    delayed_effects.extend(self.apply_effect(nested))

        return delayed_effects

    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Sample one branch of a probabilistic effect with the seeded RNG."""
        if not effect.is_probabilistic:
            return [effect], current_fluents

        branches = effect.prob_effects
        if not branches:
            return [], current_fluents

        probs = [p for p, _ in branches]
        _, effects = self._rng.choices(branches, weights=probs, k=1)[0]
        return list(effects), current_fluents

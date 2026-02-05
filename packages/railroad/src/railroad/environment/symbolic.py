"""Symbolic environment implementation."""

import random
import warnings
from typing import Any, Callable, Dict, List, Set, Tuple, Type

from railroad._bindings import Action, Fluent, GroundedEffect, State
from railroad.core import Operator

from .environment import Environment
from .skill import ActiveSkill


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
            skill_overrides={"move": InterruptableMoveSymbolicSkill},
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
    by default - use InterruptableMoveSymbolicSkill for moves that
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


class InterruptableMoveSymbolicSkill(SymbolicSkill):
    """Move skill that can be interrupted when another robot becomes free.

    When interrupted mid-execution, rewrites destination fluents to an
    intermediate location (robot_loc), enabling the planner to account
    for partial progress.

    If the environment has a LocationRegistry, the intermediate location's
    coordinates are computed via linear interpolation and registered
    automatically. Without a registry, the interrupt still works but
    subsequent move actions from the intermediate location may not have
    correct time estimates.
    """

    def __init__(
        self,
        action: Action,
        start_time: float,
    ) -> None:
        """Initialize an interruptable move skill.

        Args:
            action: The move action being executed.
            start_time: Start time of the move.
        """
        super().__init__(action, start_time)
        self._is_interruptible = True

        # Extract robot, origin, and destination from action name: "move robot from to"
        parts = action.name.split()
        self._robot = parts[1] if len(parts) >= 2 else None
        self._move_origin = parts[2] if len(parts) >= 3 else None
        self._move_destination = parts[3] if len(parts) >= 4 else None

        # Calculate end time from the last effect (when move completes)
        if self._upcoming_effects:
            self._end_time = max(t for t, _ in self._upcoming_effects)
        else:
            self._end_time = start_time

    def interrupt(self, env: "Environment") -> None:
        """Interrupt move and create intermediate location.

        Creates a robot_loc intermediate location and rewrites
        destination fluents to point there instead of the original end.
        If the environment has a location_registry, registers the
        intermediate location's coordinates.
        """
        if self._current_time <= self._start_time:
            return  # Cannot interrupt before start

        if self.is_done:
            return  # Already complete

        if self._move_destination is None or self._robot is None:
            return  # Not a valid move action

        # For move actions: create intermediate location and rewrite effects
        old_target = self._move_destination
        new_target = f"{self._robot}_loc"

        # Register intermediate location coordinates if registry available
        registry: LocationRegistry | None = getattr(env, "location_registry", None)
        if registry is not None and self._move_origin is not None:
            origin_coords = registry.get(self._move_origin)
            dest_coords = registry.get(self._move_destination)
            if origin_coords is not None and dest_coords is not None:
                # Calculate progress fraction
                total_time = self._end_time - self._start_time
                if total_time > 0:
                    progress = (self._current_time - self._start_time) / total_time
                    progress = max(0.0, min(1.0, progress))
                else:
                    progress = 1.0
                # Linear interpolation
                intermediate_coords = origin_coords + progress * (dest_coords - origin_coords)
                registry.register(new_target, intermediate_coords)

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


class SymbolicEnvironment(Environment):
    """Environment for symbolic (non-physical) execution.

    Suitable for planning simulations and unit tests where:
    - Fluents are tracked in-memory
    - Skills execute by stepping through effects
    - Probabilistic effects resolve via ground truth object locations
    """

    def __init__(
        self,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator],
        true_object_locations: Dict[str, Set[str]] | None = None,
        skill_overrides: Dict[str, Type[ActiveSkill]] | None = None,
        location_registry: LocationRegistry | None = None,
    ) -> None:
        """Initialize a symbolic environment.

        Args:
            state: Initial state (fluents, time, and optional upcoming effects).
            objects_by_type: Objects organized by type.
            operators: List of operators for action instantiation.
            true_object_locations: Ground truth object locations for search
                resolution. If None, search always fails.
            skill_overrides: Optional mapping from action type prefix to skill
                class. E.g., {"move": InterruptableMoveSymbolicSkill}
            location_registry: Optional LocationRegistry for interruptible moves.
                Required for InterruptableMoveSymbolicSkill to properly compute
                intermediate location coordinates when moves are interrupted.
        """
        # Initialize subclass state before super().__init__
        # because _create_initial_effects_skill may need these
        self._fluents: Set[Fluent] = set(state.fluents)
        self._objects_by_type = {k: set(v) for k, v in objects_by_type.items()}
        self._objects_at_locations = (
            {k: set(v) for k, v in true_object_locations.items()}
            if true_object_locations else {}
        )
        self._skill_overrides = skill_overrides or {}
        self._location_registry = location_registry

        # Warn if using InterruptableMoveSymbolicSkill without a registry
        if (
            location_registry is None
            and skill_overrides
            and any(
                issubclass(cls, InterruptableMoveSymbolicSkill)
                for cls in skill_overrides.values()
                if isinstance(cls, type)
            )
        ):
            warnings.warn(
                "Using InterruptableMoveSymbolicSkill without a LocationRegistry. "
                "Interrupted moves will create intermediate locations without "
                "coordinate information, which may cause move time calculations "
                "to fail. Consider passing a LocationRegistry to the environment.",
                UserWarning,
                stacklevel=2,
            )

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

    def _create_initial_effects_skill(
        self,
        start_time: float,
        upcoming_effects: List[Tuple[float, GroundedEffect]],
    ) -> SymbolicSkill:
        """Create a SymbolicSkill from initial upcoming effects."""
        relative_effects = [
            GroundedEffect(abs_time - start_time, effect.resulting_fluents)
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

        # Apply deterministic resulting_fluents
        for fluent in effect.resulting_fluents:
            if fluent.negated:
                self._fluents.discard(~fluent)
            else:
                self._fluents.add(fluent)

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

        # Handle revelation (objects discovered when locations searched)
        self._handle_revelation()

        return delayed_effects

    def _handle_revelation(self) -> None:
        """Reveal objects when locations are searched."""
        for fluent in list(self._fluents):
            if fluent.name == "searched":
                location = fluent.args[0]
                revealed_fluent = Fluent("revealed", location)

                if revealed_fluent not in self._fluents:
                    self._fluents.add(revealed_fluent)

                    # Reveal objects at this location
                    for obj in self._objects_at_locations.get(location, set()):
                        self._fluents.add(Fluent("found", obj))
                        self._fluents.add(Fluent("at", obj, location))
                        self._objects_by_type.setdefault("object", set()).add(obj)

    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Resolve probabilistic effect based on ground truth object locations.

        For search actions, checks if target object is actually at location.
        Otherwise, samples from the probability distribution.
        """
        if not effect.is_probabilistic:
            return [effect], current_fluents

        branches = effect.prob_effects
        if not branches:
            return [], current_fluents

        # Find success branch (contains positive "found" fluent)
        success_branch = None
        target_object = None
        location = None

        for branch in branches:
            _, branch_effects = branch
            for eff in branch_effects:
                for fluent in eff.resulting_fluents:
                    if fluent.name == "found" and not fluent.negated:
                        success_branch = branch
                        target_object = fluent.args[0]
                        location = self._find_search_location(eff, target_object)

        # If we can resolve from ground truth, do so deterministically
        if success_branch and target_object and location:
            if self._is_object_at_location(target_object, location):
                _, effects = success_branch
                return list(effects), current_fluents
            # Object not at location - return failure branch deterministically
            other_branches = [b for b in branches if b is not success_branch]
            if other_branches:
                # No need to sample - ground truth determines the outcome
                _, effects = other_branches[0]
                return list(effects), current_fluents

        # Can't determine from ground truth - sample from distribution
        probs = [p for p, _ in branches]
        _, effects = random.choices(branches, weights=probs, k=1)[0]
        return list(effects), current_fluents

    def _find_search_location(
        self, effect: GroundedEffect, target_object: str
    ) -> str | None:
        """Find the location from 'at object location' fluent in a branch."""
        for fluent in effect.resulting_fluents:
            if fluent.name == "at" and len(fluent.args) >= 2:
                if fluent.args[0] == target_object:
                    return fluent.args[1]
        return None

    def _is_object_at_location(self, obj: str, location: str) -> bool:
        """Check if object is at location using fluents + ground truth.

        Priority:
        1. If object is being held → not at any location
        2. If fluent says object is at this location → yes
        3. If fluent says object is at a different location → no
        4. Fall back to ground truth (for undiscovered objects)
        """
        # Check if held by any robot
        for f in self._fluents:
            if f.name == "holding" and len(f.args) >= 2 and f.args[1] == obj:
                return False

        # Check fluents for known location
        if Fluent("at", obj, location) in self._fluents:
            return True

        for f in self._fluents:
            if f.name == "at" and len(f.args) >= 2 and f.args[0] == obj:
                return False  # Object is at a different location

        # Fall back to ground truth
        return obj in self._objects_at_locations.get(location, set())



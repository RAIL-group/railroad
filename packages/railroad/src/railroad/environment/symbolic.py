"""Symbolic environment implementation."""

import random
from typing import Dict, List, Set, Tuple, Type

from railroad._bindings import Action, Fluent, GroundedEffect, State
from railroad.core import Operator

from .environment import Environment
from .skill import ActiveSkill, InterruptableMoveSymbolicSkill, SymbolicSkill


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

        super().__init__(state=state, operators=operators)

    @property
    def fluents(self) -> Set[Fluent]:
        return self._fluents

    @property
    def objects_by_type(self) -> Dict[str, Set[str]]:
        return self._objects_by_type

    def create_skill(self, action: Action, time: float) -> ActiveSkill:
        """Create an ActiveSkill from the action.

        Routes to skill class based on:
        1. skill_overrides (if configured)
        2. Default mapping (move → InterruptableMoveSymbolicSkill, else SymbolicSkill)
        """
        parts = action.name.split()
        action_type = parts[0] if parts else ""

        # Check for override first
        if action_type in self._skill_overrides:
            skill_class = self._skill_overrides[action_type]
            return skill_class(action=action, start_time=time)

        # Default routing
        if action_type == "move":
            return InterruptableMoveSymbolicSkill(action=action, start_time=time)

        return SymbolicSkill(action=action, start_time=time)

    def apply_effect(self, effect: GroundedEffect) -> None:
        """Apply an effect, handling adds, removes, and probabilistic branches."""
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
                self.apply_effect(nested)

        # Handle revelation (objects discovered when locations searched)
        self._handle_revelation()

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
            # Object not at location - sample from non-success branches
            other_branches = [b for b in branches if b is not success_branch]
            if other_branches:
                probs = [p for p, _ in other_branches]
                _, effects = random.choices(other_branches, weights=probs, k=1)[0]
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


class SimpleSymbolicEnvironment(SymbolicEnvironment):
    """Backward compatibility alias for SymbolicEnvironment.

    DEPRECATED: Use SymbolicEnvironment directly with new API.
    """

    def __init__(
        self,
        initial_state: State,
        objects_by_type: Dict[str, Set[str]],
        objects_at_locations: Dict[str, Set[str]],
        skill_overrides: Dict[str, Type[ActiveSkill]] | None = None,
    ) -> None:
        super().__init__(
            state=initial_state,
            objects_by_type=objects_by_type,
            operators=[],  # Old API didn't take operators
            true_object_locations=objects_at_locations,
            skill_overrides=skill_overrides,
        )

    @property
    def initial_state(self) -> State:
        """The initial state used to create this environment."""
        return State(0.0, self._fluents, [])

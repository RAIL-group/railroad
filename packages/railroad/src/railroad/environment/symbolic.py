"""Simple symbolic environment implementation."""

from typing import Dict, List, Set, Tuple

from railroad._bindings import Action, Fluent, GroundedEffect

from .skill import ActiveSkill, Environment, SymbolicMoveSkill, SymbolicSkill


class SimpleSymbolicEnvironment:
    """Simple environment for symbolic (non-physical) execution.

    Implements the Environment protocol for symbolic planning where:
    - Fluents are tracked in-memory
    - Skills execute immediately without physical delays
    - Probabilistic effects are resolved via ground truth object locations

    This is suitable for unit tests and symbolic simulations.
    """

    def __init__(
        self,
        fluents: Set[Fluent],
        objects_by_type: Dict[str, Set[str]],
        objects_at_locations: Dict[str, Set[str]],
    ) -> None:
        """Initialize the symbolic environment.

        Args:
            fluents: Initial set of fluents (ground truth).
            objects_by_type: Objects organized by type.
            objects_at_locations: Ground truth object locations for search resolution.
        """
        self._fluents = set(fluents)
        self._objects_by_type = {k: set(v) for k, v in objects_by_type.items()}
        self._objects_at_locations = {k: set(v) for k, v in objects_at_locations.items()}

    @property
    def fluents(self) -> Set[Fluent]:
        """Current ground truth fluents."""
        return self._fluents

    @property
    def objects_by_type(self) -> Dict[str, Set[str]]:
        """All known objects, organized by type."""
        return self._objects_by_type

    def create_skill(self, action: Action, time: float) -> ActiveSkill:
        """Create an ActiveSkill appropriate for this action.

        Routes to appropriate skill class based on action name.
        """
        parts = action.name.split()
        action_type = parts[0]
        robot = parts[1]

        if action_type == "move":
            # move robot from to
            start, end = parts[2], parts[3]
            return SymbolicMoveSkill(
                action=action,
                start_time=time,
                robot=robot,
                start=start,
                end=end,
            )
        else:
            # Default: non-interruptible symbolic skill
            return SymbolicSkill(
                action=action,
                start_time=time,
                robot=robot,
                is_interruptible=False,
            )

    def apply_effect(self, effect: GroundedEffect) -> None:
        """Apply an effect, handling adds and removes."""
        if effect.is_probabilistic:
            # Resolve probabilistic effect first
            nested_effects, _ = self.resolve_probabilistic_effect(effect, self._fluents)
            for nested in nested_effects:
                self.apply_effect(nested)
            return

        for fluent in effect.resulting_fluents:
            if fluent.negated:
                # Remove the positive version
                self._fluents.discard(~fluent)
            else:
                self._fluents.add(fluent)

        # Handle perception/revelation
        self._handle_revelation()

    def _handle_revelation(self) -> None:
        """Handle revelation when locations are searched."""
        # Find newly searched locations
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
                        # Add to objects_by_type
                        self._objects_by_type.setdefault("object", set()).add(obj)

    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Resolve which branch of a probabilistic effect occurs.

        For search actions, checks ground truth to determine success/failure.
        """
        if not effect.is_probabilistic:
            return [effect], current_fluents

        branches = effect.prob_effects
        if not branches:
            return [], current_fluents

        # Default: return first branch (success case)
        # More sophisticated resolution will be added in Task 7
        _, first_branch_effects = branches[0]
        return list(first_branch_effects), current_fluents

    def get_objects_at_location(self, location: str) -> Dict[str, Set[str]]:
        """Get objects at a location (ground truth)."""
        objects = self._objects_at_locations.get(location, set())
        return {"object": set(objects)}

    def remove_object_from_location(self, obj: str, location: str) -> None:
        """Update ground truth when object picked."""
        if location in self._objects_at_locations:
            self._objects_at_locations[location].discard(obj)

    def add_object_at_location(self, obj: str, location: str) -> None:
        """Update ground truth when object placed."""
        if location not in self._objects_at_locations:
            self._objects_at_locations[location] = set()
        self._objects_at_locations[location].add(obj)

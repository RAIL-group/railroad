"""Object-search environment: railroad's robot/location domain conventions."""

from typing import Collection, Dict, List, Set, Tuple, Type

from railroad._bindings import Action, Fluent, GroundedEffect, State
from railroad.core import Operator

from .environment import ActiveSkill
from .symbolic import LocationRegistry, SymbolicEnvironment


class ObjectSearchEnvironment(SymbolicEnvironment):
    """SymbolicEnvironment specialized for railroad's object-search domains.

    Layers the domain conventions the generic base deliberately avoids:

    - **Ground truth**: probabilistic search effects (a branch containing a
      positive ``found X`` fluent) resolve deterministically against
      ``true_object_locations`` instead of sampling.
    - **Revelation**: after each effect, ``searched L`` marks ``revealed L``
      and discovers the objects at L (adding ``found O`` and ``at O L``).
    - **Robot intermediate locations**: ``{robot}_loc`` markers created by
      interrupted moves are added to the grounding universe, with the
      matching hygiene filters (no moves to ``*_loc`` or self-loops, no
      zero-duration moves, no place/search at ``*_loc``).

    These behaviors key on fluent names (``searched``, ``found``, ``at``,
    ``holding``) and action-name prefixes (``move``, ``place``, ``search``),
    which is why they live here rather than on SymbolicEnvironment.
    """

    def __init__(
        self,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator] | None = None,
        true_object_locations: Dict[str, Set[str]] | None = None,
        skill_overrides: Dict[str, Type[ActiveSkill]] | None = None,
        location_registry: LocationRegistry | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize an object-search environment.

        Args:
            state: Initial state (fluents, time, and optional upcoming effects).
            objects_by_type: Objects organized by type.
            operators: Optional explicit operators for action instantiation.
                If None, Environment.define_operators() is used.
            true_object_locations: Ground truth object locations for search
                resolution. If None, search always fails.
            skill_overrides: Optional mapping from action type prefix to skill
                class. E.g., {"move": InterruptibleNavigationMoveSkill}
            location_registry: Optional LocationRegistry for interruptible moves.
            seed: Seed for the sampling RNG used when ground truth cannot
                determine a probabilistic outcome.
        """
        self._objects_at_locations = (
            {k: set(v) for k, v in true_object_locations.items()}
            if true_object_locations else {}
        )
        super().__init__(
            state=state,
            objects_by_type=objects_by_type,
            operators=operators,
            skill_overrides=skill_overrides,
            location_registry=location_registry,
            seed=seed,
        )

    @property
    def objects_at_locations(self) -> Dict[str, Set[str]]:
        """Ground-truth object locations, keyed by location name."""
        return self._objects_at_locations

    def runtime_mutated_predicates(self) -> Set[str]:
        """Revelation mutates these outside operator effects (must stay
        dynamic for grounding's static inference)."""
        return {"revealed", "found", "at"}

    def _grounding_objects(self) -> Dict[str, Collection[str]]:
        """Add robot intermediate locations (robot_loc) to the location set."""
        objects = super()._grounding_objects()
        robot_locs = {
            f"{rob}_loc"
            for rob in objects.get("robot", set())
            if Fluent("at", rob, f"{rob}_loc") in self.fluents
        }
        objects["location"] = set(objects.get("location", set())) | robot_locs
        return objects

    def _is_valid_action(self, action: Action) -> bool:
        """Filter actions that violate robot/location domain conventions."""
        if not super()._is_valid_action(action):
            return False

        parts = action.name.split()

        if parts[0].startswith("move"):  # "move" and variants e.g. "move-to-goal"
            if len(parts) > 3 and parts[2] == parts[3]:
                return False
            if action.effects and all(eff.time <= 1e-9 for eff in action.effects):
                return False
            if len(parts) > 3 and parts[3].endswith("_loc"):
                return False

        if parts[0] == "place" and len(parts) > 2 and parts[2].endswith("_loc"):
            return False
        if parts[0] == "search" and len(parts) > 2 and parts[2].endswith("_loc"):
            return False

        return True

    def apply_effect(
        self, effect: GroundedEffect
    ) -> List[Tuple[float, GroundedEffect]]:
        delayed_effects = super().apply_effect(effect)
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
        return super().resolve_probabilistic_effect(effect, current_fluents)

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

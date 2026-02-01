"""Environment class for resolving probabilistic outcomes.

This module provides a minimal Environment class for the execution module.
It holds ground truth and resolves probabilistic effect branches.
"""

import random
from typing import Dict, List, Set, Tuple

from railroad._bindings import Fluent, GroundedEffect


class Environment:
    """Resolves probabilistic outcomes for plan execution.

    A minimal environment that holds ground truth object locations and
    resolves probabilistic effects. This is simpler than the full
    railroad.environment.AbstractEnvironment.

    Responsibilities:
    1. Hold initial ground truth (objects at unsearched locations)
    2. Resolve which prob_effects branch happens:
       - For "found X" effects: check if X is at the location
       - For other effects: sample according to probabilities
    """

    def __init__(self, objects_at_locations: Dict[str, Set[str]]) -> None:
        """Initialize the environment.

        Args:
            objects_at_locations: Initial ground truth mapping location names
                to sets of object names present at that location.
        """
        self._objects_at_locations: Dict[str, Set[str]] = {
            loc: set(objs) for loc, objs in objects_at_locations.items()
        }

    def get_objects_at_location(self, location: str) -> Set[str]:
        """Get objects at a location (ground truth).

        Args:
            location: The location to query.

        Returns:
            Set of object names at that location.
        """
        return self._objects_at_locations.get(location, set())

    def remove_object_from_location(self, obj: str, location: str) -> None:
        """Remove an object from a location (when picked).

        Args:
            obj: Name of the object to remove.
            location: Name of the location.
        """
        if location in self._objects_at_locations:
            self._objects_at_locations[location].discard(obj)

    def add_object_at_location(self, obj: str, location: str) -> None:
        """Add an object to a location (when placed).

        Args:
            obj: Name of the object to add.
            location: Name of the location.
        """
        if location not in self._objects_at_locations:
            self._objects_at_locations[location] = set()
        self._objects_at_locations[location].add(obj)

    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Resolve which probabilistic branch happens.

        For search-related effects (those with "found X" fluents), resolution
        is based on ground truth - if the object is at the searched location,
        the success branch is chosen. For other probabilistic effects,
        branches are sampled according to their probabilities.

        Args:
            effect: A probabilistic GroundedEffect to resolve.
            current_fluents: Current state fluents (used to determine search location).

        Returns:
            Tuple of (nested_effects, immediate_fluents):
            - nested_effects: List of GroundedEffects from the chosen branch
            - immediate_fluents: Set of immediate fluents from the chosen branch
        """
        if not effect.is_probabilistic:
            return [], effect.resulting_fluents

        # Check if this is a search-related effect (contains "found X" in any branch)
        search_info = self._extract_search_info(effect, current_fluents)

        if search_info is not None:
            obj, location = search_info
            objects_at_loc = self.get_objects_at_location(location)
            found = obj in objects_at_loc

            # Find the appropriate branch
            for branch in effect.prob_effects:
                branch_has_found = self._branch_has_found_fluent(branch.effects, obj)
                if found and branch_has_found:
                    return list(branch.effects), set()
                elif not found and not branch_has_found:
                    return list(branch.effects), set()

            # Fallback: return first branch if no match found
            first_branch = effect.prob_effects[0]
            return list(first_branch.effects), set()

        # Not search-related: sample by probability
        return self._sample_branch(effect.prob_effects)

    def _extract_search_info(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[str, str] | None:
        """Extract search object and location from a probabilistic effect.

        Looks for "found X" fluents in any branch to identify search effects,
        then finds the searched location from current fluents.

        Args:
            effect: The probabilistic effect.
            current_fluents: Current state fluents.

        Returns:
            Tuple of (object_name, location) if this is a search effect,
            None otherwise.
        """
        # Find any "found X" fluent in the branches
        found_obj = None
        for branch in effect.prob_effects:
            for nested_eff in branch.effects:
                for fluent in nested_eff.resulting_fluents:
                    if fluent.name == "found" and not fluent.negated:
                        found_obj = fluent.args[0]
                        break
                if found_obj:
                    break
            if found_obj:
                break

        if found_obj is None:
            return None

        # Find the searched location from current fluents
        # Look for "searched <loc> <obj>" fluent
        for fluent in current_fluents:
            if fluent.name == "searched" and len(fluent.args) >= 2:
                loc, obj = fluent.args[0], fluent.args[1]
                if obj == found_obj:
                    return (found_obj, loc)

        # Also check for robot location via "at <robot> <loc>" fluents
        # This handles search_and_pick where searched fluent is added immediately
        for fluent in current_fluents:
            if fluent.name == "at" and len(fluent.args) >= 2:
                entity, loc = fluent.args[0], fluent.args[1]
                # Check if this is a robot (robots typically have names like "robot1")
                if entity.startswith("robot") or entity.startswith("r"):
                    return (found_obj, loc)

        return None

    def _branch_has_found_fluent(
        self,
        effects: List[GroundedEffect],
        obj: str,
    ) -> bool:
        """Check if a branch contains a 'found <obj>' fluent.

        Args:
            effects: List of effects in the branch.
            obj: Object name to look for.

        Returns:
            True if branch contains 'found <obj>', False otherwise.
        """
        for eff in effects:
            for fluent in eff.resulting_fluents:
                if fluent.name == "found" and fluent.args[0] == obj and not fluent.negated:
                    return True
            # Check recursively in nested probabilistic effects
            if eff.is_probabilistic:
                for branch in eff.prob_effects:
                    if self._branch_has_found_fluent(list(branch.effects), obj):
                        return True
        return False

    def _sample_branch(
        self,
        prob_effects: List,
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Sample a branch according to probabilities.

        Args:
            prob_effects: List of ProbBranch objects.

        Returns:
            Tuple of (nested_effects, immediate_fluents) from the sampled branch.
        """
        if not prob_effects:
            return [], set()

        probs = [branch.prob for branch in prob_effects]
        total = sum(probs)
        if total == 0:
            # Equal probability fallback
            branch = random.choice(prob_effects)
        else:
            # Normalize and sample
            r = random.random() * total
            cumulative = 0.0
            branch = prob_effects[-1]  # default to last
            for b in prob_effects:
                cumulative += b.prob
                if r <= cumulative:
                    branch = b
                    break

        return list(branch.effects), set()

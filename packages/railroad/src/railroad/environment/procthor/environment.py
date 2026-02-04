"""ProcTHOR environment for PDDL planning."""

from typing import Dict, List, Set

from railroad._bindings import State
from railroad.core import Operator
from railroad.environment import SymbolicEnvironment

from .scene import ProcTHORScene


class ProcTHOREnvironment(SymbolicEnvironment):
    """Symbolic environment backed by a ProcTHOR scene.

    Subclass of SymbolicEnvironment that provides:
    - Direct access to the ProcTHOR scene
    - Optional validation that objects/locations exist
    - Convenience methods for scene data

    Example:
        scene = ProcTHORScene(seed=4001)
        move_op = operators.construct_move_operator_blocking(scene.get_move_cost_fn())

        env = ProcTHOREnvironment(
            scene=scene,
            state=State(0.0, {F("at robot1 start_loc"), F("free robot1")}),
            objects_by_type={
                "robot": {"robot1"},
                "location": set(scene.locations.keys()),
                "object": {"teddybear_6"},
            },
            operators=[move_op, search_op, pick_op, place_op],
        )
    """

    def __init__(
        self,
        scene: ProcTHORScene,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator],
        validate: bool = True,
    ) -> None:
        """Initialize ProcTHOR environment.

        Args:
            scene: ProcTHORScene data provider
            state: Initial planning state
            objects_by_type: Objects organized by type
            operators: Planning operators
            validate: Whether to validate objects/locations exist in scene
        """
        self.scene = scene

        if validate:
            self._validate(objects_by_type)

        super().__init__(
            state=state,
            objects_by_type=objects_by_type,
            operators=operators,
            true_object_locations=scene.object_locations,
        )

    def _validate(self, objects_by_type: Dict[str, Set[str]]) -> None:
        """Validate that objects and locations exist in scene."""
        # Validate locations
        if "location" in objects_by_type:
            scene_locations = set(self.scene.locations.keys())
            for loc in objects_by_type["location"]:
                if loc not in scene_locations:
                    # Allow robot_loc intermediate locations
                    if not loc.endswith("_loc"):
                        raise ValueError(
                            f"Location '{loc}' not found in scene. "
                            f"Available: {sorted(scene_locations)[:5]}..."
                        )

        # Validate objects
        if "object" in objects_by_type:
            scene_objects = self.scene.objects
            for obj in objects_by_type["object"]:
                if obj not in scene_objects:
                    raise ValueError(
                        f"Object '{obj}' not found in scene. "
                        f"Available: {sorted(scene_objects)[:5]}..."
                    )

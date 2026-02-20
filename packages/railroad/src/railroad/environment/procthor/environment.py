"""ProcTHOR environment for PDDL planning."""

from abc import ABC, abstractmethod
from typing import Dict, List, Set

import numpy as np

from railroad._bindings import State
from railroad.core import Operator
from railroad.navigation import OccupancyGridPathingMixin
from ..symbolic import LocationRegistry, SymbolicEnvironment

from .scene import ProcTHORScene


class ProcTHOREnvironment(OccupancyGridPathingMixin, SymbolicEnvironment, ABC):
    """Symbolic environment backed by a ProcTHOR scene.

    Subclass of SymbolicEnvironment that provides:
    - Internal scene creation from a seed
    - Direct access to the ProcTHOR scene via ``self.scene``
    - Optional validation that objects/locations exist
    - Occupancy-grid pathing via mixin methods
    - A required ``define_operators`` hook for subclasses

    Example:
        class LocalProcTHOREnvironment(ProcTHOREnvironment):
            def define_operators(self) -> list[Operator]:
                from railroad import operators
                move_op = operators.construct_move_operator_blocking(
                    self.estimate_move_time
                )
                return [move_op]

        env = LocalProcTHOREnvironment(
            seed=4001,
            state=State(0.0, {F("at robot1 start_loc"), F("free robot1")}),
            objects_by_type={
                "robot": {"robot1"},
                "location": {"start_loc"},
                "object": {"teddybear_6"},
            },
        )
    """

    def __init__(
        self,
        seed: int,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator] | None = None,
        resolution: float = 0.05,
        validate: bool = True,
    ) -> None:
        """Initialize ProcTHOR environment.

        Args:
            seed: ProcTHOR scene seed
            state: Initial planning state
            objects_by_type: Objects organized by type
            operators: Optional explicit operators. If omitted, this class
                resolves operators from ``define_operators()``.
            resolution: Grid resolution in meters
            validate: Whether to validate objects/locations exist in scene
        """
        self.scene = ProcTHORScene(seed=seed, resolution=resolution)

        location_registry = LocationRegistry(
            {
                name: np.array(coords, dtype=float)
                for name, coords in self.scene.locations.items()
            }
        )

        # Bootstrap for define_operators(): make pathing helpers available
        # before SymbolicEnvironment.__init__ runs.
        self._location_registry = location_registry

        if validate:
            self._validate(objects_by_type)

        super().__init__(
            state=state,
            objects_by_type=objects_by_type,
            operators=operators,
            true_object_locations=self.scene.object_locations,
            location_registry=location_registry,
        )

    @abstractmethod
    def define_operators(self) -> List[Operator]:
        """Build operators for this environment instance."""
        ...

    @property
    def occupancy_grid(self) -> np.ndarray:
        return self.scene.grid

    @property
    def _pathing_unknown_as_obstacle(self) -> bool:
        return False

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

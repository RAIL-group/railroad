"""ProcTHOR scene data provider."""

from typing import Any, Callable, Dict, Set, Tuple

import numpy as np

from railroad._bindings import Action

from .thor_interface import ThorInterface
from . import utils as procthor_utils


class ProcTHORScene:
    """Data provider for ProcTHOR environments.

    Loads a ProcTHOR scene and extracts all information needed for planning:
    - Location names and coordinates
    - Object names and their ground truth locations
    - Move cost function
    - Path interpolation for interrupted moves

    Example:
        scene = ProcTHORScene(seed=4001)
        print(scene.locations)  # All container locations
        print(scene.objects)    # All objects in scene

        # Create move operator with scene's cost function
        move_op = operators.construct_move_operator_blocking(scene.get_move_cost_fn())
    """

    def __init__(self, seed: int, resolution: float = 0.05) -> None:
        """Initialize ProcTHOR scene.

        Args:
            seed: Random seed for scene selection (0-9999 for ProcTHOR-10k)
            resolution: Grid resolution in meters
        """
        self._thor = ThorInterface(seed=seed, resolution=resolution)

        # Build location registry
        self._locations = self._build_locations()

        # Extract all objects
        self._objects = self._extract_objects()

        # Build ground truth object locations (location -> objects)
        self._object_locations = self._build_object_locations()

    @property
    def grid(self) -> np.ndarray:
        """Occupancy grid (0=free, 1=occupied)."""
        return self._thor.occupancy_grid

    @property
    def scene_graph(self):
        """Scene graph for visualization/debugging."""
        return self._thor.scene_graph

    @property
    def locations(self) -> Dict[str, Tuple[int, int]]:
        """Location names mapped to grid coordinates."""
        return self._locations

    @property
    def objects(self) -> Set[str]:
        """All objects in the scene."""
        return self._objects

    @property
    def object_locations(self) -> Dict[str, Set[str]]:
        """Ground truth: location name -> set of object names at that location."""
        return self._object_locations

    def _build_locations(self) -> Dict[str, Tuple[int, int]]:
        """Extract locations from scene graph containers."""
        locations = {'start_loc': self._thor.robot_pose}

        for idx in self._thor.scene_graph.container_indices:
            node = self._thor.scene_graph.nodes[idx]
            name = f"{node['name']}_{idx}"
            locations[name] = tuple(node['position'])

        return locations

    def _extract_objects(self) -> Set[str]:
        """Extract all object names from scene graph."""
        objects = set()
        for idx in self._thor.scene_graph.object_indices:
            name = self._thor.scene_graph.get_node_name_by_idx(idx)
            objects.add(f"{name}_{idx}")
        return objects

    def _build_object_locations(self) -> Dict[str, Set[str]]:
        """Build mapping of location -> objects at that location."""
        result: Dict[str, Set[str]] = {}

        for container_idx in self._thor.scene_graph.container_indices:
            container_node = self._thor.scene_graph.nodes[container_idx]
            location_name = f"{container_node['name']}_{container_idx}"

            object_idxs = self._thor.scene_graph.get_adjacent_nodes_idx(
                container_idx, filter_by_type=3
            )

            for obj_idx in object_idxs:
                obj_name = f"{self._thor.scene_graph.get_node_name_by_idx(obj_idx)}_{obj_idx}"
                result.setdefault(location_name, set()).add(obj_name)

        return result

    def get_move_cost_fn(self) -> Callable[[str, str, str], float]:
        """Get move cost function for operator construction.

        Returns:
            Function (robot, loc_from, loc_to) -> cost
        """
        # Build lookup from location name to container ID
        loc_to_id: Dict[str, str] = {'start_loc': 'initial_robot_pose'}
        for container in self._thor.containers:
            idx = self._thor.scene_graph.asset_id_to_node_idx_map[container['id']]
            name = f"{procthor_utils.get_generic_name(container['id'])}_{idx}"
            loc_to_id[name] = container['id']

        def move_cost_fn(robot: str, loc_from: str, loc_to: str) -> float:
            id_from = loc_to_id.get(loc_from)
            id_to = loc_to_id.get(loc_to)

            if id_from and id_to and id_from in self._thor.known_cost:
                return self._thor.known_cost[id_from].get(id_to, float('inf'))

            # Fall back to grid-based cost
            coord_from = self._locations.get(loc_from)
            coord_to = self._locations.get(loc_to)
            if coord_from is None or coord_to is None:
                return float('inf')

            return procthor_utils.get_cost(self._thor.occupancy_grid, coord_from, coord_to)

        return move_cost_fn

    def get_intermediate_coordinates(
        self,
        action: Action,
        elapsed_time: float
    ) -> Tuple[int, int]:
        """Get coordinates along move path at elapsed time.

        Args:
            action: A move action with name "move robot loc_from loc_to"
            elapsed_time: Time elapsed since move started

        Returns:
            Grid coordinates at that time
        """
        parts = action.name.split()
        if len(parts) < 4 or parts[0] != 'move':
            raise ValueError(f"Expected move action, got: {action.name}")

        loc_from = parts[2]
        loc_to = parts[3]

        coord_from = self._locations.get(loc_from)
        coord_to = self._locations.get(loc_to)

        if coord_from is None or coord_to is None:
            raise ValueError(f"Unknown location: {loc_from} or {loc_to}")

        # Get full path
        _, path = procthor_utils.get_cost_and_path(
            self._thor.occupancy_grid,
            coord_from,
            coord_to
        )

        # Interpolate along path
        coords = procthor_utils.get_coordinates_at_time(path, elapsed_time)
        return int(coords[0]), int(coords[1])

    def get_top_down_image(self, orthographic: bool = True) -> np.ndarray:
        """Get top-down view image of the scene."""
        return self._thor.get_top_down_image(orthographic=orthographic)

"""ProcTHOR scene data provider."""

from typing import Callable, Dict, Set, Tuple, List

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

    def get_object_find_prob_fn(
        self,
        nn_model_path: str,
        objects_to_find: List[str],
        objects_with_idx: bool = True
    ) -> Callable[[str, str, str], float]:
        """
        Get learned object find probability function.
        Args:
            nn_model_path: Path to trained neural network model
            objects_to_find: List of object names to find
            objects_with_idx: Whether objects_to_find includes indices (e.g., 'teddybear_6')
        Returns:
            Function (robot, location, object) -> probability
        """
        # Get the learned model
        from . import learning
        self.obj_prob_net = learning.models.FCNN.get_net_eval_fn(nn_model_path)

        # if objects in objects_to_find have index (e.g., 'teddybear_6'), remove the index
        if objects_with_idx:
            objects_without_idx = []
            for obj in objects_to_find:
                objects_without_idx.append(obj.split('_')[0])
            objects_to_find = objects_without_idx
        object_free_scene_graph = self.scene_graph.get_object_free_graph()
        node_features_dict = learning.utils.prepare_fcnn_input(
            object_free_scene_graph, self.scene_graph.container_indices, objects_to_find)

        object_container_prop_dict = {}
        for obj in objects_to_find:
            datum = {'node_feats': node_features_dict[obj]}
            object_container_prop_dict[obj] = self.obj_prob_net(datum, self.scene_graph.container_indices)

        def get_object_find_prob(robot: str, location: str, obj: str) -> float:
            idx = location.split('_')[1]
            if idx == 'loc':
                return 0.0
            idx = int(idx)
            obj_name = obj.split('_')[0]
            '''TODO: Fix this (discuss to find more elegant solution). When reinstantiation of search operator occurs
            with newly found object, it is not found in dict, because we don't need to search for it. e.g.,
            objects_to_find = ['teddybear', 'pencil'], but after searching bed, robot finds 'pillow'. Reinstantiation
            of search operator with pillow for other containers fails here because pillow is not in objects_to_find.
            '''
            if obj_name not in object_container_prop_dict:
                return 0.0
            object_find_prob = round(object_container_prop_dict[obj_name][idx], 3)
            return object_find_prob

        return get_object_find_prob

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

    def compute_move_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        use_theta: bool = True,
    ) -> np.ndarray:
        """Compute 2xN grid path between two coordinates.

        Args:
            start: Start coordinate (x, y).
            end: End coordinate (x, y).
            use_theta: Whether to try Theta* before Dijkstra.

        Returns:
            2xN integer path array.
        """
        start_xy = (int(round(float(start[0]))), int(round(float(start[1]))))
        end_xy = (int(round(float(end[0]))), int(round(float(end[1]))))
        return procthor_utils.compute_move_path(
            self._thor.occupancy_grid,
            start_xy,
            end_xy,
            use_theta=use_theta,
        )

    def get_top_down_image(self, orthographic: bool = True) -> np.ndarray:
        """Get top-down view image of the scene."""
        return self._thor.get_top_down_image(orthographic=orthographic)

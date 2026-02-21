"""ProcTHOR scene data provider."""

from typing import Callable, Dict, List, Set, Tuple

import numpy as np

from .thor_interface import ThorInterface


class ProcTHORScene:
    """Data provider for ProcTHOR environments.

    Loads a ProcTHOR scene and extracts all information needed for planning:
    - Location names and coordinates
    - Object names and their ground truth locations

    Example:
        scene = ProcTHORScene(seed=4001)
        print(scene.locations)  # All container locations
        print(scene.objects)    # All objects in scene
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

    def get_object_find_prob_fn(
        self,
        nn_model_path: str,
    ) -> Callable[[str, str, str], float]:
        """Get learned object find probability function.

        Returns a function ``(robot, location, object) -> probability`` that
        computes and caches NN-based probabilities lazily per object on first
        access, so callers do not need to know the target object set upfront.
        """
        from . import learning
        obj_prob_net = learning.models.FCNN.get_net_eval_fn(nn_model_path)
        object_free_scene_graph = self.scene_graph.get_object_free_graph()
        containers = self.scene_graph.container_indices
        cache: dict[str, dict[int, float]] = {}

        def _ensure_cached(obj_name: str) -> dict[int, float]:
            if obj_name not in cache:
                node_features_dict = learning.utils.prepare_fcnn_input(
                    object_free_scene_graph, containers, [obj_name])
                datum = {'node_feats': node_features_dict[obj_name]}
                cache[obj_name] = obj_prob_net(datum, containers)
            return cache[obj_name]

        def get_object_find_prob(robot: str, location: str, obj: str) -> float:
            idx = location.split('_')[1]
            if idx == 'loc':
                return 0.0
            idx = int(idx)
            obj_name = obj.split('_')[0]
            return round(_ensure_cached(obj_name)[idx], 3)

        return get_object_find_prob

    def get_top_down_image(self, orthographic: bool = True) -> np.ndarray:
        """Get top-down view image of the scene."""
        return self._thor.get_top_down_image(orthographic=orthographic)

"""AI2-THOR interface for ProcTHOR environments."""

import copy
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from shapely import geometry

from railroad.navigation import pathing
from .scenegraph import SceneGraph
from . import utils
from .resources import get_procthor_10k_dir

IGNORE_CONTAINERS = [
    'baseballbat', 'basketball', 'boots', 'desklamp', 'painting',
    'floorlamp', 'houseplant', 'roomdecor', 'showercurtain',
    'showerhead', 'television', 'vacuumcleaner', 'photo', 'plunger',
    'box'
]


class ThorInterface:
    """Interface to AI2-THOR/ProcTHOR simulator.

    Handles scene loading, occupancy grid generation, and scene graph construction.

    Args:
        seed: Random seed for scene selection
        resolution: Grid resolution in meters
        preprocess: Whether to filter containers
        use_cache: Whether to use cached data
    """

    def __init__(
        self,
        seed: int,
        resolution: float = 0.05,
        preprocess: bool = True,
        use_cache: bool = True,
    ) -> None:
        self.seed = seed
        self.grid_resolution = resolution
        random.seed(seed)

        self.scene = self._load_scene()
        self.rooms = self.scene['rooms']
        self.agent = self.scene['metadata']['agent']

        self.containers = self.scene['objects']
        if preprocess:
            self._preprocess_containers()

        self.cached_data = self._load_cache() if use_cache else None
        if self.cached_data is None:
            from ai2thor.controller import Controller
            self.controller = Controller(
                scene=self.scene,
                gridSize=self.grid_resolution,
                width=480,
                height=480
            )
            self.cached_data = self._save_and_get_cache()
        else:
            print("-----------Using cached procthor data-----------")
            self.controller = None

        self.occupancy_grid = self._get_occupancy_grid()
        self.scene_graph = self._get_scene_graph()
        self.robot_pose = self._get_robot_pose()
        self.known_cost = self._get_known_costs()

    def _preprocess_containers(self) -> None:
        """Filter containers and their children."""
        container_types = {c['id'].split('|')[0].lower() for c in self.containers}

        for container in self.containers:
            if 'children' in container:
                container['children'] = [
                    child for child in container['children']
                    if child['id'].split('|')[0].lower() not in container_types
                ]

        self.containers = [
            c for c in self.containers
            if c['id'].split('|')[0].lower() not in IGNORE_CONTAINERS
        ]

    def _load_scene(self) -> Dict[str, Any]:
        """Load scene from ProcTHOR-10k dataset."""
        data_dir = get_procthor_10k_dir()
        with open(data_dir / 'data.jsonl', 'r') as f:
            json_list = list(f)
        return json.loads(json_list[self.seed])

    def _save_and_get_cache(self, path: str = './resources/procthor-10k/cache') -> Dict:
        """Cache expensive computations."""
        cache = {
            'reachable_positions': self._get_reachable_positions_from_controller(),
            'image_ortho': self._get_top_down_image_from_controller(orthographic=True),
            'image_persp': self._get_top_down_image_from_controller(orthographic=False)
        }
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f'scene_{self.seed}.pkl', 'wb') as f:
            pickle.dump(cache, f)
        return cache

    def _load_cache(self, path: str = './resources/procthor-10k/cache') -> Optional[Dict]:
        """Load cached scene data."""
        cache_file = Path(path) / f'scene_{self.seed}.pkl'
        if not cache_file.exists():
            return None
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    def _get_reachable_positions_from_controller(self) -> List[Dict[str, float]]:
        """Get reachable positions from controller."""
        assert self.controller is not None
        event = self.controller.step(action="GetReachablePositions")
        return event.metadata["actionReturn"]

    def get_reachable_positions(self) -> List[Dict[str, float]]:
        """Get reachable positions (from cache or controller)."""
        if self.cached_data is not None:
            return self.cached_data['reachable_positions']
        return self._get_reachable_positions_from_controller()

    def _set_grid_offset(self, min_x: float, min_y: float) -> None:
        """Set grid coordinate offset."""
        self.grid_offset = np.array([min_x, min_y])

    def scale_to_grid(self, point: Union[Tuple[float, float], Sequence[float]]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        x = round((point[0] - self.grid_offset[0]) / self.grid_resolution)
        y = round((point[1] - self.grid_offset[1]) / self.grid_resolution)
        return x, y

    def _get_robot_pose(self) -> Tuple[int, int]:
        """Get initial robot pose in grid coordinates."""
        position = self.agent['position']
        return self.scale_to_grid((position['x'], position['z']))

    def _get_occupancy_grid(self) -> np.ndarray:
        """Build occupancy grid from reachable positions."""
        rps = self.get_reachable_positions()
        xs = [rp["x"] for rp in rps]
        zs = [rp["z"] for rp in rps]

        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        x_offset = min_x - self.grid_resolution if min_x < 0 else 0
        z_offset = min_z - self.grid_resolution if min_z < 0 else 0
        self._set_grid_offset(x_offset, z_offset)

        points = list(zip(xs, zs))
        self.g2p_map = {self.scale_to_grid(p): rps[i] for i, p in enumerate(points)}

        height, width = self.scale_to_grid([max_x, max_z])
        occupancy_grid = np.ones((height + 2, width + 2), dtype=int)
        for pos in self.g2p_map.keys():
            occupancy_grid[pos] = 0

        # Update container positions to nearest free grid cell
        for container in self.containers:
            position = container['position']
            if position is not None:
                nearest_fp = utils.get_nearest_free_point(position, points)
                scaled = self.scale_to_grid((nearest_fp[0], nearest_fp[1]))
                container['position'] = scaled
                container['id'] = container['id'].lower()

                if 'children' in container:
                    for child in container['children']:
                        child['position'] = container['position']
                        child['id'] = child['id'].lower()

        for room in self.rooms:
            floor = [(rp["x"], rp["z"]) for rp in room["floorPolygon"]]
            room_poly = geometry.Polygon(floor)
            point = room_poly.centroid
            nearest_fp = utils.get_nearest_free_point({'x': point.x, 'z': point.y}, points)
            room['position'] = self.scale_to_grid((nearest_fp[0], nearest_fp[1]))

        return occupancy_grid

    def _get_scene_graph(self) -> SceneGraph:
        """Build scene graph from scene data."""
        graph = SceneGraph()

        # Add apartment node
        apt_idx = graph.add_node({
            'id': 'Apartment|0',
            'name': 'apartment',
            'position': (0, 0),
            'type': [1, 0, 0, 0]
        })

        # Add room nodes
        for room in self.rooms:
            room_idx = graph.add_node({
                'id': room['id'],
                'name': room['roomType'].lower(),
                'position': room['position'],
                'type': [0, 1, 0, 0]
            })
            graph.add_edge(apt_idx, room_idx)

        # Add edges between connected rooms
        room_indices = graph.room_indices
        for i, src_idx in enumerate(room_indices):
            for dst_idx in room_indices[i + 1:]:
                src_node = graph.nodes[src_idx]
                dst_node = graph.nodes[dst_idx]
                if utils.has_edge(self.scene['doors'], src_node['id'], dst_node['id']):
                    graph.add_edge(src_idx, dst_idx)

        # Add container nodes
        for container in self.containers:
            room_id = utils.get_room_id(container['id'])
            room_node_idx = next(
                idx for idx, node in graph.nodes.items()
                if node['type'][1] == 1 and utils.get_room_id(node['id']) == room_id
            )
            cnt_idx = graph.add_node({
                'id': container['id'],
                'name': utils.get_generic_name(container['id']),
                'position': container['position'],
                'type': [0, 0, 1, 0]
            })
            graph.add_edge(room_node_idx, cnt_idx)

        # Add object nodes
        for container in self.containers:
            children = container.get('children', [])
            if children:
                cnt_idx = graph.asset_id_to_node_idx_map[container['id']]
                for obj in children:
                    obj_idx = graph.add_node({
                        'id': obj['id'],
                        'name': utils.get_generic_name(obj['id']),
                        'position': obj['position'],
                        'type': [0, 0, 0, 1]
                    })
                    graph.add_edge(cnt_idx, obj_idx)

        # Ensure connectivity
        graph.edges.extend(utils.get_edges_for_connected_graph(
            self.occupancy_grid,
            {
                'nodes': graph.nodes,
                'edge_index': graph.edges,
                'cnt_node_idx': graph.container_indices,
                'obj_node_idx': graph.object_indices,
                'idx_map': graph.asset_id_to_node_idx_map
            },
            pos='position'
        ))

        return graph

    def _get_known_costs(self) -> Dict[str, Dict[str, float]]:
        """Pre-compute costs between all containers."""
        known_cost: Dict[str, Dict[str, float]] = {'initial_robot_pose': {}}
        init_r = [self.robot_pose[0], self.robot_pose[1]]
        cnt_ids = ['initial_robot_pose'] + [c['id'] for c in self.containers]
        cnt_positions = [init_r] + [c['position'] for c in self.containers]

        for i, cnt1_id in enumerate(cnt_ids):
            known_cost[cnt1_id] = {}
            for j, cnt2_id in enumerate(cnt_ids):
                if cnt2_id not in known_cost:
                    known_cost[cnt2_id] = {}
                if cnt1_id == cnt2_id:
                    known_cost[cnt1_id][cnt2_id] = 0.0
                    continue
                cost, _ = pathing.get_cost_and_path(
                    self.occupancy_grid,
                    (int(cnt_positions[i][0]), int(cnt_positions[i][1])),
                    (int(cnt_positions[j][0]), int(cnt_positions[j][1])),
                    use_soft_cost=True,
                    unknown_as_obstacle=False,
                    soft_cost_scale=12.0,
                )
                known_cost[cnt1_id][cnt2_id] = round(cost, 4)
                known_cost[cnt2_id][cnt1_id] = round(cost, 4)

        return known_cost

    def _get_top_down_image_from_controller(self, orthographic: bool = True) -> np.ndarray:
        """Get top-down image from controller."""
        assert self.controller is not None
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = orthographic
        pose["farClippingPlane"] = 50
        if orthographic:
            pose["orthographicSize"] = 0.5 * max_bound
        else:
            del pose["orthographicSize"]

        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        image = event.third_party_camera_frames[-1]
        return image[::-1, ...]

    def get_top_down_image(self, orthographic: bool = True) -> np.ndarray:
        """Get top-down image (from cache or controller)."""
        if self.cached_data is not None:
            key = 'image_ortho' if orthographic else 'image_persp'
            return self.cached_data[key]
        return self._get_top_down_image_from_controller(orthographic)

    def get_target_objs_info(self, num_objects: int = 1) -> Dict | List[Dict]:
        """Get info about target objects for search tasks."""
        object_name_to_idxs: Dict[str, List[int]] = {}
        for idx in self.scene_graph.object_indices:
            name = self.scene_graph.get_node_name_by_idx(idx)
            object_name_to_idxs.setdefault(name, []).append(idx)

        num_objects = min(num_objects, len(object_name_to_idxs))
        target_names = random.sample(list(object_name_to_idxs.keys()), num_objects)

        result = []
        for name in target_names:
            idxs = object_name_to_idxs[name]
            container_idxs = [self.scene_graph.get_parent_node_idx(idx) for idx in idxs]
            result.append({
                'name': name,
                'idxs': idxs,
                'type': self.scene_graph.nodes[idxs[0]]['type'],
                'container_idxs': container_idxs
            })

        return result[0] if num_objects == 1 else result

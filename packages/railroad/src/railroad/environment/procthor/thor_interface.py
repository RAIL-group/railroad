"""AI2-THOR interface for ProcTHOR environments."""

import copy
import io
import json
import os
import pickle
import random
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from shapely import geometry

from railroad.environment.types import TopDownView
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

SCENE_CACHE_VERSION = 3
"""Version 2 adds ``image_ortho_extent_m``; version 3 stores images as JPEG.

The extent is stored even though :meth:`ThorInterface.top_down_footprint`
recomputes it, because its *presence* is what distinguishes an image framed
that way from a version 1 image framed on AI2-THOR's ``sceneBounds``. A missing
extent degrades to no overhead image rather than relaunching Unity for every
cached scene; raw-array images from either older version still load.
"""

_MAP_VIEW_ROTATION = {"x": 90.0, "y": 0.0, "z": 0.0}
"""Straight down, with camera-right along world +x and camera-up along +z."""

TOP_DOWN_RENDER_PX = 2048
"""Square render size for the cached top-down images.

Saved plots are dpi=300, which puts ~2000 pixels across the map axes; at the
old 480 the image was upscaled about four times. Costs generation only -- the
controller is stopped as soon as the cache is written.
"""

JPEG_QUALITY = 75
"""Cached images are JPEG, which is ~50x smaller than raw at this size."""

TOP_DOWN_MARGIN_M = 0.5
"""Slack around the room polygons, for wall thickness and exterior trim."""

_EXTENT_WARNED: set = set()
"""Seeds already warned about, so a per-frame render path stays quiet.

Not left to ``warnings``' own deduplication, which is reset to "always" inside
``pytest.warns``.
"""


def _encode_image(image: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, "JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def _decode_image(data: Any) -> Any:
    """Cached images are JPEG bytes; older caches hold raw arrays."""
    if not isinstance(data, bytes):
        return data
    from PIL import Image

    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def _is_axis_aligned_top_down(rotation: Any) -> bool:
    """Whether *rotation* already points straight down with no yaw."""
    if not isinstance(rotation, dict):
        return False
    return all(
        abs(float(rotation.get(axis, 0.0)) - expected) < 1e-6
        for axis, expected in _MAP_VIEW_ROTATION.items()
    )


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
            # Generating a scene starts a Unity controller. Several of those at
            # once will take a machine down, and benchmark workers are separate
            # processes, so this is a cross-process file lock rather than a
            # threading one. Re-check the cache once we hold it: the process
            # ahead of us in the queue may have generated this very scene.
            from ._scene_lock import scene_generation_lock

            with scene_generation_lock():
                self.cached_data = self._load_cache() if use_cache else None
                if self.cached_data is None:
                    from ai2thor.controller import Controller
                    self.controller = Controller(
                        scene=self.scene,
                        gridSize=self.grid_resolution,
                        width=TOP_DOWN_RENDER_PX,
                        height=TOP_DOWN_RENDER_PX,
                    )
                    self.cached_data = self._save_and_get_cache()
                    # Inside the lock: otherwise every worker that generates a
                    # scene keeps its Unity alive for the rest of the run, and
                    # they accumulate exactly as the lock exists to prevent.
                    self.controller.stop()
                    self.controller = None
                else:
                    print("-----------Using cached procthor data-----------")
                    self.controller = None
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

    def _cache_dir(self) -> Path:
        """Where per-seed scene caches live.

        Follows the resources directory, so ``PROCTHOR_RESOURCES_DIR`` reaches
        it. Unset, that still falls back to ``Path.cwd() / "resources"`` fixed
        at import, so running from elsewhere misses every cached scene and
        starts Unity -- set the variable to work from more than one directory.
        """
        return get_procthor_10k_dir() / 'cache'

    def _save_and_get_cache(self, path: Optional[str] = None) -> Dict:
        """Cache expensive computations."""
        image_ortho, extent_m = self._render_top_down_from_controller(orthographic=True)
        image_persp, _ = self._render_top_down_from_controller(orthographic=False)
        cache = {
            'cache_version': SCENE_CACHE_VERSION,
            'reachable_positions': self._get_reachable_positions_from_controller(),
            'image_ortho': _encode_image(image_ortho),
            'image_persp': _encode_image(image_persp),
            # Meters, not cells: the filename encodes no resolution, but
            # `resolution` is a constructor argument, so cells would silently
            # corrupt any run that does not use the default. Plain floats, not
            # a dataclass -- pickling one pins its module path, and renaming it
            # later would turn every cache into an AttributeError on load.
            'image_ortho_extent_m': extent_m,
        }
        save_dir = Path(path) if path is not None else self._cache_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        self._write_cache_atomically(cache, save_dir / f'scene_{self.seed}.pkl')
        return cache

    @staticmethod
    def _write_cache_atomically(cache: Dict, target: Path) -> None:
        """Write *cache* to *target*, or leave whatever was there alone.

        Dumping straight to the destination leaves a truncated pickle on a
        Ctrl-C, and the scene is then permanently broken rather than simply
        regenerated. Not a small window: two 480x480 images plus thousands of
        reachable positions, slow enough that interrupting is normal. Writing
        beside the target and renaming makes the swap atomic.
        """
        handle, temp_name = tempfile.mkstemp(
            dir=target.parent, prefix=f'{target.stem}.', suffix='.tmp',
        )
        try:
            with os.fdopen(handle, 'wb') as file:
                pickle.dump(cache, file)
            # mkstemp is 0600, where the plain open() this replaced took the
            # umask default. PROCTHOR_RESOURCES_DIR exists to put this cache
            # somewhere shared, and a 0600 entry there is unreadable to the
            # next user -- who then regenerates it, launching Unity, every run.
            umask = os.umask(0o777)
            os.umask(umask)
            os.chmod(temp_name, 0o666 & ~umask)
            os.replace(temp_name, target)
        except BaseException:  # including KeyboardInterrupt, the likely one
            Path(temp_name).unlink(missing_ok=True)
            raise

    def _load_cache(self, path: Optional[str] = None) -> Optional[Dict]:
        """Load cached scene data, treating an unreadable file as a miss."""
        base = Path(path) if path is not None else self._cache_dir()
        cache_file = base / f'scene_{self.seed}.pkl'
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as error:
            # Recover from files written before the atomic swap above, rather
            # than failing every run until someone deletes them by hand. Loud,
            # because regenerating means starting Unity.
            warnings.warn(
                f"ProcTHOR scene cache {cache_file} could not be read "
                f"({type(error).__name__}: {error}); regenerating it.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

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

    def scale_to_grid_continuous(
        self, point: Union[Tuple[float, float], Sequence[float]]
    ) -> Tuple[float, float]:
        """World (x, z) meters -> fractional grid cells, unrounded.

        Callers positioning an image need this: rounding an outer edge to a
        cell center shifts it by up to half a cell.
        """
        x = (point[0] - self.grid_offset[0]) / self.grid_resolution
        y = (point[1] - self.grid_offset[1]) / self.grid_resolution
        return x, y

    def scale_to_grid(self, point: Union[Tuple[float, float], Sequence[float]]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        x, y = self.scale_to_grid_continuous(point)
        return round(x), round(y)

    def grid_to_world(
        self, cell: Union[Tuple[float, float], Sequence[float]]
    ) -> Tuple[float, float]:
        """Inverse of :meth:`scale_to_grid`: world (x, z) at a cell's center."""
        x = float(cell[0]) * self.grid_resolution + float(self.grid_offset[0])
        y = float(cell[1]) * self.grid_resolution + float(self.grid_offset[1])
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

    def top_down_footprint(self) -> Tuple[float, float, float, float]:
        """World footprint the camera frames, ``(min_x, max_x, min_z, max_z)``.

        Chosen here rather than taken from THOR's map-view camera, so it is
        known offline. THOR frames on ``sceneBounds`` -- the union of every
        enabled renderer, which sweeps in geometry invisible from above and is
        inconsistent enough across ProcTHOR-10k to be an open upstream bug
        (allenai/ai2thor#1181).

        The room floor polygons are exact, in the scene JSON, with the walls on
        their boundary, so the house plus a margin frames the whole scene
        including rooms the agent cannot enter. Square, since a third-party
        camera inherits the main camera's resolution.
        """
        corners = [
            (vertex["x"], vertex["z"])
            for room in self.rooms
            for vertex in room["floorPolygon"]
        ]
        xs = [corner[0] for corner in corners]
        zs = [corner[1] for corner in corners]
        center_x = 0.5 * (min(xs) + max(xs))
        center_z = 0.5 * (min(zs) + max(zs))
        half = 0.5 * max(max(xs) - min(xs), max(zs) - min(zs)) + TOP_DOWN_MARGIN_M
        return (center_x - half, center_x + half, center_z - half, center_z + half)

    def _render_top_down_from_controller(
        self, orthographic: bool = True
    ) -> Tuple[np.ndarray, Optional[Tuple[float, float, float, float]]]:
        """Render a top-down frame, plus its world footprint in meters.

        The second element is ``(min_x, max_x, min_z, max_z)`` for the
        orthographic camera and ``None`` for the perspective one -- a
        projective view has no rectangular footprint, so there is nothing
        honest to return.
        """
        assert self.controller is not None
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        # Pin the camera straight down. A non-zero yaw would rotate the world
        # footprint out of axis-alignment, which no rectangular imshow extent
        # can express -- the overlay would be silently skewed rather than
        # obviously broken. Warn rather than swallow it if THOR ever disagrees.
        returned_rotation = pose.get("rotation")
        pose["rotation"] = dict(_MAP_VIEW_ROTATION)
        if orthographic and not _is_axis_aligned_top_down(returned_rotation):
            warnings.warn(
                f"AI2-THOR's map-view camera for scene {self.seed} returned "
                f"rotation {returned_rotation}; overriding with "
                f"{_MAP_VIEW_ROTATION} so the render stays axis-aligned.",
                RuntimeWarning,
                stacklevel=2,
            )

        pose["fieldOfView"] = 50
        # Height only, so it never affects an orthographic footprint -- this is
        # the one place sceneBounds stays useful, lifting the camera clear of
        # whatever the scene contains.
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = orthographic
        pose["farClippingPlane"] = 50

        extent_m = None
        if orthographic:
            extent_m = self.top_down_footprint()
            min_x, max_x, min_z, max_z = extent_m
            pose["position"]["x"] = 0.5 * (min_x + max_x)
            pose["position"]["z"] = 0.5 * (min_z + max_z)
            # Unity's orthographicSize is the camera's HALF-height in world
            # units. The footprint is square and the frame is square, so this
            # is also its half-width.
            pose["orthographicSize"] = 0.5 * (max_z - min_z)
        else:
            del pose["orthographicSize"]

        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        image = event.third_party_camera_frames[-1][::-1, ...]

        if extent_m is not None:
            # The footprint above assumes a square frame. Guard it rather than
            # trust it: a non-square controller resolution would stretch the
            # horizontal axis and silently skew every overlay.
            height_px, width_px = image.shape[:2]
            if height_px != width_px:
                raise RuntimeError(
                    f"top-down render is {width_px}x{height_px}; the orthographic "
                    "footprint is only square when the frame is. Set the "
                    "controller to a square resolution, or widen the footprint "
                    "by the aspect ratio here."
                )
        return image, extent_m

    def get_top_down_image(self, orthographic: bool = True) -> np.ndarray:
        """Get top-down image (from cache or controller)."""
        if self.cached_data is not None:
            key = 'image_ortho' if orthographic else 'image_persp'
            return _decode_image(self.cached_data[key])
        return self._render_top_down_from_controller(orthographic)[0]

    def get_top_down_view(self) -> Optional[TopDownView]:
        """The orthographic top-down image, placed on the occupancy grid.

        Returns ``None`` -- after warning once for this scene -- when the
        cached scene predates the recorded camera extent. An unpositioned
        image can only be drawn misaligned, which is worse than no image.
        """
        if self.cached_data is not None:
            image = _decode_image(self.cached_data.get('image_ortho'))
            extent_m = self.cached_data.get('image_ortho_extent_m')
            if image is None or extent_m is None:
                self._warn_unplaceable(
                    "was cached before the top-down camera extent was recorded"
                )
                return None
            # The footprint is recomputable, so a cache that disagrees with it
            # was rendered against different framing -- a changed
            # TOP_DOWN_MARGIN_M, say. Its pixels sit somewhere else than we
            # would now say, so treat it as stale rather than misplace it.
            expected = self.top_down_footprint()
            if not np.allclose(extent_m, expected, atol=1e-6):
                self._warn_unplaceable(
                    f"was rendered with the footprint "
                    f"{tuple(round(v, 3) for v in extent_m)}, but this build "
                    f"frames it at {tuple(round(v, 3) for v in expected)}"
                )
                return None
        else:
            image, extent_m = self._render_top_down_from_controller(orthographic=True)
            if extent_m is None:
                return None
        return self._view_from_extent(image, extent_m)

    def _view_from_extent(
        self, image: np.ndarray, extent_m: Tuple[float, float, float, float],
    ) -> TopDownView:
        """Place an image given its world footprint, in meters.

        The bounds are outer *edges*, and ``scale_to_grid_continuous`` maps
        meters to cell coordinates where integers are cell centers, which is
        where matplotlib draws them. So no half-cell correction belongs here --
        rounding, as ``scale_to_grid`` does, would put an edge on a center.
        """
        min_x, min_y = self.scale_to_grid_continuous((extent_m[0], extent_m[2]))
        max_x, max_y = self.scale_to_grid_continuous((extent_m[1], extent_m[3]))
        return TopDownView(
            image=image, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
        )

    def _warn_unplaceable(self, reason: str) -> None:
        """Warn once per scene that its cached image cannot be positioned."""
        if self.seed in _EXTENT_WARNED:
            return
        _EXTENT_WARNED.add(self.seed)
        warnings.warn(
            f"ProcTHOR scene {self.seed} {reason}, so its overhead image cannot "
            f"be aligned with the occupancy grid and will be omitted. Delete "
            f"{self._cache_dir()} to regenerate it (this relaunches Unity for "
            f"the scenes you use).",
            RuntimeWarning,
            stacklevel=3,
        )
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

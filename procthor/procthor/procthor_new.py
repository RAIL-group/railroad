import json
import numpy as np
import random
import copy
from shapely import geometry
from common import Pose
from ai2thor.controller import Controller
from . import utils
from .scenegraph import SceneGraph
import pickle
from pathlib import Path

IGNORE_CONTAINERS = [
    'baseballbat', 'basketBall', 'boots', 'desklamp', 'painting',
    'floorlamp', 'houseplant', 'roomdecor', 'showercurtain',
    'showerhead', 'television', 'vacuumcleaner', 'photo', 'plunger',
    'basketball', 'box'
]


class ThorInterface:
    def __init__(self, args, preprocess=True, use_cache=True):
        self.args = args
        self.seed = args.current_seed
        random.seed(self.seed)

        self.grid_resolution = args.resolution
        self.scene = self.load_scene()

        self.rooms = self.scene['rooms']
        self.agent = self.scene['metadata']['agent']

        self.containers = self.scene['objects']
        if preprocess:
            # prevent adding objects if a container of that type already exists
            container_types = set()
            for container in self.containers:
                container_types.add(container['id'].split('|')[0].lower())
            for container in self.containers:
                filtered_children = []
                if 'children' in container:
                    for child in container['children']:
                        if child['id'].split('|')[0].lower() in container_types:
                            continue
                        filtered_children.append(child)
                    container['children'] = filtered_children
            # filter containers from IGNORE list
            self.containers = [
                container for container in self.containers
                if container['id'].split('|')[0].lower() not in IGNORE_CONTAINERS
            ]

        self.cached_data = self.load_cache() if use_cache else None
        if self.cached_data is None:
            self.controller = Controller(scene=self.scene,
                                         gridSize=self.grid_resolution,
                                         width=480, height=480)
            self.cached_data = self.save_and_get_cache()
        else:
            print("-----------Using cached procthor data-----------")
            self.controller = None

        self.occupancy_grid = self.get_occupancy_grid()
        self.scene_graph = self.get_scene_graph()
        self.robot_pose = self.get_robot_pose()
        self.known_cost = self.get_known_costs()

    def save_and_get_cache(self, path='/resources/procthor-10k/cache'):
        """Cache the scene data."""
        cache = {
            'reachable_positions': self.get_reachable_positions(),
            'image_ortho': self.get_top_down_image(orthographic=True),
            'image_persp': self.get_top_down_image(orthographic=False)
        }
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f'scene_{self.args.current_seed}.pkl', 'wb') as f:
            pickle.dump(cache, f)
        return cache

    def load_cache(self, path='/resources/procthor-10k/cache'):
        """Load the cached scene data."""
        cache_file = Path(path) / f'scene_{self.args.current_seed}.pkl'
        if not cache_file.exists():
            return None
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        return cache

    def gen_map_and_poses(self, num_objects=1):
        """Returns scene graph, grid, initial robot pose and target object info."""
        self.target_objs_info = self.get_target_objs_info(num_objects)
        return (self.scene_graph,
                self.occupancy_grid,
                self.robot_pose,
                self.target_objs_info)

    def load_scene(self, path='/resources/procthor-10k'):
        with open(
            f'{path}/data.jsonl',
            "r",
        ) as json_file:
            json_list = list(json_file)
        return json.loads(json_list[self.seed])

    def set_grid_offset(self, min_x, min_y):
        self.grid_offset = np.array([min_x, min_y])

    def scale_to_grid(self, point):
        x = round((point[0] - self.grid_offset[0]) / self.grid_resolution)
        y = round((point[1] - self.grid_offset[1]) / self.grid_resolution)
        return x, y

    def get_robot_pose(self):
        position = self.agent['position']
        position = np.array([position['x'], position['z']])
        pose = self.scale_to_grid(position)
        return Pose(*pose)

    def get_target_objs_info(self, num_objects=1):
        object_name_to_idxs = {}
        for idx in self.scene_graph.object_indices:
            name = self.scene_graph.get_node_name_by_idx(idx)
            if name not in object_name_to_idxs.keys():
                object_name_to_idxs[name] = [idx]
            else:
                object_name_to_idxs[name].append(idx)
        if num_objects > len(object_name_to_idxs):
            num_objects = len(object_name_to_idxs)

        target_obj_names = random.sample(list(object_name_to_idxs.keys()), num_objects)
        target_objs_info = []
        for name in target_obj_names:
            idxs = object_name_to_idxs[name]
            container_idxs = [self.scene_graph.get_parent_node_idx(idx) for idx in idxs]
            node_type = self.scene_graph.nodes[idxs[0]]['type']
            target_objs_info.append({
                'name': name,
                'idxs': object_name_to_idxs[name],
                'type': node_type,
                'container_idxs': container_idxs
            })

        if num_objects == 1:
            return target_objs_info[0]
        return target_objs_info

    def get_reachable_positions(self):
        if self.cached_data is not None:
            return self.cached_data['reachable_positions']
        event = self.controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]
        return reachable_positions

    def get_occupancy_grid(self):
        rps = self.get_reachable_positions()
        xs = [rp["x"] for rp in rps]
        zs = [rp["z"] for rp in rps]

        # Calculate the mins and maxs
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        x_offset = min_x - self.grid_resolution if min_x < 0 else 0
        z_offset = min_z - self.grid_resolution if min_z < 0 else 0
        self.set_grid_offset(x_offset, z_offset)

        # Create list of free points
        points = list(zip(xs, zs))
        grid_to_points_map = {self.scale_to_grid(point): rps[idx]
                              for idx, point in enumerate(points)}
        height, width = self.scale_to_grid([max_x, max_z])
        occupancy_grid = np.ones((height + 2, width + 2), dtype=int)
        free_positions = grid_to_points_map.keys()
        for pos in free_positions:
            occupancy_grid[pos] = 0

        # store the mapping from grid coordinates to simulator positions
        self.g2p_map = grid_to_points_map

        # set the nearest freespace container positions
        for container in self.containers:
            position = container['position']
            if position is not None:
                # get nearest free space pose
                nearest_fp = utils.get_nearest_free_point(position, points)
                # then scale the free space pose to grid
                scaled_position = self.scale_to_grid(np.array([nearest_fp[0], nearest_fp[1]]))  # noqa: E501
                # finally set the scaled grid pose as the container position
                container['position'] = scaled_position  # 2d only
                container['id'] = container['id'].lower()  # 2d only

                # next do the same if there is any children of this container
                if 'children' in container:
                    children = container['children']
                    for child in children:
                        child['position'] = container['position']
                        child['id'] = child['id'].lower()

        for room in self.rooms:
            floor = [(rp["x"], rp["z"]) for rp in room["floorPolygon"]]
            room_poly = geometry.Polygon(floor)
            point = room_poly.centroid
            point = {'x': point.x, 'z': point.y}
            nearest_fp = utils.get_nearest_free_point(point, points)
            scaled_position = self.scale_to_grid(np.array([nearest_fp[0], nearest_fp[1]]))  # noqa: E501
            room['position'] = scaled_position  # 2d only

        return occupancy_grid

    def get_scene_graph(self):
        """Create a scene graph from scene data."""
        graph = SceneGraph()

        # Add apartment node
        apartment_idx = graph.add_node(
            {
                'id': 'Apartment|0',
                'name': 'apartment',
                'position': (0, 0),
                'type': [1, 0, 0, 0]
            }
        )

        # Add room nodes
        for room in self.rooms:
            room_idx = graph.add_node(
                {
                    'id': room['id'],
                    'name': room['roomType'].lower(),
                    'position': room['position'],
                    'type': [0, 1, 0, 0]
                }
            )
            graph.add_edge(apartment_idx, room_idx)

        # Add edges between connected rooms
        room_indices = graph.get_node_indices_by_type(1)
        for i, src_idx in enumerate(room_indices):
            for dst_idx in room_indices[i + 1:]:
                src_node = graph.nodes[src_idx]
                dst_node = graph.nodes[dst_idx]
                if utils.has_edge(self.scene['doors'], src_node['id'], dst_node['id']):
                    graph.add_edge(src_idx, dst_idx)

        # Add container nodes
        for container in self.containers:
            room_id = utils.get_room_id(container['id'])
            room_node_idx = next(idx for idx, node in graph.nodes.items()
                                 if node['type'][1] == 1 and utils.get_room_id(node['id']) == room_id)

            container_idx = graph.add_node(
                {
                    'id': container['id'],
                    'name': utils.get_generic_name(container['id']),
                    'position': container['position'],
                    'type': [0, 0, 1, 0]
                }
            )
            # graph.asset_id_to_node_idx_map[container['id']] = container_idx
            graph.add_edge(room_node_idx, container_idx)

        # Add object nodes for container contents
        for container in self.containers:
            connected_objects = container.get('children')
            if connected_objects is not None:
                container_idx = graph.asset_id_to_node_idx_map[container['id']]
                for obj in connected_objects:
                    # graph.asset_id_to_node_idx_map[obj['id']] = len(graph)
                    obj_idx = graph.add_node(
                        {
                            'id': obj['id'],
                            'name': utils.get_generic_name(obj['id']),
                            'position': obj['position'],
                            'type': [0, 0, 0, 1]
                        }
                    )
                    graph.add_edge(container_idx, obj_idx)

        # Ensure graph connectivity
        graph.ensure_connectivity(self.occupancy_grid)

        return graph

    def get_known_costs(self):
        known_cost = {'initial_robot_pose': {}}
        init_r = [self.robot_pose.x, self.robot_pose.y]
        cnt_ids = ['initial_robot_pose'] + [cnt['id'] for cnt in self.containers]
        cnt_positions = [init_r] + [cnt['position'] for cnt in self.containers]

        # get cost from one container to another
        for index1, cnt1_id in enumerate(cnt_ids):
            cnt1_position = cnt_positions[index1]
            known_cost[cnt1_id] = {}
            for index2, cnt2_id in enumerate(cnt_ids):
                if cnt2_id not in known_cost:
                    known_cost[cnt2_id] = {}
                if cnt1_id == cnt2_id:
                    known_cost[cnt1_id][cnt2_id] = 0.0
                    continue
                cnt2_position = cnt_positions[index2]
                cost = utils.get_cost(grid=self.occupancy_grid,
                                      robot_pose=cnt1_position,
                                      end=cnt2_position)
                known_cost[cnt1_id][cnt2_id] = round(cost, 4)
                known_cost[cnt2_id][cnt1_id] = round(cost, 4)

        return known_cost

    def get_top_down_image(self, orthographic=True):
        if self.cached_data is not None:
            if orthographic:
                return self.cached_data['image_ortho']
            else:
                return self.cached_data['image_persp']
        # Setup top down camera
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

        # Add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_image = event.third_party_camera_frames[-1]
        top_down_image = top_down_image[::-1, ...]
        return top_down_image

import json
import copy
import numpy as np
from shapely import geometry
from ai2thor.controller import Controller

import procthor
from . import utils

IGNORE_CONTAINERS = [
    'baseballbat', 'basketBall', 'boots', 'desklamp', 'painting',
    'floorlamp', 'houseplant', 'roomdecor', 'showercurtain',
    'showerhead', 'television', 'vacuumcleaner', 'photo', 'plunger',
    'basketball', 'box'
]


class ThorInterface:
    def __init__(self, args, preprocess=True):
        self.args = args
        self.seed = args.current_seed
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

        self.cache = None
        if hasattr(args, "cache_path") and args.cache_path:
            self.cache = procthor.utils.load_cache(self.seed, args.cache_path)
        if self.cache is None:
            self.controller = Controller(scene=self.scene,
                                         gridSize=self.grid_resolution,
                                         width=480, height=480)
        self.occupancy_grid = self.get_occupancy_grid()

        if type(preprocess) is dict:
            # add custom objects in their prefered locations; todo so,
            # we first check if necessary container exists, create an
            # eligible pool dictionary then randomly choose 1 or 40% from
            # the pool, whichever max and add it to those containers
            eligible_pool = {}
            for container in self.containers:
                cnt_ID = container['id'].split('|')[0].lower()
                for obj in preprocess:
                    if cnt_ID in preprocess[obj]:
                        if obj not in eligible_pool:
                            eligible_pool[obj] = [container['id']]
                        else:
                            eligible_pool[obj].append(container['id'])
            chosen_containers = {}
            for obj in eligible_pool:
                choice_count = max(1, int(0.4 * len(eligible_pool[obj])))
                chosen_containers[obj] = np.random.choice(eligible_pool[obj], choice_count)

            counter = 0
            for container in self.containers:
                for obj in chosen_containers:
                    if container['id'] in chosen_containers[obj]:
                        counter += 1
                        if 'children' not in container:
                            container['children'] = []
                        container['children'].append({
                            'id': f'{obj}|surface|{counter}',
                            'kinematic': False,
                            'position': container['position']
                        })

        self.known_cost = self.get_known_costs()

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
        return self.scale_to_grid(position)

    def get_occupancy_grid(self):
        if self.cache:
            RPs = self.cache['RPs']
        else:
            event = self.controller.step(action="GetReachablePositions")
            reachable_positions = event.metadata["actionReturn"]
            RPs = reachable_positions
            # save the data in cache as pickle and create necessary directories
            if hasattr(self.args, "cache_path") and self.args.cache_path:
                data = {
                    'RPs': RPs,
                    'image': self.get_top_down_frame()
                }
                procthor.utils.save_cache(self.seed, self.args.cache_path, data)

        xs = [rp["x"] for rp in RPs]
        zs = [rp["z"] for rp in RPs]

        # Calculate the mins and maxs
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        x_offset = min_x - self.grid_resolution if min_x < 0 else 0
        z_offset = min_z - self.grid_resolution if min_z < 0 else 0
        self.set_grid_offset(x_offset, z_offset)

        # Create list of free points
        points = list(zip(xs, zs))
        grid_to_points_map = {self.scale_to_grid(point): RPs[idx]
                              for idx, point in enumerate(points)}
        height, width = self.scale_to_grid([max_x, max_z])
        occupancy_grid = np.ones((height + 2, width + 2), dtype=int)
        free_positions = grid_to_points_map.keys()
        for pos in free_positions:
            occupancy_grid[pos] = 0

        min_height, min_width = self.scale_to_grid([min_x, min_z])
        if height > width:
            offset_x = - min_height / 2
            offset_z = (height - width) / 2 - min_width / 2
            x_max = height
            z_max = height
        else:
            offset_x = (width - height) / 2 - min_height / 2
            offset_z = - min_width / 2
            x_max = width
            z_max = width
        self.plot_offset = [offset_x, offset_z]
        self.plot_extent = [0, x_max, z_max, 0]

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

    def get_top_down_frame(self):
        if self.cache:
            return self.cache['image']
        # Setup the top-down camera
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = True
        pose["farClippingPlane"] = 50
        pose["orthographicSize"] = .5 * max_bound

        # add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_frame = event.third_party_camera_frames[-1]
        top_down_frame = top_down_frame[::-1, ...]
        return top_down_frame

    def get_graph(self, include_node_embeddings=True):
        ''' This method creates graph data from the procthor-10k data'''

        # Create dummy apartment node
        node_count = 0
        nodes = {}
        assetId_idx_map = {}
        edges = []
        nodes[node_count] = {
            'id': 'Apartment|0',
            'name': 'apartment',
            'pos': (0, 0),
            'type': [1, 0, 0, 0]
        }
        node_count += 1

        # Iterate over rooms but skip position coordinate scaling since not
        # required in distance calculations
        for room in self.rooms:
            nodes[node_count] = {
                'id': room['id'],
                'name': room['roomType'].lower(),
                'pos': room['position'],
                'type': [0, 1, 0, 0]
            }
            edges.append(tuple([0, node_count]))
            node_count += 1

        # add an edge between two rooms adjacent by a passable shared door
        room_edges = set()
        for i in range(1, len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_1, node_2 = nodes[i], nodes[j]
                if utils.has_edge(self.scene['doors'], node_1['id'], node_2['id']):
                    room_edges.add(tuple(sorted((i, j))))
        edges.extend(room_edges)

        node_keys = list(nodes.keys())
        node_ids = [utils.get_room_id(nodes[key]['id']) for key in node_keys]
        cnt_node_idx = []

        for container in self.containers:
            cnt_id = utils.get_room_id(container['id'])
            src = node_ids.index(cnt_id)
            assetId = container['id']
            assetId_idx_map[assetId] = node_count
            name = utils.get_generic_name(container['id'])
            nodes[node_count] = {
                'id': container['id'],
                'name': name,
                'pos': container['position'],
                'type': [0, 0, 1, 0]
            }
            edges.append(tuple([src, node_count]))
            cnt_node_idx.append(node_count)
            node_count += 1

        node_keys = list(nodes.keys())
        node_ids = [nodes[key]['id'] for key in node_keys]
        obj_node_idx = []

        for container in self.containers:
            connected_objects = container.get('children')
            if connected_objects is not None:
                src = node_ids.index(container['id'])
                for object in connected_objects:
                    assetId = object['id']
                    assetId_idx_map[assetId] = node_count
                    name = utils.get_generic_name(object['id'])
                    nodes[node_count] = {
                        'id': object['id'],
                        'name': name,
                        'pos': object['position'],
                        'type': [0, 0, 0, 1]
                    }
                    edges.append(tuple([src, node_count]))
                    obj_node_idx.append(node_count)
                    node_count += 1

        graph = {
            'nodes': nodes,  # dictionary {id, name, pos, type}
            'edge_index': edges,  # pairwise edge list
            'cnt_node_idx': cnt_node_idx,  # indices of contianers
            'obj_node_idx': obj_node_idx,  # indices of objects
            'idx_map': assetId_idx_map  # mapping from assedId to graph index position
        }

        # Add edges to get a connected graph if not already connected
        req_edges = utils.get_edges_for_connected_graph(self.occupancy_grid, graph, pos='pos')
        graph['edge_index'] = req_edges + graph['edge_index']

        if not include_node_embeddings:
            return graph

        # perform some more formatting for the graph, then return
        node_coords = {}
        node_names = {}
        graph_nodes = []
        node_color_list = []

        for count, node_key in enumerate(graph['nodes']):
            node_coords[node_key] = graph['nodes'][node_key]['pos']
            node_names[node_key] = graph['nodes'][node_key]['name']
            node_feature = np.concatenate((
                utils.get_sentence_embedding(graph['nodes'][node_key]['name']),
                graph['nodes'][node_key]['type']
            ))
            assert count == node_key
            graph_nodes.append(node_feature)
            node_color_list.append(utils.get_object_color_from_type(
                graph['nodes'][node_key]['type']))

        graph['node_coords'] = node_coords
        graph['node_names'] = node_names
        graph['graph_nodes'] = graph_nodes  # node features
        src = []
        dst = []
        for edge in graph['edge_index']:
            src.append(edge[0])
            dst.append(edge[1])
        graph['graph_edge_index'] = [src, dst]

        graph['graph_image'] = utils.get_graph_image(
            graph['edge_index'],
            node_names, node_color_list
        )

        return graph

    def get_known_costs(self):
        known_cost = {'initial_robot_pose': {}}
        init_r = self.get_robot_pose()
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

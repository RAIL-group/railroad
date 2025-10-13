import itertools
import numpy as np
from procthor import ThorInterface
from . import utils
from .environments import BaseEnvironment


class ProcTHOREnvironment(BaseEnvironment):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.thor_interface = ThorInterface(self.args)
        self.known_graph, self.grid, self.robot_pose, self.target_object_info = self.thor_interface.gen_map_and_poses()
        self.locations = self._get_location_to_coordinates_dict()
        self.all_objects = self._get_all_objects()
        self.target_object = f'{self.target_object_info['name']}_{self.target_object_info['idxs'][0]}'

        self.partial_graph = self.known_graph.get_object_free_graph()
        self.move_cost = self.get_move_cost_fn()

    def _get_all_objects(self):
        all_objects = set()
        for container_idx in self.known_graph.container_indices:
            object_idxs = self.known_graph.get_adjacent_nodes_idx(container_idx, filter_by_type=3)
            all_objects.update({f'{self.known_graph.get_node_name_by_idx(obj_idx)}_{obj_idx}' for obj_idx in object_idxs})
        return all_objects

    def get_objects_at_location(self, location):
        if location not in self.locations or location == 'start':
            raise ValueError(f"Location {location} is not valid.")
        location_node_idx = int(location.split('_')[1])

        object_idxs = self.known_graph.get_adjacent_nodes_idx(location_node_idx, filter_by_type=3)

        object_names = []
        for obj_idx in object_idxs:
            obj_node = self.known_graph.nodes[obj_idx].copy()
            obj_name = f"{obj_node['name']}_{obj_idx}"
            obj_node['object_name'] = obj_name
            object_names.append(obj_name)
            new_obj_idx = self.partial_graph.add_node(obj_node)
            self.partial_graph.add_edge(location_node_idx, new_obj_idx)

        return {'object': object_names}

    def _get_location_to_coordinates_dict(self):
        loc_to_coords = {'start': (self.robot_pose.x, self.robot_pose.y, self.robot_pose.yaw)}
        for container_idx in self.known_graph.container_indices:
            loc_name = self.known_graph.get_node_name_by_idx(container_idx)
            coords = self.known_graph.get_node_position_by_idx(container_idx)
            coords = coords if len(coords) == 3 else (coords[0], coords[1], 0)
            loc_to_coords[f"{loc_name}_{container_idx}"] = coords
        return loc_to_coords

    def get_move_cost_fn(self):
        inter_container_distances = {}
        for cnt1_idx, cnt2_idx in itertools.combinations(self.known_graph.container_indices, 2):
            loc1 = f"{self.known_graph.get_node_name_by_idx(cnt1_idx)}_{cnt1_idx}"
            loc2 = f"{self.known_graph.get_node_name_by_idx(cnt2_idx)}_{cnt2_idx}"
            cnt1_id = self.known_graph.nodes[cnt1_idx]['id']
            cnt2_id = self.known_graph.nodes[cnt2_idx]['id']
            inter_container_distances[frozenset([loc1, loc2])] = self.thor_interface.known_cost[cnt1_id][cnt2_id]

        def move_cost_fn(robot, loc_from, loc_to):
            if frozenset([loc_from, loc_to]) in inter_container_distances:
                return inter_container_distances[frozenset([loc_from, loc_to])]
            loc_from_coords = self.locations[loc_from]
            loc_to_coords = self.locations[loc_to]
            cost = utils.get_cost_between_two_coords(self.grid, loc_from_coords, loc_to_coords)
            return cost
        return move_cost_fn

    def get_intermediate_coordinates(self, time, loc_from, loc_to):
        loc_from_coords = self.locations[loc_from]
        loc_to_coords = self.locations[loc_to]
        _, path = utils.get_cost_between_two_coords(self.grid, loc_from_coords, loc_to_coords, return_path=True)
        return utils.get_coordinates_at_time(path, time)

    def remove_object_from_location(self, obj, location):
        location_node_idx = int(location.split('_')[1])
        objects_at_location_idxs = self.partial_graph.get_adjacent_nodes_idx(location_node_idx, filter_by_type=3)
        for obj_idx in objects_at_location_idxs:
            obj_name = self.partial_graph.nodes[obj_idx]['object_name']
            if obj_name == obj:
                self.partial_graph.delete_node(obj_idx)
                break

    def add_object_at_location(self, obj, location):
        location_node_idx = int(location.split('_')[1])
        obj_idx = int(obj.split('_')[1])
        obj_node = self.known_graph.nodes[obj_idx].copy()
        obj_node['object_name'] = obj
        new_obj_idx = self.partial_graph.add_node(obj_node)
        self.partial_graph.add_edge(location_node_idx, new_obj_idx)

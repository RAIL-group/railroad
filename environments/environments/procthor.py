import itertools
import numpy as np
from procthor import ThorInterface, utils

# World
objects_at_locations = {
    "start": dict(),
    "roomA": {"object": {"objA", "objC"}},
    "roomB": {"object": {"objB"}},
    "roomC": {"object": {"objD"}},
}

class ProcTHOREnvironment():
    def __init__(self, args):
        self.args = args
        self.thor_interface = ThorInterface(self.args)
        self.graph, self.grid, self.robot_pose, self.target_object_info = self.thor_interface.gen_map_and_poses()
        self.objects_at_locations = self._get_objects_at_locations(self.graph)
        self.target_object = f'{self.target_object_info['name']}_{self.target_object_info['idxs'][0]}'
        self.locations = self._get_location_to_coordinates()
        self.move_time = self.get_move_cost_fn()


    def _get_objects_at_locations(self, graph):
        self.all_objects = set()
        objects_at_locations = {'start': {}}
        for container_idx in graph.container_indices:
            object_idxs = graph.get_adjacent_nodes_idx(container_idx, filter_by_type=3)
            container_name = graph.get_node_name_by_idx(container_idx)
            container_id = f'{container_name}_{container_idx}'
            object_names = [graph.get_node_name_by_idx(obj_idx) for obj_idx in object_idxs]
            objects_id = {f'{name}_{idx}' for idx, name in zip(object_idxs, object_names)}
            objects_at_locations[container_id] = {"object": objects_id}
            self.all_objects.update(objects_id)
        return objects_at_locations

    def _get_location_to_coordinates(self):
        loc_to_coords = dict()
        for loc in self.objects_at_locations.keys():
            if loc == 'start':
                loc_to_coords['start'] = (self.robot_pose.x, self.robot_pose.y, self.robot_pose.yaw)
            else:
                loc_idx = int(loc.split('_')[1])
                coords = self.graph.get_node_position_by_idx(loc_idx)
                coords = coords if len(coords) == 3 else (coords[0], coords[1], 0)
                loc_to_coords[loc] = coords
        # for i in range(self.args.num_robots):
        #     loc_to_coords[f'r{i+1}_loc'] = loc_to_coords['start']
        return loc_to_coords

    def get_move_cost_fn(self):
        inter_container_distances = {}
        for cnt1_idx, cnt2_idx in itertools.combinations(self.graph.container_indices, 2):
            loc1 = f"{self.graph.get_node_name_by_idx(cnt1_idx)}_{cnt1_idx}"
            loc2 = f"{self.graph.get_node_name_by_idx(cnt2_idx)}_{cnt2_idx}"
            cnt1_id = self.graph.nodes[cnt1_idx]['id']
            cnt2_id = self.graph.nodes[cnt2_idx]['id']
            inter_container_distances[frozenset([loc1, loc2])] = self.thor_interface.known_cost[cnt1_id][cnt2_id]

        def move_cost_fn(robot, loc_from, loc_to):
            if frozenset([loc_from, loc_to]) in inter_container_distances:
                return inter_container_distances[frozenset([loc_from, loc_to])]
            loc_from_coords = self.locations[loc_from]
            loc_to_coords = self.locations[loc_to]
            cost = utils.get_cost(self.grid, loc_from_coords, loc_to_coords)
            return cost
            # TODO: Use A* to compute distance from to
            return np.linalg.norm(np.array(loc_from_coords) - np.array(loc_to_coords))
        return move_cost_fn

from procthor import ThorInterface

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
        graph, grid, robot_pose, _ = self.thor_interface.gen_map_and_poses()
        self.objects_at_locations = self._get_objects_at_locations(graph)
        self.all_objects = {}

    def _get_objects_at_locations(self, graph):
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

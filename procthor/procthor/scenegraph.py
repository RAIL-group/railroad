import copy
from . import utils


class SceneGraph:
    def __init__(self):
        """Initialize an empty scene graph."""
        self.nodes = {}  # id -> node dictionary mapping
        self.edges = []  # list of (src, dst) tuples
        self.asset_id_to_node_idx_map = {}  # asset id -> node index mapping

    def add_node(self, node_dict, node_idx=None):
        """Add a new node to the graph."""
        node_idx = len(self.nodes) if node_idx is None else node_idx
        self.nodes[node_idx] = node_dict
        if 'id' in node_dict:
            self.asset_id_to_node_idx_map[node_dict['id']] = node_idx
        return node_idx

    def add_edge(self, src_idx, dst_idx):
        """Add an edge between two nodes by their indices."""
        if src_idx not in self.nodes or dst_idx not in self.nodes:
            raise ValueError('Invalid node indices')
        self.edges.append((src_idx, dst_idx))

    def delete_node(self, node_idx):
        """Delete a node from the graph."""
        if node_idx not in self.nodes:
            raise ValueError('Invalid node index')
        del self.nodes[node_idx]
        self.edges = [(src, dst) for src, dst in self.edges
                      if src != node_idx and dst != node_idx]

    def delete_edge(self, src_idx, dst_idx):
        """Delete an edge between two nodes by their indices."""
        if (src_idx, dst_idx) not in self.edges:
            raise ValueError('Invalid edge')
        self.edges.remove((src_idx, dst_idx))

    def get_node_indices_by_type(self, type_idx):
        """Get indices of all nodes of a given type."""
        return [idx for idx, node in self.nodes.items()
                if node['type'][type_idx] == 1]

    def get_node_indices_by_id(self, id):
        """Get indices of all nodes of a given id."""
        return [idx for idx, node in self.nodes.items()
                if node['id'] == id]

    def get_node_indices_by_name(self, name):
        """Get indices of all nodes of a given name."""
        return [idx for idx, node in self.nodes.items()
                if node['name'] == name]

    def check_if_node_exists_by_id(self, id):
        """Check if a node with a given id exists."""
        return any(node['id'] == id for node in self.nodes.values())

    def get_object_free_graph(self):
        """Get a copy of the graph with object nodes removed."""
        graph = self.copy()
        obj_idx = graph.object_indices
        for _, v in self.edges:
            if v in obj_idx:
                graph.delete_node(v)
        return graph

    def get_node_name_by_idx(self, node_idx):
        """Get name of a node by its index."""
        return self.nodes[node_idx]['name']

    def get_node_position_by_idx(self, node_idx):
        """Get position of a node by its index."""
        return self.nodes[node_idx]['position']

    def get_node_idx_by_position(self, position):
        """Get index of a node by its position."""
        for idx, node in self.nodes.items():
            if node['position'][0] == position[0] and node['position'][1] == position[1]:
                return idx
        return None

    def __len__(self):
        return len(self.nodes)

    def copy(self):
        """Create a deep copy of the scene graph."""
        graph_copy = SceneGraph()
        graph_copy.nodes = copy.deepcopy(self.nodes)
        graph_copy.edges = copy.deepcopy(self.edges)
        graph_copy.asset_id_to_node_idx_map = copy.deepcopy(self.asset_id_to_node_idx_map)
        return graph_copy

    @property
    def room_indices(self):
        """Get indices of all room nodes."""
        return self.get_node_indices_by_type(1)

    @property
    def container_indices(self):
        """Get indices of all container nodes."""
        return self.get_node_indices_by_type(2)

    @property
    def object_indices(self):
        """Get indices of all object nodes."""
        return self.get_node_indices_by_type(3)

    def get_adjacent_nodes_idx(self, node_idx, filter_by_type=None):
        """Get indices of all adjacent nodes of a given type."""
        adj_nodes_idx = set()
        for src, dst in self.edges:
            if src == node_idx:
                if filter_by_type is None or self.nodes[dst]['type'][filter_by_type] == 1:
                    adj_nodes_idx.add(dst)
            elif dst == node_idx:
                if filter_by_type is None or self.nodes[src]['type'][filter_by_type] == 1:
                    adj_nodes_idx.add(src)
        return list(adj_nodes_idx)

    def get_parent_node_idx(self, node_idx):
        """Get the index of the parent node of a given node."""
        node_type = self.nodes[node_idx]['type'].index(1)
        parent_nodes_idx = self.get_adjacent_nodes_idx(node_idx, filter_by_type=node_type - 1)
        if len(parent_nodes_idx) == 0:
            return None
        return parent_nodes_idx[0]  # Assuming only one parent node

    def ensure_connectivity(self, occupancy_grid):
        """Ensure the graph is connected by adding necessary edges."""
        required_edges = utils.get_edges_for_connected_graph(occupancy_grid, {
            'nodes': self.nodes,
            'edge_index': self.edges,
            'cnt_node_idx': self.container_indices,
            'obj_node_idx': self.object_indices,
            'idx_map': self.asset_id_to_node_idx_map

        }, pos='position')
        self.edges.extend(required_edges)

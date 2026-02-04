"""Scene graph data structure for ProcTHOR environments."""

import copy
from typing import Any, Dict, List, Optional, Tuple


class SceneGraph:
    """Hierarchical scene graph for indoor environments.

    Node types (one-hot encoded):
    - [1,0,0,0]: apartment
    - [0,1,0,0]: room
    - [0,0,1,0]: container
    - [0,0,0,1]: object
    """

    def __init__(self) -> None:
        """Initialize an empty scene graph."""
        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.edges: List[Tuple[int, int]] = []
        self.asset_id_to_node_idx_map: Dict[str, int] = {}

    def add_node(self, node_dict: Dict[str, Any], node_idx: Optional[int] = None) -> int:
        """Add a new node to the graph."""
        node_idx = len(self.nodes) if node_idx is None else node_idx
        while node_idx in self.nodes:
            node_idx += 1
        self.nodes[node_idx] = node_dict
        if 'id' in node_dict:
            self.asset_id_to_node_idx_map[node_dict['id']] = node_idx
        return node_idx

    def add_edge(self, src_idx: int, dst_idx: int) -> None:
        """Add an edge between two nodes."""
        if src_idx not in self.nodes or dst_idx not in self.nodes:
            raise ValueError('Invalid node indices')
        self.edges.append((src_idx, dst_idx))

    def delete_node(self, node_idx: int) -> None:
        """Delete a node and its edges from the graph."""
        if node_idx not in self.nodes:
            raise ValueError('Invalid node index')
        del self.nodes[node_idx]
        self.edges = [(src, dst) for src, dst in self.edges
                      if src != node_idx and dst != node_idx]

    def delete_edge(self, src_idx: int, dst_idx: int) -> None:
        """Delete an edge between two nodes."""
        if (src_idx, dst_idx) not in self.edges:
            raise ValueError('Invalid edge')
        self.edges.remove((src_idx, dst_idx))

    def get_node_indices_by_type(self, type_idx: int) -> List[int]:
        """Get indices of all nodes of a given type."""
        return [idx for idx, node in self.nodes.items()
                if node['type'][type_idx] == 1]

    def get_node_indices_by_id(self, node_id: str) -> List[int]:
        """Get indices of all nodes with a given id."""
        return [idx for idx, node in self.nodes.items()
                if node.get('id') == node_id]

    def get_node_indices_by_name(self, name: str) -> List[int]:
        """Get indices of all nodes with a given name."""
        return [idx for idx, node in self.nodes.items()
                if node.get('name') == name]

    def check_if_node_exists_by_id(self, node_id: str) -> bool:
        """Check if a node with a given id exists."""
        return any(node.get('id') == node_id for node in self.nodes.values())

    def get_object_free_graph(self) -> "SceneGraph":
        """Get a copy of the graph with object nodes removed."""
        graph = self.copy()
        obj_indices = graph.object_indices
        for _, dst in list(self.edges):
            if dst in obj_indices:
                graph.delete_node(dst)
        return graph

    def get_node_name_by_idx(self, node_idx: int) -> str:
        """Get name of a node by its index."""
        return self.nodes[node_idx]['name']

    def get_node_position_by_idx(self, node_idx: int) -> Any:
        """Get position of a node by its index."""
        return self.nodes[node_idx]['position']

    def get_node_idx_by_position(self, position: Tuple[float, float]) -> Optional[int]:
        """Get index of a node by its position."""
        for idx, node in self.nodes.items():
            if node['position'][0] == position[0] and node['position'][1] == position[1]:
                return idx
        return None

    def __len__(self) -> int:
        return len(self.nodes)

    def copy(self) -> "SceneGraph":
        """Create a deep copy of the scene graph."""
        graph_copy = SceneGraph()
        graph_copy.nodes = copy.deepcopy(self.nodes)
        graph_copy.edges = copy.deepcopy(self.edges)
        graph_copy.asset_id_to_node_idx_map = copy.deepcopy(self.asset_id_to_node_idx_map)
        return graph_copy

    @property
    def room_indices(self) -> List[int]:
        """Get indices of all room nodes."""
        return self.get_node_indices_by_type(1)

    @property
    def container_indices(self) -> List[int]:
        """Get indices of all container nodes."""
        return self.get_node_indices_by_type(2)

    @property
    def object_indices(self) -> List[int]:
        """Get indices of all object nodes."""
        return self.get_node_indices_by_type(3)

    def get_adjacent_nodes_idx(
        self,
        node_idx: int,
        filter_by_type: Optional[int] = None
    ) -> List[int]:
        """Get indices of all adjacent nodes, optionally filtered by type."""
        adj_nodes_idx = set()
        for src, dst in self.edges:
            if src == node_idx:
                if filter_by_type is None or self.nodes[dst]['type'][filter_by_type] == 1:
                    adj_nodes_idx.add(dst)
            elif dst == node_idx:
                if filter_by_type is None or self.nodes[src]['type'][filter_by_type] == 1:
                    adj_nodes_idx.add(src)
        return list(adj_nodes_idx)

    def get_parent_node_idx(self, node_idx: int) -> Optional[int]:
        """Get the index of the parent node (one level up in hierarchy)."""
        node_type = self.nodes[node_idx]['type'].index(1)
        if node_type == 0:
            return None  # Apartment has no parent
        parent_nodes_idx = self.get_adjacent_nodes_idx(node_idx, filter_by_type=node_type - 1)
        return parent_nodes_idx[0] if parent_nodes_idx else None

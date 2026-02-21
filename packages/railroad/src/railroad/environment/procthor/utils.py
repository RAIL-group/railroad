"""Utility functions for ProcTHOR environments."""

from typing import Any, Dict, List, Sequence, Tuple, cast

import networkx as nx
import numpy as np

from railroad.navigation import pathing


def get_nearest_free_point(
    point: Dict[str, float],
    free_points: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Find the nearest free point to a given position."""
    min_dist = float("inf")
    nearest = free_points[0]
    for fp in free_points:
        dist = (fp[0] - point["x"]) ** 2 + (fp[1] - point["z"]) ** 2
        if dist < min_dist:
            min_dist = dist
            nearest = fp
    return nearest


def has_edge(doors: List[Dict], room_0: str, room_1: str) -> bool:
    """Check if two rooms are connected by a door."""
    for door in doors:
        if (
            (door["room0"] == room_0 and door["room1"] == room_1)
            or (door["room1"] == room_0 and door["room0"] == room_1)
        ):
            return True
    return False


def get_generic_name(name: str) -> str:
    """Extract generic name from asset ID (e.g., 'table|1|2' -> 'table')."""
    return name.split("|")[0].lower()


def get_room_id(name: str) -> int:
    """Extract room ID from asset ID (e.g., 'table|1|2' -> 1)."""
    return int(name.split("|")[1])


def _path_cost(
    grid: np.ndarray,
    start: Sequence[float],
    end: Sequence[float],
) -> float:
    start_rc = (int(start[0]), int(start[1]))
    end_rc = (int(end[0]), int(end[1]))
    cost, _ = pathing.get_cost_and_path(
        grid,
        start_rc,
        end_rc,
        use_soft_cost=True,
        unknown_as_obstacle=False,
        soft_cost_scale=12.0,
    )
    return float(cost)


def get_edges_for_connected_graph(
    grid: np.ndarray,
    graph: Dict[str, Any],
    pos: str = "position",
) -> List[Tuple[int, int]]:
    """Find edges needed to make graph connected."""
    edges_to_add = []

    # Find room node indices (between apartment and first container)
    room_node_idx = list(range(1, graph["cnt_node_idx"][0]))

    # Extract room-only edges
    filtered_edges = [
        edge for edge in graph["edge_index"] if edge[1] in room_node_idx and edge[0] != 0
    ]

    sorted_dc = _get_disconnected_components(room_node_idx, filtered_edges)

    while len(sorted_dc) > 1:
        comps = sorted_dc[0]
        merged_set = set()
        for s in sorted_dc[1:]:
            merged_set |= s

        min_cost = float("inf")
        min_edge = None

        for comp in comps:
            for target in merged_set:
                cost = _path_cost(
                    grid,
                    graph["nodes"][comp][pos],
                    graph["nodes"][target][pos],
                )
                if cost < min_cost:
                    min_cost = cost
                    min_edge = (comp, target)

        if min_edge:
            edges_to_add.append(min_edge)
            filtered_edges.append(min_edge)
            sorted_dc = _get_disconnected_components(room_node_idx, filtered_edges)

    return edges_to_add


def _get_disconnected_components(
    node_indices: List[int],
    edges: List[Tuple[int, int]],
) -> List[set[int]]:
    """Get disconnected components sorted by size."""
    graph = nx.Graph()
    graph.add_nodes_from(node_indices)
    graph.add_edges_from(edges)
    components = list(nx.connected_components(graph))
    return cast(List[set[int]], sorted(components, key=len))

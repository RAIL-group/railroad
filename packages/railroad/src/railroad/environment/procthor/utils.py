"""Utility functions for ProcTHOR environments."""

from typing import Any, Dict, List, Tuple, cast

import networkx as nx
import numpy as np


def get_nearest_free_point(
    point: Dict[str, float],
    free_points: List[Tuple[float, float]]
) -> Tuple[float, float]:
    """Find the nearest free point to a given position.

    Args:
        point: Dict with 'x' and 'z' keys
        free_points: List of (x, z) tuples

    Returns:
        The closest free point as (x, z) tuple
    """
    min_dist = float('inf')
    nearest = free_points[0]
    for fp in free_points:
        dist = (fp[0] - point['x'])**2 + (fp[1] - point['z'])**2
        if dist < min_dist:
            min_dist = dist
            nearest = fp
    return nearest


def has_edge(doors: List[Dict], room_0: str, room_1: str) -> bool:
    """Check if two rooms are connected by a door."""
    for door in doors:
        if ((door['room0'] == room_0 and door['room1'] == room_1) or
            (door['room1'] == room_0 and door['room0'] == room_1)):
            return True
    return False


def get_generic_name(name: str) -> str:
    """Extract generic name from asset ID (e.g., 'table|1|2' -> 'table')."""
    return name.split('|')[0].lower()


def get_room_id(name: str) -> int:
    """Extract room ID from asset ID (e.g., 'table|1|2' -> 1)."""
    return int(name.split('|')[1])


def get_cost(grid: np.ndarray, robot_pose: Tuple[int, int], end: Tuple[int, int]) -> float:
    """Compute path cost between two grid positions.

    Args:
        grid: Occupancy grid (0=free, 1=occupied)
        robot_pose: Start position (x, y)
        end: End position (x, y)

    Returns:
        Path cost (distance)
    """
    import gridmap

    occ_grid = np.copy(grid)
    occ_grid[int(robot_pose[0])][int(robot_pose[1])] = 0
    occ_grid[end[0], end[1]] = 0

    cost_grid = gridmap.planning.compute_cost_grid_from_position(
        occ_grid,
        start=[robot_pose[0], robot_pose[1]],
        use_soft_cost=True,
        only_return_cost_grid=True
    )
    return cost_grid[end[0], end[1]]


def get_cost_and_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int]
) -> Tuple[float, np.ndarray]:
    """Compute path cost and path between two grid positions.

    Args:
        grid: Occupancy grid
        start: Start position (x, y)
        end: End position (x, y)

    Returns:
        Tuple of (cost, path) where path is 2xN array
    """
    import gridmap

    occ_grid = np.copy(grid)
    occ_grid[int(start[0])][int(start[1])] = 0
    occ_grid[end[0], end[1]] = 0

    cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
        occ_grid,
        start=[start[0], start[1]],
        use_soft_cost=True
    )
    cost = cost_grid[end[0], end[1]]
    _, path = get_path(target=[end[0], end[1]])
    return cost, path


def get_coordinates_at_time(path: np.ndarray, time: float) -> np.ndarray:
    """Get coordinates along a path at a given time (distance).

    Args:
        path: 2xN array of path coordinates
        time: Distance along path

    Returns:
        Coordinates at that time
    """
    diffs = np.diff(path, axis=1)
    segment_lengths = np.linalg.norm(diffs, axis=0)
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
    idx = np.searchsorted(cumulative_lengths, time, side='left')
    idx = min(idx, path.shape[1] - 1)
    return path[:, idx]


def get_edges_for_connected_graph(
    grid: np.ndarray,
    graph: Dict[str, Any],
    pos: str = 'position'
) -> List[Tuple[int, int]]:
    """Find edges needed to make graph connected.

    Args:
        grid: Occupancy grid for cost computation
        graph: Dict with 'nodes', 'edge_index', 'cnt_node_idx' keys
        pos: Key for position in node dict

    Returns:
        List of edges to add
    """
    edges_to_add = []

    # Find room node indices (between apartment and first container)
    room_node_idx = list(range(1, graph['cnt_node_idx'][0]))

    # Extract room-only edges
    filtered_edges = [
        edge for edge in graph['edge_index']
        if edge[1] in room_node_idx and edge[0] != 0
    ]

    sorted_dc = _get_disconnected_components(room_node_idx, filtered_edges)

    while len(sorted_dc) > 1:
        comps = sorted_dc[0]
        merged_set = set()
        for s in sorted_dc[1:]:
            merged_set |= s

        min_cost = float('inf')
        min_edge = None

        for comp in comps:
            for target in merged_set:
                cost = get_cost(
                    grid,
                    graph['nodes'][comp][pos],
                    graph['nodes'][target][pos]
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
    edges: List[Tuple[int, int]]
) -> List[set[int]]:
    """Get disconnected components sorted by size."""
    G: nx.Graph[int] = nx.Graph()
    G.add_nodes_from(node_indices)
    G.add_edges_from(edges)
    components: List[set[int]] = list(nx.connected_components(G))
    return cast(List[set[int]], sorted(components, key=len))

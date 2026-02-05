"""Utility functions for ProcTHOR environments."""

import math
from typing import Any, Callable, Dict, List, Tuple, Union, cast

import networkx as nx
import numpy as np
import scipy.ndimage
import skimage.graph

# Grid constants
COLLISION_VAL = 1
FREE_VAL = 0
OBSTACLE_THRESHOLD = 0.5 * (FREE_VAL + COLLISION_VAL)


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


def inflate_grid(
    grid: np.ndarray,
    inflation_radius: float,
    obstacle_threshold: float = OBSTACLE_THRESHOLD,
    collision_val: float = COLLISION_VAL,
) -> np.ndarray:
    """Inflate obstacles in an occupancy grid.

    Args:
        grid: Occupancy grid
        inflation_radius: Radius (in grid units) to inflate obstacles
        obstacle_threshold: Value above which a cell is an obstacle
        collision_val: Value obstacles are given after inflation

    Returns:
        Grid with inflated obstacles
    """
    obstacle_grid = np.zeros(grid.shape)
    obstacle_grid[grid >= obstacle_threshold] = 1

    kernel_size = int(1 + 2 * math.ceil(inflation_radius))
    cind = int(math.ceil(inflation_radius))
    y, x = np.ogrid[-cind : kernel_size - cind, -cind : kernel_size - cind]
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[y * y + x * x <= inflation_radius * inflation_radius] = 1
    inflated_mask = scipy.ndimage.convolve(
        obstacle_grid, kernel, mode="constant", cval=0
    )
    inflated_mask = inflated_mask >= 1.0
    inflated_grid = grid.copy()
    inflated_grid[inflated_mask] = collision_val

    return inflated_grid


def compute_cost_grid_from_position(
    occupancy_grid: np.ndarray,
    start: Union[List[float], np.ndarray],
    use_soft_cost: bool = False,
    obstacle_cost: float = -1,
    ends: Union[List[Tuple[int, int]], None] = None,
    only_return_cost_grid: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Callable]]:
    """Get the cost grid and planning function for a grid/start position.

    Uses Dijkstra's algorithm via skimage.graph.MCP_Geometric.

    Args:
        occupancy_grid: Input grid over which cost is computed
        start: Location of the start position for cost computation
        use_soft_cost: Whether to use soft inflation costs
        obstacle_cost: Cost assigned to obstacles
        ends: Optional list of end positions
        only_return_cost_grid: If True, only return the cost grid

    Returns:
        If only_return_cost_grid is True:
            cost_grid: The cost of traveling from start to other positions
        Otherwise:
            Tuple of (cost_grid, get_path function)
    """
    if len(np.array(start).shape) > 1:
        starts = start.T
    else:
        starts = [start]

    if use_soft_cost:
        scale_factor = 50
        input_cost_grid = np.ones(occupancy_grid.shape) * scale_factor
        g1 = inflate_grid(occupancy_grid, 1.5)
        g2 = inflate_grid(g1, 1.0)
        g3 = inflate_grid(g2, 1.5)
        soft_cost_grid = 8 * g1 + 5 * g2 + g3
        input_cost_grid += soft_cost_grid
    else:
        scale_factor = 1
        input_cost_grid = np.ones(occupancy_grid.shape)

    input_cost_grid[occupancy_grid >= OBSTACLE_THRESHOLD] = obstacle_cost

    mcp = skimage.graph.MCP_Geometric(input_cost_grid)
    if ends is None:
        cost_grid = mcp.find_costs(starts=starts)[0] / (1.0 * scale_factor)
    else:
        cost_grid = mcp.find_costs(starts=starts, ends=ends)[0] / (1.0 * scale_factor)

    if only_return_cost_grid:
        return cost_grid

    def get_path(target: Tuple[int, int]) -> Tuple[bool, np.ndarray]:
        try:
            path_list = mcp.traceback(target)
        except ValueError:
            return False, np.array([[]])
        path = np.zeros((2, len(path_list)))
        for ii in range(len(path_list)):
            path[:, ii] = path_list[ii]
        return True, path.astype(int)

    return cost_grid, get_path


def get_cost(grid: np.ndarray, robot_pose: Tuple[int, int], end: Tuple[int, int]) -> float:
    """Compute path cost between two grid positions.

    Args:
        grid: Occupancy grid (0=free, 1=occupied)
        robot_pose: Start position (x, y)
        end: End position (x, y)

    Returns:
        Path cost (distance)
    """
    occ_grid = np.copy(grid)
    occ_grid[int(robot_pose[0])][int(robot_pose[1])] = 0
    occ_grid[end[0], end[1]] = 0

    cost_grid = compute_cost_grid_from_position(
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
    occ_grid = np.copy(grid)
    occ_grid[int(start[0])][int(start[1])] = 0
    occ_grid[end[0], end[1]] = 0

    cost_grid, get_path = compute_cost_grid_from_position(
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


def get_trajectory(
    grid: np.ndarray,
    waypoints: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Compute obstacle-respecting trajectory through waypoints.

    Given a list of waypoints (e.g., location coordinates), computes the full
    path that avoids obstacles using the occupancy grid.

    Args:
        grid: Occupancy grid (0=free, 1=occupied)
        waypoints: List of (x, y) waypoint coordinates to visit in order

    Returns:
        List of (x, y) coordinates forming the complete trajectory
    """
    if len(waypoints) < 2:
        return list(waypoints)

    trajectory: List[Tuple[float, float]] = []

    for i in range(len(waypoints) - 1):
        start = (int(waypoints[i][0]), int(waypoints[i][1]))
        end = (int(waypoints[i + 1][0]), int(waypoints[i + 1][1]))

        _, path = get_cost_and_path(grid, start, end)

        if path.size == 0:
            # Planning failed, fall back to straight line
            trajectory.append(waypoints[i])
            continue

        # Add path points (skip first point for subsequent segments to avoid duplicates)
        start_idx = 0 if i == 0 else 1
        for j in range(start_idx, path.shape[1]):
            trajectory.append((float(path[0, j]), float(path[1, j])))

    return trajectory


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

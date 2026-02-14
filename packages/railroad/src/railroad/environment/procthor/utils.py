"""Utility functions for ProcTHOR environments."""

import heapq
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
THETA_STAR_HEURISTIC_WEIGHT = 1.5


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


def build_traversal_costs(
    occupancy_grid: np.ndarray,
    use_soft_cost: bool = True,
) -> np.ndarray:
    """Build a per-cell traversal cost array from an occupancy grid.

    Free cells get a base cost of 1.0, optionally augmented by soft inflation
    costs near obstacles. Obstacle cells are set to infinity.

    Args:
        occupancy_grid: Binary occupancy grid (0=free, 1=occupied).
        use_soft_cost: Whether to add soft inflation costs near obstacles.

    Returns:
        Array of traversal costs (same shape as input). Obstacle cells are inf.
    """
    costs = np.ones(occupancy_grid.shape)
    if use_soft_cost:
        g1 = inflate_grid(occupancy_grid, 1.5)
        g2 = inflate_grid(g1, 1.0)
        g3 = inflate_grid(g2, 1.5)
        soft = 8 * g1 + 5 * g2 + g3
        costs += soft / 50.0
    costs[occupancy_grid >= OBSTACLE_THRESHOLD] = np.inf
    return costs


def _supercover_line(
    x0: int, y0: int, x1: int, y1: int
) -> list[tuple[int, int]]:
    """Return all grid cells touched by the line segment (x0,y0)->(x1,y1).

    Uses an Amanatides-Woo supercover variant of Bresenham's algorithm.
    Unlike standard Bresenham, this emits both axis-aligned neighbors on
    diagonal steps, ensuring no corner-touching cells are missed.
    """
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0

    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy and e2 < dx:
            # Diagonal step: emit both axis-aligned neighbors first
            cells.append((x + sx, y))
            cells.append((x, y + sy))
            err -= dy
            err += dx
            x += sx
            y += sy
        elif e2 > -dy:
            err -= dy
            x += sx
        else:
            err += dx
            y += sy

    return cells


def _line_of_sight(
    costs: np.ndarray, a: tuple[int, int], b: tuple[int, int]
) -> bool:
    """Check whether a straight line between two cells is obstacle-free.

    Args:
        costs: Traversal cost grid (inf marks obstacles).
        a: Start cell (row, col).
        b: End cell (row, col).

    Returns:
        True if every cell along the supercover line is in-bounds and not
        an obstacle (cost != inf).
    """
    rows, cols = costs.shape
    for r, c in _supercover_line(a[0], a[1], b[0], b[1]):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if np.isinf(costs[r, c]):
            return False
    return True


def _theta_star(
    costs: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    heuristic_grid: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Theta* any-angle path planning on a weighted grid.

    Args:
        costs: Per-cell traversal cost grid (inf = obstacle).
        start: Start cell (row, col).
        goal: Goal cell (row, col).
        heuristic_grid: Optional precomputed cost-to-go from goal. If None,
            Euclidean distance is used.

    Returns:
        (cost, path) where path is a 2xN int array. If unreachable, returns
        (inf, array([[]])).
    """
    if start == goal:
        return 0.0, np.array([[start[0]], [start[1]]], dtype=int)

    rows, cols = costs.shape

    def heuristic(n: tuple[int, int]) -> float:
        if heuristic_grid is not None:
            return float(heuristic_grid[n[0], n[1]])
        return math.hypot(n[0] - goal[0], n[1] - goal[1])

    def _edge_cost(a: tuple[int, int], b: tuple[int, int]) -> float:
        line = _supercover_line(a[0], a[1], b[0], b[1])
        dist = math.hypot(a[0] - b[0], a[1] - b[1])
        total = sum(costs[r, c] for r, c in line)
        return dist * total / len(line)

    # 8-connected neighbor offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {start: start}
    _HeapItem = tuple[float, int, tuple[int, int]]

    def _push(item: _HeapItem) -> None:
        heapq.heappush(open_heap, item)

    counter: int = 0
    open_heap: list[_HeapItem] = []
    _push((THETA_STAR_HEURISTIC_WEIGHT * heuristic(start), counter, start))
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, _, s = heapq.heappop(open_heap)
        if s == goal:
            # Reconstruct path
            path_cells: list[tuple[int, int]] = []
            node = goal
            while node != start:
                path_cells.append(node)
                node = parent[node]
            path_cells.append(start)
            path_cells.reverse()
            path = np.array(path_cells, dtype=int).T  # 2xN
            return g_score[goal], path

        if s in closed:
            continue
        closed.add(s)

        for dr, dc in neighbors:
            nr, nc = s[0] + dr, s[1] + dc
            n = (nr, nc)
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if np.isinf(costs[nr, nc]):
                continue
            if n in closed:
                continue

            # Theta* rewiring: try parent-of-s -> n
            ps = parent[s]
            if _line_of_sight(costs, ps, n):
                new_g = g_score[ps] + _edge_cost(ps, n)
                if new_g < g_score.get(n, float('inf')):
                    g_score[n] = new_g
                    parent[n] = ps
                    counter += 1
                    _push((new_g + THETA_STAR_HEURISTIC_WEIGHT * heuristic(n), counter, n))
            else:
                new_g = g_score[s] + _edge_cost(s, n)
                if new_g < g_score.get(n, float('inf')):
                    g_score[n] = new_g
                    parent[n] = s
                    counter += 1
                    _push((new_g + THETA_STAR_HEURISTIC_WEIGHT * heuristic(n), counter, n))

    return float('inf'), np.array([[]])


def get_cost_and_path_theta(
    grid: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    use_soft_cost: bool = True,
) -> Tuple[float, np.ndarray]:
    """Compute cost and any-angle path between two grid positions using Theta*.

    Drop-in replacement for ``get_cost_and_path`` that produces smoother
    paths by allowing line-of-sight shortcuts between non-adjacent cells.

    Args:
        grid: Occupancy grid (0=free, 1=occupied).
        start: Start position (row, col).
        end: End position (row, col).
        use_soft_cost: Whether to apply soft inflation costs near obstacles.

    Returns:
        (cost, path) where path is a 2xN int array.
    """
    occ_grid = np.copy(grid)
    occ_grid[int(start[0])][int(start[1])] = 0
    occ_grid[end[0], end[1]] = 0

    trav_costs = build_traversal_costs(occ_grid, use_soft_cost=use_soft_cost)

    heuristic_grid = cast(
        np.ndarray,
        compute_cost_grid_from_position(
            occ_grid,
            start=[end[0], end[1]],
            use_soft_cost=use_soft_cost,
            only_return_cost_grid=True,
        ),
    )

    return _theta_star(trav_costs, start, end, heuristic_grid=heuristic_grid)


# Re-export from canonical location for backward compatibility.
from railroad.environment.navigation.pathing import compute_cost_grid_from_position as compute_cost_grid_from_position  # noqa: F401


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

    cost_grid = cast(
        np.ndarray,
        compute_cost_grid_from_position(
            occ_grid,
            start=[robot_pose[0], robot_pose[1]],
            use_soft_cost=True,
            only_return_cost_grid=True
        )
    )
    return cost_grid[end[0], end[1]]


def get_cost_and_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int]
) -> Tuple[float, np.ndarray]:
    """Compute path cost and path between two grid positions.

    Uses Theta* any-angle planning for smoother trajectories.

    Args:
        grid: Occupancy grid
        start: Start position (x, y)
        end: End position (x, y)

    Returns:
        Tuple of (cost, path) where path is 2xN array
    """
    return get_cost_and_path_theta(grid, start, end, use_soft_cost=True)


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

"""Grid path planning utilities (Theta*) for unknown-space navigation."""

from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import scipy.ndimage
import skimage.graph

from .constants import COLLISION_VAL, OBSTACLE_THRESHOLD, UNOBSERVED_VAL


def inflate_grid(
    grid: np.ndarray,
    inflation_radius: float,
    obstacle_threshold: float = OBSTACLE_THRESHOLD,
    collision_val: float = COLLISION_VAL,
) -> np.ndarray:
    """Inflate obstacles in an occupancy grid."""
    obstacle_grid = np.zeros(grid.shape)
    obstacle_grid[grid >= obstacle_threshold] = 1

    kernel_size = int(1 + 2 * math.ceil(inflation_radius))
    cind = int(math.ceil(inflation_radius))
    y, x = np.ogrid[-cind: kernel_size - cind, -cind: kernel_size - cind]
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[y * y + x * x <= inflation_radius * inflation_radius] = 1
    inflated_mask = scipy.ndimage.convolve(
        obstacle_grid,
        kernel,
        mode="constant",
        cval=0,
    )
    inflated_mask = inflated_mask >= 1.0
    out = grid.copy()
    out[inflated_mask] = collision_val
    return out


def build_traversal_costs(
    occupancy_grid: np.ndarray,
    use_soft_cost: bool = True,
    unknown_as_obstacle: bool = True,
) -> np.ndarray:
    """Build per-cell traversal costs from an occupancy grid."""
    costs = np.ones(occupancy_grid.shape)
    if use_soft_cost:
        g1 = inflate_grid(occupancy_grid, 1.5)
        g2 = inflate_grid(g1, 1.0)
        g3 = inflate_grid(g2, 1.5)
        soft = 8 * g1 + 5 * g2 + g3
        costs += soft / 50.0
    obstacle_mask = occupancy_grid >= OBSTACLE_THRESHOLD
    if unknown_as_obstacle:
        obstacle_mask = np.logical_or(obstacle_mask, occupancy_grid == UNOBSERVED_VAL)
    costs[obstacle_mask] = np.inf
    return costs


def _supercover_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Return all cells touched by the line segment (x0,y0)->(x1,y1)."""
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


def _line_of_sight(costs: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Check whether straight line between two cells is obstacle free."""
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
) -> tuple[float, np.ndarray]:
    """Theta* any-angle path planning on a weighted grid."""
    if start == goal:
        return 0.0, np.array([[start[0]], [start[1]]], dtype=int)

    rows, cols = costs.shape

    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return float("inf"), np.array([[]])
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return float("inf"), np.array([[]])
    if np.isinf(costs[start[0], start[1]]) or np.isinf(costs[goal[0], goal[1]]):
        return float("inf"), np.array([[]])

    def heuristic(n: tuple[int, int]) -> float:
        return math.hypot(n[0] - goal[0], n[1] - goal[1])

    def edge_cost(a: tuple[int, int], b: tuple[int, int]) -> float:
        line = _supercover_line(a[0], a[1], b[0], b[1])
        dist = math.hypot(a[0] - b[0], a[1] - b[1])
        total = sum(costs[r, c] for r, c in line)
        return dist * total / max(1, len(line))

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    g_score: Dict[tuple[int, int], float] = {start: 0.0}
    parent: Dict[tuple[int, int], tuple[int, int]] = {start: start}

    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start), start))
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _f, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path_cells: list[tuple[int, int]] = []
            node = goal
            while node != start:
                path_cells.append(node)
                node = parent[node]
            path_cells.append(start)
            path_cells.reverse()
            path = np.array(path_cells, dtype=int).T
            return g_score[goal], path

        closed.add(current)

        for dr, dc in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            nbr = (nr, nc)
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if np.isinf(costs[nr, nc]):
                continue

            pcur = parent[current]
            if _line_of_sight(costs, pcur, nbr):
                tentative_parent = pcur
                tentative_g = g_score[pcur] + edge_cost(pcur, nbr)
            else:
                tentative_parent = current
                tentative_g = g_score[current] + edge_cost(current, nbr)

            if tentative_g < g_score.get(nbr, float("inf")):
                g_score[nbr] = tentative_g
                parent[nbr] = tentative_parent
                f = tentative_g + heuristic(nbr)
                heapq.heappush(open_heap, (f, nbr))

    return float("inf"), np.array([[]])


def get_cost_and_path_theta(
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    use_soft_cost: bool = True,
    unknown_as_obstacle: bool = True,
) -> tuple[float, np.ndarray]:
    """Compute cost and path with Theta* on occupancy grid."""
    costs = build_traversal_costs(
        grid,
        use_soft_cost=use_soft_cost,
        unknown_as_obstacle=unknown_as_obstacle,
    )

    if 0 <= start[0] < costs.shape[0] and 0 <= start[1] < costs.shape[1]:
        costs[start[0], start[1]] = 1.0
    if 0 <= end[0] < costs.shape[0] and 0 <= end[1] < costs.shape[1]:
        costs[end[0], end[1]] = 1.0

    return _theta_star(costs, start, end)


def get_cost_and_path(
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[float, np.ndarray]:
    """Compute cost and path between two grid positions."""
    return get_cost_and_path_theta(
        grid,
        start,
        end,
        use_soft_cost=True,
        unknown_as_obstacle=True,
    )


def path_total_length(path: np.ndarray) -> float:
    """Return total Euclidean length for a 2xN path."""
    if path.size == 0 or path.shape[1] < 2:
        return 0.0
    diffs = np.diff(path, axis=1)
    return float(np.sum(np.linalg.norm(diffs, axis=0)))


def get_coordinates_at_distance(path: np.ndarray, distance: float) -> np.ndarray:
    """Interpolate coordinates along a 2xN path at cumulative distance."""
    if path.size == 0:
        return np.array([np.nan, np.nan])
    if path.shape[1] == 1:
        return path[:, 0].astype(float)

    diffs = np.diff(path, axis=1)
    segment_lengths = np.linalg.norm(diffs, axis=0)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    if distance <= 0.0:
        return path[:, 0].astype(float)
    if distance >= cumulative_lengths[-1]:
        return path[:, -1].astype(float)

    idx = int(np.searchsorted(cumulative_lengths, distance, side="right") - 1)
    seg_len = segment_lengths[idx]
    if seg_len <= 1e-9:
        return path[:, idx].astype(float)

    t = (distance - cumulative_lengths[idx]) / seg_len
    start = path[:, idx].astype(float)
    end = path[:, idx + 1].astype(float)
    return start + t * (end - start)


def get_trajectory(
    grid: np.ndarray,
    waypoints: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Compute obstacle-respecting trajectory through waypoints."""
    if len(waypoints) < 2:
        return list(waypoints)

    trajectory: List[Tuple[float, float]] = []
    for i in range(len(waypoints) - 1):
        start = (int(waypoints[i][0]), int(waypoints[i][1]))
        end = (int(waypoints[i + 1][0]), int(waypoints[i + 1][1]))
        _cost, path = get_cost_and_path(grid, start, end)
        if path.size == 0:
            trajectory.append(waypoints[i])
            continue
        start_idx = 0 if i == 0 else 1
        for j in range(start_idx, path.shape[1]):
            trajectory.append((float(path[0, j]), float(path[1, j])))
    return trajectory


def compute_cost_grid_from_position(
    occupancy_grid: np.ndarray,
    start: Union[List[float], np.ndarray],
    use_soft_cost: bool = False,
    obstacle_cost: float = -1,
    unknown_as_obstacle: bool = True,
    ends: Union[List[Tuple[int, int]], None] = None,
    only_return_cost_grid: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Callable]]:
    """Compute a cost grid (distances from *start* to every cell) via Dijkstra.

    Uses ``skimage.graph.MCP_Geometric`` for efficient single-source
    shortest-path computation over the occupancy grid.

    Args:
        occupancy_grid: Input grid over which cost is computed.
        start: Source position ``(row, col)`` or array of positions.
        use_soft_cost: Whether to use soft inflation costs.
        obstacle_cost: Cost assigned to obstacles (use -1 for impassable).
        unknown_as_obstacle: If True, treat UNOBSERVED cells as impassable.
        ends: Optional list of end positions for early termination.
        only_return_cost_grid: If True, only return the cost grid.

    Returns:
        If *only_return_cost_grid* is True, the cost grid (ndarray).
        Otherwise ``(cost_grid, get_path)`` where *get_path(target)*
        returns ``(success, 2xN_path)``.
    """
    start_arr = np.array(start)
    if len(start_arr.shape) > 1:
        starts = start_arr.T
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

    obstacle_mask = occupancy_grid >= OBSTACLE_THRESHOLD
    if unknown_as_obstacle:
        obstacle_mask = np.logical_or(obstacle_mask, occupancy_grid == UNOBSERVED_VAL)
    input_cost_grid[obstacle_mask] = obstacle_cost

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

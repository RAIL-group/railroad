"""Plotting utilities for ProcTHOR environments."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import compute_move_path

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import networkx as nx
    from skimage.morphology import erosion
    HAS_PLOTTING_DEPS = True
except ImportError:
    HAS_PLOTTING_DEPS = False

COLLISION_VAL = 1
FREE_VAL = 0
FOOT_PRINT = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])


def make_plotting_grid(grid_map: np.ndarray) -> np.ndarray:
    """Convert occupancy grid to RGB plotting grid."""
    if not HAS_PLOTTING_DEPS:
        raise ImportError("Plotting requires scikit-image: pip install scikit-image")

    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3]) * 0.75
    collision = grid_map >= 0.5
    thinned = erosion(collision, footprint=FOOT_PRINT)
    boundary = np.logical_xor(collision, thinned)
    free = np.logical_and(grid_map < 0.5, grid_map >= FREE_VAL)

    grid[:, :, :][free] = 1
    grid[:, :, 0][boundary] = 0
    grid[:, :, 1][boundary] = 0
    grid[:, :, 2][boundary] = 0

    return grid


def plot_graph_on_grid(ax: Any, grid: np.ndarray, graph: Any) -> None:
    """Plot scene graph on occupancy grid."""
    plotting_grid = make_plotting_grid(grid.T)
    ax.imshow(plotting_grid)

    room_node_idx = graph.room_indices
    rc_idx = room_node_idx + graph.container_indices

    filtered_edges = [
        edge for edge in graph.edges
        if edge[1] in rc_idx and edge[0] != 0
    ]

    for (start, end) in filtered_edges:
        p1 = graph.nodes[start]['position']
        p2 = graph.nodes[end]['position']
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'c', linestyle="--", linewidth=0.3)

    for room in rc_idx:
        room_pos = graph.nodes[room]['position']
        room_name = graph.nodes[room]['name']
        ax.text(room_pos[0], room_pos[1], room_name, color='brown', size=6, rotation=40)


def plot_graph(
    ax: Any,
    nodes: Dict[int, Dict],
    edges: List[Tuple[int, int]],
    highlight_node: Optional[int] = None
) -> None:
    """Plot scene graph with nodes and edges."""
    if not HAS_PLOTTING_DEPS:
        raise ImportError("Plotting requires networkx")

    node_type_to_color = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
    G = nx.Graph()
    node_colors = []

    for k, v in nodes.items():
        G.add_node(k, label=f"{k}: {v['name']}")
        color = node_type_to_color.get(v['type'].index(1), 'violet')
        if k == highlight_node:
            color = 'cyan'
        node_colors.append(color)

    G.add_edges_from(edges)
    node_labels = nx.get_node_attributes(G, 'label')
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos, ax,
        with_labels=True,
        labels=node_labels,
        node_color=node_colors,
        node_size=20,
        font_size=4,
        edge_color='black',
        width=0.5
    )
    ax.axis('off')


def extract_robot_poses(
    actions: List[str],
    initial_locations: Dict[str, str],
    location_coords: Dict[str, Tuple[float, float]],
) -> Dict[str, List[Tuple[float, float]]]:
    """Extract robot waypoints from action sequence.

    Args:
        actions: List of action names (e.g., "move robot1 kitchen bedroom")
        initial_locations: Mapping from robot name to starting location name
        location_coords: Mapping from location name to (x, y) grid coordinates

    Returns:
        Mapping from robot name to list of (x, y) waypoints visited
    """
    robot_poses: Dict[str, List[Tuple[float, float]]] = {}

    # Initialize with start poses
    for robot_name, start_loc in initial_locations.items():
        if start_loc in location_coords:
            robot_poses[robot_name] = [location_coords[start_loc]]

    # Extract waypoints from move actions
    for action in actions:
        if not action.startswith("move"):
            continue

        parts = action.split()
        if len(parts) == 4:  # move robot from to
            robot_name = parts[1]
            to_loc = parts[3]

            if robot_name in robot_poses and to_loc in location_coords:
                robot_poses[robot_name].append(location_coords[to_loc])

    return robot_poses


def plot_grid(ax: Any, grid: np.ndarray) -> None:
    """Plot occupancy grid."""
    plotting_grid = make_plotting_grid(grid.T)
    ax.imshow(plotting_grid, origin="upper")


def plot_robot_trajectory(
    ax: Any,
    waypoints: List[Tuple[float, float]],
    grid: np.ndarray,
    graph: Any,
    robot_name: str,
    color_map: str = "viridis",
    robot_id: int = 0,
) -> None:
    """Plot a single robot's trajectory on the grid.

    Args:
        ax: Matplotlib axes
        waypoints: List of (x, y) grid coordinates (location waypoints)
        grid: Occupancy grid for path planning
        graph: SceneGraph for location name lookup
        robot_name: Name of the robot for labeling
        color_map: Matplotlib colormap name
        robot_id: Numeric ID for labeling waypoints
    """
    if not HAS_PLOTTING_DEPS:
        raise ImportError("Plotting requires matplotlib")

    if len(waypoints) < 2:
        return

    # Compute obstacle-respecting trajectory through waypoints
    trajectory: list[tuple[float, float]] = []
    for i in range(len(waypoints) - 1):
        start = (int(waypoints[i][0]), int(waypoints[i][1]))
        end = (int(waypoints[i + 1][0]), int(waypoints[i + 1][1]))
        path = compute_move_path(grid, start, end, use_theta=True)
        if path.size == 0:
            trajectory.append(waypoints[i])
            continue
        start_idx = 0 if i == 0 else 1
        for j in range(start_idx, path.shape[1]):
            trajectory.append((float(path[0, j]), float(path[1, j])))

    if len(trajectory) < 2:
        return

    # Draw trajectory lines with color gradient
    x = [p[0] for p in trajectory]
    y = [p[1] for p in trajectory]

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(0, len(x))
    lc = LineCollection(segments.tolist(), cmap=color_map, norm=norm)
    lc.set_array(np.arange(len(x)))
    lc.set_linewidth(2)
    ax.add_collection(lc)

    # Label start position
    ax.text(
        waypoints[0][0], waypoints[0][1],
        f"{robot_id} - {robot_name}",
        color="brown", size=4, weight="bold"
    )

    # Label intermediate waypoints with location names
    for i, (px, py) in enumerate(waypoints[1:], 1):
        idx = graph.get_node_idx_by_position([px, py])
        if idx is not None:
            name = graph.get_node_name_by_idx(idx)
            ax.text(px, py, f"{i} - {name}", color="brown", size=4, weight="bold")


def plot_multi_robot_trajectories(
    ax: Any,
    grid: np.ndarray,
    robots_data: Dict[str, List[Tuple[float, float]]],
    graph: Any,
) -> None:
    """Plot trajectories for multiple robots on the occupancy grid.

    Args:
        ax: Matplotlib axes
        grid: Occupancy grid array
        robots_data: Mapping from robot name to list of (x, y) waypoints
        graph: SceneGraph for location name lookup
    """
    plot_grid(ax, grid)

    colormaps = [
        "viridis", "plasma", "inferno", "magma", "cividis",
        "spring", "summer", "autumn", "winter", "cool"
    ]

    for i, (robot_name, waypoints) in enumerate(robots_data.items()):
        cmap = colormaps[i % len(colormaps)]
        plot_robot_trajectory(ax, waypoints, grid, graph, robot_name, cmap, robot_id=i)

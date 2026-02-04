"""Plotting utilities for ProcTHOR environments."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
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

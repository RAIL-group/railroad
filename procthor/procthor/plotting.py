import numpy as np
from skimage.morphology import erosion
import networkx as nx


COLLISION_VAL = 1
FREE_VAL = 0
UNOBSERVED_VAL = -1
assert (COLLISION_VAL > FREE_VAL)
assert (FREE_VAL > UNOBSERVED_VAL)
OBSTACLE_THRESHOLD = 0.5 * (FREE_VAL + COLLISION_VAL)


FOOT_PRINT = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])


def make_plotting_grid(grid_map):
    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3]) * 0.75
    collision = grid_map >= OBSTACLE_THRESHOLD
    # Take one pixel boundary of the region collision
    thinned = erosion(collision, footprint=FOOT_PRINT)
    boundary = np.logical_xor(collision, thinned)
    free = np.logical_and(grid_map < OBSTACLE_THRESHOLD, grid_map >= FREE_VAL)
    grid[:, :, 0][free] = 1
    grid[:, :, 1][free] = 1
    grid[:, :, 2][free] = 1
    grid[:, :, 0][boundary] = 0
    grid[:, :, 1][boundary] = 0
    grid[:, :, 2][boundary] = 0

    return grid


def make_blank_grid(grid_map):
    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3])
    return grid


def plot_graph_on_grid(ax, grid, graph):
    '''Plot the scene graph on the occupancy grid to scale'''
    plotting_grid = make_plotting_grid(grid.T)
    ax.imshow(plotting_grid)

    # find the room nodes
    room_node_idx = graph.room_indices

    rc_idx = room_node_idx + graph.container_indices

    # plot the edge connectivity between rooms and their containers only
    filtered_edges = [
        edge
        for edge in graph.edges
        if edge[1] in rc_idx and edge[0] != 0
    ]

    for (start, end) in filtered_edges:
        p1 = graph.nodes[start]['position']
        p2 = graph.nodes[end]['position']
        x_values = [p1[0], p2[0]]
        y_values = [p1[1], p2[1]]
        ax.plot(x_values, y_values, 'c', linestyle="--", linewidth=0.3)

    # plot room nodes
    for room in rc_idx:
        room_pos = graph.nodes[room]['position']
        room_name = graph.nodes[room]['name']
        ax.text(room_pos[0], room_pos[1], room_name, color='brown',
                size=6, rotation=40)


def plot_graph(ax, nodes, edges, highlight_node=None):
    '''Plot scene graph with nodes and edges'''
    node_type_to_color = {0: 'red',
                          1: 'blue',
                          2: 'green',
                          3: 'orange'}
    G = nx.Graph()
    node_colors = []
    for k, v in nodes.items():
        G.add_node(k, label=f"{k}: {v['name']}")
        color = node_type_to_color.get(v['type'].index(1), 'violet') if k != highlight_node else 'cyan'
        node_colors.append(color)
    G.add_edges_from(edges)
    node_labels = nx.get_node_attributes(G, 'label')
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax, with_labels=True, labels=node_labels, node_color=node_colors, node_size=20,
            font_size=4, font_weight='regular', edge_color='black', width=0.5)
    ax.axis('off')

import os
import io
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image

import gridmap


SBERT_PATH = '/resources/sentence_transformers/'


def get_nearest_free_point(point, free_points):
    _min = 1000000000
    tp = point
    fp_idx = 0
    for idx, rp in enumerate(free_points):
        dist = (rp[0] - tp['x'])**2 + (rp[1] - tp['z'])**2
        if dist < _min:
            _min = dist
            fp_idx = idx
    return free_points[fp_idx]


def has_edge(doors, room_0, room_1):
    for door in doors:
        if (door['room0'] == room_0 and door['room1'] == room_1) or \
           (door['room1'] == room_0 and door['room0'] == room_1):
            return True
    return False


def get_generic_name(name):
    return name.split('|')[0].lower()


def get_room_id(name):
    return int(name.split('|')[1])


def get_cost(grid, robot_pose, end):
    occ_grid = np.copy(grid)
    occ_grid[int(robot_pose[0])][int(robot_pose[1])] = 0

    occ_grid[end[0], end[1]] = 0

    cost_grid = gridmap.planning.compute_cost_grid_from_position(
        occ_grid,
        start=[
            robot_pose[0],
            robot_pose[1]
        ],
        use_soft_cost=True,
        only_return_cost_grid=True)

    cost = cost_grid[end[0], end[1]]
    return cost


def get_edges_for_connected_graph(grid, graph, pos='position'):
    """ This function finds edges that needs to exist to have a connected graph """
    edges_to_add = []

    # find the room nodes
    room_node_idx = [idx for idx in range(1, graph['cnt_node_idx'][0])]

    # extract the edges only for the rooms
    filtered_edges = [
        edge
        for edge in graph['edge_index']
        if edge[1] in room_node_idx and edge[0] != 0
    ]

    # Get a list (sorted by length) of disconnected components
    sorted_dc = get_dc_comps(room_node_idx, filtered_edges)

    length_of_dc = len(sorted_dc)
    while length_of_dc > 1:
        comps = sorted_dc[0]
        merged_set = set()
        min_cost = 9999
        min_index = -9999
        for s in sorted_dc[1:]:
            merged_set |= s
        for comp in comps:
            for idx, target in enumerate(merged_set):
                cost = get_cost(grid, graph['nodes'][comp][pos],
                                graph['nodes'][target][pos])
                if cost < min_cost:
                    min_cost = cost
                    min_index = list(merged_set)[idx]

        edge_to_add = (comp, min_index)
        edges_to_add.append(edge_to_add)
        filtered_edges = filtered_edges + [edge_to_add]
        sorted_dc = get_dc_comps(room_node_idx, filtered_edges)
        length_of_dc = len(sorted_dc)

    return edges_to_add


def get_dc_comps(room_idxs, edges):
    # Create a sample graph
    G = nx.Graph()
    G.add_nodes_from(room_idxs)
    G.add_edges_from(edges)

    # Find disconnected components
    disconnected_components = list(nx.connected_components(G))
    sorted_dc = sorted(disconnected_components, key=lambda x: len(x))

    return sorted_dc

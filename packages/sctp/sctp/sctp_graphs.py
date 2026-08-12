import numpy as np
from sctp.utils import paths
# from scipy.spatial import Delaunay, distance
from sctp.param import TRAV_LEVEL, MAX_EDGE_LENGTH, MIN_EDGE_LENGTH
from sctp import param
from sctp import graph as g
import math


def random_graph(n_vertex=8, xmin=0, ymin=0):
    """Generate a random graph with Delaunay triangulation and weighted edges."""    
    count = 0
    while True:
        start, goal, graph = g.generate_random_graph(n_vertex=n_vertex, xmin=xmin, ymin=ymin,\
                                max_edge_len=MAX_EDGE_LENGTH, min_edge_len=MIN_EDGE_LENGTH)
        if g.check_graph_valid(startID=start.id, goalID=goal.id, graph=graph):
            break
        count += 1
        if count > 10000:
            print("Cannot find a valid graph, try other seed ranges")
            raise ValueError("Cannot find a valid graph, try other seed ranges")
    return start, goal, graph

def random_island_graph(n_island=5, xmin=0, ymin=0, SG_dist_min=10):
    count = 0
    graph, islands, points = g.generate_island_graph(n_islands=n_island, xmin=xmin, ymin=ymin, 
                                max_edge_len=param.MAX_ISLAND_DISTANCE,min_edge_len=param.MIN_ISLAND_DISTANCE)
    while True:
        islands_vertices = [vertex for graph in islands for vertex in graph.vertices]
        start = min(enumerate(islands_vertices), key=lambda v: v[1].coord[0])[1]
        while True:
            goal = islands_vertices[np.random.randint(len(islands_vertices))]
            _, num_edges = paths.get_shortestPath_cost(graph=graph, start=start.id, goal=goal.id)
            if num_edges >=SG_dist_min:
                break
            
        if g.check_graph_valid(startID=start.id, goalID=goal.id, graph=graph):
            g.add_highways(out_graph=graph, islands=islands, points=points)
            break
        
        # reset the status of blocking point
        for poi in graph.pois:
            poi.block_status = int(0) if np.random.random() > poi.block_prob else int(1)
        
        count += 1
        if count > 2000:
            raise ValueError("Cannot find a valid graph, try other seed ranges")
    return start, goal, graph

def linear_graph_unc():
    start_node = g.Vertex(coord=(0.0, 0.0))
    node1 = g.Vertex(coord=(5.0, 0.0))
    goal_node = g.Vertex(coord=(15.0, 0.0))
    nodes = [start_node, node1, goal_node]
    graph = g.Graph(nodes)
    graph.edges.clear()
    graph.add_edge(start_node, node1, 0.5)
    graph.add_edge(node1, goal_node, 0.3)
    paths.dijkstra(graph=graph, goal=goal_node)
    return start_node, goal_node, graph


def disjoint_unc():  # edge 34 is blocked
    # this disjoint graph have 4 nodes (1,2,3,4) and 4 edges: (1,4), (1,2), (3,4), (2,3)
    nodes = []
    node1 = g.Vertex(coord=(0.0, 0.0))
    nodes.append(node1)
    node2 =  g.Vertex(coord=(4.0, 0.0))
    nodes.append(node2)
    node3 =  g.Vertex(coord=(8.0, 0.0)) # goal node
    nodes.append(node3)
    node4 =  g.Vertex(coord=(4.0, 4.0))
    nodes.append(node4)

    graph = g.Graph(nodes)
    graph.edges.clear()
    graph.add_edge(node1, node2, 0.1)
    graph.add_edge(node3, node4, 0.1)
    graph.add_edge(node2, node3, 0.9)
    graph.add_edge(node1, node4, 0.2)
    # vertices = graph.vertices + graph.pois
    paths.dijkstra(graph=graph, goal=node3)
    return node1, node3, graph


def s_graph_unc():
    """Generate a simple graph for testing purposes."""
    nodes = []
    node1 = g.Vertex(coord=(0.0, 0.0)) # start node
    nodes.append(node1)
    node2 = g.Vertex(coord=(4.0, 4.0))
    nodes.append(node2)
    node3 = g.Vertex(coord=(4.0, 0.0))
    nodes.append(node3)
    node4 = g.Vertex(coord=(8.0, 0.0)) # goal node
    nodes.append(node4)
    graph = g.Graph(nodes)
    graph.edges.clear()

    # adding edges
    graph.add_edge(node1, node2, 0.1)
    graph.add_edge(node1, node3, 0.1)
    graph.add_edge(node2, node3, 0.1)
    graph.add_edge(node2, node4, 0.1)
    graph.add_edge(node3, node4, 0.9)
    paths.dijkstra(graph=graph, goal=node4)
    return node1, node4, graph


def m_graph_unc():
    """Generate a simple graph for testing purposes."""
    nodes = []
    node1 = g.Vertex(coord=(-3.0, 4.0)) # start node
    nodes.append(node1)
    node2 = g.Vertex(coord=(-4.0, 8.5))
    nodes.append(node2)
    node3 = g.Vertex(coord=(0.0, 2.0))
    nodes.append(node3)
    node4 = g.Vertex(coord=(4.0, 0.0))
    nodes.append(node4)
    node5 = g.Vertex(coord=(4.0, 4.0))
    nodes.append(node5)
    node6 = g.Vertex(coord=(4.0, 8.0))
    nodes.append(node6)
    node7 = g.Vertex(coord=(8.0, 4.0)) # goal node
    nodes.append(node7)

    graph = g.Graph(nodes)
    graph.edges.clear()

    # add edges
    graph.add_edge(node1, node2, 0.1) #8
    graph.add_edge(node1, node3, 0.1) #9
    graph.add_edge(node2, node3, 0.1) #10
    graph.add_edge(node2, node5, 0.1) #11
    graph.add_edge(node2, node6, 0.1)#12
    graph.add_edge(node3, node4, 0.1) #13
    graph.add_edge(node3, node5, 0.1) #14
    graph.add_edge(node4, node5, 0.1) #15
    graph.add_edge(node4, node7, 0.90) #16
    graph.add_edge(node5, node7, 0.90) #17
    graph.add_edge(node6, node7, 0.1) #18
    graph.add_edge(node6, node5, 0.1) #19
    paths.dijkstra(graph=graph, goal=node7)
    return node1, node7, graph


def graph_stuck():
    """Generate a simple graph for testing purposes."""
    nodes = []
    node1 = g.Vertex(coord=(0.0, 0.0)) # start node
    nodes.append(node1)
    # node2 = Vertex(coord=(0.2, 5.0))
    # node2 = Vertex(coord=(2.0, 2.5))
    node2 = g.Vertex(coord=(0.0, 2.5))
    nodes.append(node2)
    node3 = g.Vertex(coord=(8.0, 2.5))
    nodes.append(node3)
    node4 = g.Vertex(coord=(12.0, 1.0))
    nodes.append(node4)
    node5 = g.Vertex(coord=(8.0, 0.0))
    nodes.append(node5)

    graph = g.Graph(nodes)
    graph.edges.clear()

    # add edges
    graph.add_edge(node1, node2, 0.25) #6
    graph.add_edge(node2, node3, 0.15) #7
    graph.add_edge(node3, node4, 0.84) #8
    # graph.add_edge(node3, node5, 0.88) #9
    graph.add_edge(node4, node5, 0.86) #10
    graph.add_edge(node1, node5, 0.77) #11
    paths.dijkstra(graph=graph, goal=node4)
    return node1, node4, graph


def island_sgraph():
    """Generate a simple graph for testing purposes."""
    nodes = []
    node1 = g.Vertex(coord=(0.0, 0.0)) # start node
    nodes.append(node1)
    node2 = g.Vertex(coord=(4.0, 3.0))
    nodes.append(node2)
    node3 = g.Vertex(coord=(4.0, 0.0))
    nodes.append(node3)
    node4 = g.Vertex(coord=(4.0, -4.0)) 
    nodes.append(node4)
    node5 = g.Vertex(coord=(8.0, -4.0))
    nodes.append(node5)
    node6 = g.Vertex(coord=(8.0, 0.0))
    nodes.append(node6)
    node7 = g.Vertex(coord=(8.0, 3.0))
    nodes.append(node7)
    node8 = g.Vertex(coord=(12.0, 0.0)) # goal node
    nodes.append(node8)
    # create a graph object
    graph = g.Graph(nodes)
    
    graph.edges.clear()
    # adding edges
    graph.add_edge(node1, node2, 0.2) #9
    graph.add_edge(node1, node3, 0.2) #10
    graph.add_edge(node1, node4, 0.1) #11
    graph.add_edge(node2, node3, 0.1) #12
    graph.add_edge(node3, node4, 0.1) #13
    graph.add_edge(node2, node7, 0.1) #14 - should be the edge with highest value
    graph.add_edge(node7, node8, 0.1) #15
    graph.add_edge(node6, node8, 0.1) #16
    graph.add_edge(node5, node8, 0.1) #17
    graph.add_edge(node7, node6, 0.1) #18
    graph.add_edge(node6, node5, 0.1) #19
    paths.dijkstra(graph=graph, goal=node8)
    return node1, node8, graph

def island_mgraph():
    """Generate a simple graph for testing purposes."""
    nodes = []
    node1 = g.Vertex(coord=(0.0, 3.0)) # start node
    nodes.append(node1)
    node2 = g.Vertex(coord=(4.0, 3.0))
    nodes.append(node2)
    node3 = g.Vertex(coord=(4.0, 0.0))
    nodes.append(node3)
    node4 = g.Vertex(coord=(4.0, -4.0)) 
    nodes.append(node4)
    node5 = g.Vertex(coord=(8.0, -4.0))
    nodes.append(node5)
    node6 = g.Vertex(coord=(8.0, 0.0))
    nodes.append(node6)
    node7 = g.Vertex(coord=(8.0, 3.0))
    nodes.append(node7)
    node8 = g.Vertex(coord=(12.0, 0.0)) # goal node
    nodes.append(node8)
    # create a graph object
    graph = g.Graph(nodes)
    
    graph.edges.clear()
    # adding edges
    graph.add_edge(node1, node2, 0.1) #9
    graph.add_edge(node1, node3, 0.1) #10
    graph.add_edge(node1, node4, 0.1) #11
    graph.add_edge(node2, node3, 0.1) #12
    graph.add_edge(node3, node4, 0.1) #13
    graph.add_edge(node2, node7, 0.5) #14 - should be the edge with highest value
    graph.add_edge(node7, node8, 0.4) #15
    graph.add_edge(node6, node8, 0.1) #16
    graph.add_edge(node5, node8, 0.1) #17
    graph.add_edge(node7, node6, 0.1) #18
    graph.add_edge(node6, node5, 0.1) #19
    graph.add_edge(node4, node5, 0.45) #20
    paths.dijkstra(graph=graph, goal=node8)
    return node1, node8, graph

def island_m2graph():
    """Generate a simple graph for testing purposes."""
    nodes = []
    node1 = g.Vertex(coord=(0.0, 3.0)) # start node
    nodes.append(node1)
    node2 = g.Vertex(coord=(4.0, 3.0))
    nodes.append(node2)
    node3 = g.Vertex(coord=(4.0, 0.0))
    nodes.append(node3)
    node4 = g.Vertex(coord=(4.0, -4.0)) 
    nodes.append(node4)
    node5 = g.Vertex(coord=(8.0, 3.0))
    nodes.append(node5)
    node6 = g.Vertex(coord=(8.0, 0.0))
    nodes.append(node6)
    node7 = g.Vertex(coord=(8.0, -4.0))
    nodes.append(node7)
    node8 = g.Vertex(coord=(8.0, -7.0))
    nodes.append(node8)
    node9 = g.Vertex(coord=(12.0, 0.0)) # goal node
    nodes.append(node9)
    # create a graph object
    graph = g.Graph(nodes)
    
    graph.edges.clear()
    # adding edges
    graph.add_edge(node1, node2, 0.18) #10
    graph.add_edge(node1, node3, 0.21) #11
    graph.add_edge(node1, node4, 0.15) #12
    graph.add_edge(node2, node3, 0.11) #13
    graph.add_edge(node2, node5, 0.50) #14 - could be the edge with highest value
    graph.add_edge(node9, node5, 0.4) #15
    graph.add_edge(node9, node6, 0.25) #16
    graph.add_edge(node9, node7, 0.29) #17
    graph.add_edge(node9, node8, 0.30) #18
    graph.add_edge(node5, node6, 0.19) #19
    graph.add_edge(node7, node8, 0.4) #20
    graph.add_edge(node4, node8, 0.45) #21 could be the edge with highest value
    paths.dijkstra(graph=graph, goal=node8)
    return node1, node8, graph



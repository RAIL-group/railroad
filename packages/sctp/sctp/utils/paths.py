import numpy as np
import heapq
from collections import defaultdict, deque # find shortest path
import sctp.sctp_graphs as graphs

# helper functions
def dijkstra(graph, goal):
    """Find the shortest paths to a goal from any node in a graph."""
    # Initialize the distance to all nodes as infinity
    vertices = graph.vertices + graph.pois
    edges = graph.edges
    dist = {node: float('inf') for node in vertices}
    dist[goal] = 0.0
    parent = {node: None for node in vertices}
    visited = set()
    queue = [(0.0, goal.id,  goal)]

    while queue:
        # Get the node with the smallest distance
        node_dist, id, node = heapq.heappop(queue)
        if node in visited:
            continue
        # Mark the node as visited
        visited.add(node)
        # Update the distance to the neighbors of the node
        for neighbor in node.neighbors:
            edge = [edge for edge in edges if ((edge.v1.id == node.id and edge.v2.id == neighbor) \
                        or (edge.v1.id == neighbor and edge.v2.id == node.id))][0]
            neigh_vertex = [node for node in vertices if node.id == neighbor][0]
            if dist[node] + edge.cost < dist[neigh_vertex]:
                dist[neigh_vertex] = dist[node] + edge.cost
                parent[neigh_vertex] = node
            heapq.heappush(queue, (dist[neigh_vertex], neigh_vertex.id, neigh_vertex))
    for node in vertices:
        node.heur2goal = dist[node]


def get_random_path(graph, start, goal):
    queue = [(0, start, [start])]
    visited = set()
    costs = {start: 0}
    
    while queue:
        (path_cost, current_node, path) = heapq.heappop(queue)
        if current_node in visited:
            continue
        visited.add(current_node)
        if current_node == goal:
            return path
        
        # Explore neighbors
        vertex = [v for v in graph.vertices+graph.pois if v.id == current_node][0]
        for nei in vertex.neighbors:
            step_cost = 0
            for edge in graph.edges:
                if (edge.v1.id == nei and edge.v2.id== current_node) or \
                        (edge.v1.id == current_node and edge.v2.id==nei):
                    step_cost = edge.rand_cost
                    break
                ValueError("Not find an edge")
            new_path_cost = path_cost + step_cost
            # If we found a shorter path to neighbor
            if nei not in costs or new_path_cost < costs[nei]:
                costs[nei] = new_path_cost
                new_path = path + [nei]
                heapq.heappush(queue, (new_path_cost, nei, new_path))
    ValueError("Found no path - error")
    return []


def get_shortestPath_cost(graph, start, goal):
    queue = [(0.0, start, [start])]
    visited = set()
    costs = {start: 0.0}
    # count = 0
    while queue:
        (path_cost, current_node, path) = heapq.heappop(queue)
        if current_node in visited:
            continue
        visited.add(current_node)
        if current_node == goal:
            return costs[current_node], len(path)
        
        # Explore neighbors
        vertices = [v for v in graph.vertices+graph.pois if v.id == current_node]
        if len(vertices) == 0: # the start node was removed
            return -1.0, 1000
        for nei in vertices[0].neighbors:
            step_cost = 0.0
            for edge in graph.edges:
                if (edge.v1.id == nei and edge.v2.id== current_node) or \
                        (edge.v1.id == current_node and edge.v2.id==nei):
                    step_cost = edge.dist
                    break
                ValueError("Not find an edge")
            new_path_cost = path_cost + step_cost
            # If we found a shorter path to neighbor
            if nei not in costs or new_path_cost < costs[nei]:
                costs[nei] = new_path_cost
                new_path = path + [nei]
                heapq.heappush(queue, (new_path_cost, nei, new_path))
    return -1.0, 1000

def get_shortest_path_with_blocknode(graph, start, goal, redge = [], block_nodes = []):
    block_pen = 500.0
    new_graph = graphs.remove_edge(graph, redge)
    queue = [(0.0, start, [start])]
    visited = set()
    costs = {start: 0.0}
    while queue:
        (path_cost, current_node, path) = heapq.heappop(queue)
        if current_node in visited:
            continue
        visited.add(current_node)
        if current_node == goal:
            return costs[current_node]
        
        # Explore neighbors
        vertices = [v for v in new_graph.vertices+new_graph.pois if v.id == current_node]
        if len(vertices) == 0: # the start node was removed
            return -1.0
        if len(vertices) > 1:
            print("The node id is not unique - the graph is wrong")
            return -2.0
        for nei in vertices[0].neighbors:
            step_cost = 0.0
            for edge in new_graph.edges:
                if (edge.v1.id == nei and edge.v2.id== current_node) or \
                        (edge.v1.id == current_node and edge.v2.id==nei):
                    step_cost = edge.dist
                    if  nei in block_nodes:
                        step_cost += block_pen
                    break
                ValueError("Not find an edge")
            new_path_cost = path_cost + step_cost
            # If we found a shorter path to neighbor
            if nei not in costs or new_path_cost < costs[nei]:
                costs[nei] = new_path_cost
                new_path = path + [nei]
                heapq.heappush(queue, (new_path_cost, nei, new_path))
    return -1.0



def is_reachable(graph, start, goal):
    if start == goal:
        return True
    # Queue for BFS
    queue = deque([start])
    # Set to keep track of visited vertices
    visited = {start}
    # print(f"The start node is: {start}")
    # BFS loop
    while queue:
        current = queue.popleft()  # Dequeue the next vertex
        vertices = [v for v in graph.pois+graph.vertices if v.id == current]
        if vertices == []:
            print(f"Robot is at the node: {current}")
            # exit()
        # Explore all neighbors of the current vertex
        for neighbor in vertices[0].neighbors:
            if neighbor not in visited:
                # If we found the goal, a path exists
                if neighbor == goal:
                    return True
                # Mark neighbor as visited and add to queue
                visited.add(neighbor)
                queue.append(neighbor)
    
    # If queue is empty and goal wasn't found, no path exists
    return False
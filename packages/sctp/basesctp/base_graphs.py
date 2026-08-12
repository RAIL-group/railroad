import numpy as np
import random, copy
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, distance
from itertools import combinations
import heapq

class RobotData:
    def __init__(self, *,robot_id=None, position=None,
                    cur_node=None, last_robot=None):
        if last_robot is not None:
            self.robotID = last_robot.robotID
            if position is not None:
                self.position = position
            else:
                self.position = last_robot.position
            self.cur_vertex = last_robot.cur_vertex
            self.last_vertex = self.cur_vertex
        else:
            self.robotID = robot_id
            if position is not None:
                self.position = position
            else:
                self.position = [cur_node.coord[0], cur_node.coord[1]] 
            self.cur_vertex = cur_node.id
            self.last_vertex = None

   # def __eq__(self, other):
   #    return self == other.__hash__()
   
   # def __hash_(self):
   #    return hash(self.robotID
    def getID(self):
        return self.robotID

class Graph():
    def __init__(self, vertices=[], edges=[]):
        self.vertices = vertices
        self.edges = edges

    def add_vertex(self, vertex):
        if isinstance(vertex, list):
            self.vertices.extend(vertex)
        else:
            self.vertices.append(vertex)

    def add_edge(self, vertex1, vertex2, block_prob=0.0):
        if vertex1 not in self.vertices or vertex2 not in self.vertices:
            raise ValueError("Vertices not in graph. Add vertices before adding edges.")
        edge = Edge(vertex1, vertex2, block_prob)
        self.edges.append(edge)
        vertex1.neighbors.append(vertex2.id)
        vertex2.neighbors.append(vertex1.id)
    
    def get_edge(self, id1, id2):
        for edge in self.edges:
            if (edge.v1.id == id1 and edge.v2.id == id2) or (edge.v1.id == id2 and edge.v2.id == id1):
                return edge
        raise ValueError("Edge not found in graph.")
        # return None


class Vertex:
    _id_counter = 1
    def __init__(self, coord):
        self.id = Vertex._id_counter
        Vertex._id_counter += 1
        self.parent = None
        self.coord = coord
        self.neighbors = []
        self.heur2goal = 0.0

    def get_id(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id) + hash(str(self.coord))


class Edge:
    def __init__(self, v1, v2, block_prob=0.0):
        self.v1 = v1
        self.v2 = v2
        self.hash_id = self.__hash__()
        self.dist = np.linalg.norm(
            np.array((v1.coord[0], v1.coord[1])) - np.array((v2.coord[0], v2.coord[1])))
        self.cost = self.dist
        self.poi_coord = None
        self.block_prob = block_prob
        self.block_status = 1 if random.random() < block_prob else 0

    def get_cost(self) -> float:
        return self.cost

    def update_cost(self, value):
        self.cost = value

    def get_poi_coord(self):
        return self.poi_coord

    def get_blocked_prob(self):
        return self.block_prob

    def __eq__(self, other):
        return self.hash_id == other.hash_id

    def __hash__(self):
        return hash(self.v1) + hash(self.v2)

def generate_random_coordinates(n, xmin, ymin, xmax, ymax):
    points = []
    attempts = 0
    max_attempts = 5000
    min_dist = 1.0
    max_dist = 6.0
    while len(points) < n and attempts < max_attempts:
        point = np.array([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)])
        if not points:
            points.append(point)
            point = np.array([np.random.uniform(point[0]+1.5, point[0]+2.0), np.random.uniform(point[1]+1.5, point[1]+2.0)])
            points.append(point)
            continue
        dists = distance.cdist(np.array([point]), np.array(points)).flatten()
        if np.all(dists >= min_dist) and np.sum(dists <= max_dist) >= 2:
            points.append(point)
        attempts +=1
    
    if len(points) < n:
        raise ValueError("Cannot get enough vertices")
    return points


def random_graph(n_vertex=10, xmin=0, ymin=0):
    """Generate a random graph with Delaunay triangulation and weighted edges."""    
    size = 5.0 * (np.sqrt(n_vertex)-1.0)
    points = generate_random_coordinates(n_vertex, xmin=xmin, ymin=ymin, xmax=1.5*size, ymax=size)
    tri = Delaunay(np.array(points))
    graph = Graph(vertices=[Vertex(coord=point) for point in points])
    edge_count = {}

    # Use a set to avoid duplicate edges
    edges = set()    
    for simplex in tri.simplices:
        edges.update(tuple(sorted((simplex[i], simplex[j]))) for i in range(3) for j in range(i + 1, 3))    
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            edge = tuple(sorted((simplex[i], simplex[j])))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    # Add edges to the graph with random weights
    for i, j in edges:
        dist = np.linalg.norm(np.array(graph.vertices[i].coord) - np.array(graph.vertices[j].coord))
        if dist > 6.0:
            continue
        if ((i, j) in boundary_edges or (j, i) in boundary_edges) and dist >5.0:
            continue

        if np.random.random() <0.85:
            graph.add_edge(graph.vertices[i], graph.vertices[j], np.random.uniform(0.1, 0.4))
        else:
            graph.add_edge(graph.vertices[i], graph.vertices[j], np.random.uniform(0.7, 0.90))
    startId = min(enumerate(points), key=lambda p: p[1][0])[0]
    start_pos = points[startId]
    goalId = max(enumerate(points), key=lambda p: np.linalg.norm(np.array(start_pos)- np.array(p[1])))[0]
    goal = graph.vertices[goalId]
    start = graph.vertices[startId]
    robots = RobotData(robot_id=1, position=[start.coord[0],start.coord[1]], cur_node=start)
    dijkstra(graph, goal)
    return start, goal, graph, robots


def m_graph_unc():
   """Generate a simple graph for testing purposes."""
   nodes = []
   node1 = Vertex(coord=(-3.0, 4.0)) # start node
   nodes.append(node1)
   node2 = Vertex(coord=(-15.0, 7.5))
   nodes.append(node2)
   node3 = Vertex(coord=(0.0, 2.0))
   nodes.append(node3)
   node4 = Vertex(coord=(4.0, 0.0))
   nodes.append(node4)
   node5 = Vertex(coord=(4.0, 4.0))
   nodes.append(node5)
   node6 = Vertex(coord=(4.0, 8.0))
   nodes.append(node6)
   node7 = Vertex(coord=(8.0, 4.0)) # goal node
   nodes.append(node7)
   
   graph = Graph(nodes)
   graph.edges.clear()
   
   # add edges
   graph.add_edge(node1, node2, 0.1)
   graph.add_edge(node1, node3, 0.1)
   graph.add_edge(node2, node3, 0.1)
   graph.add_edge(node2, node5, 0.1)
   graph.add_edge(node2, node6, 0.1)
   graph.add_edge(node3, node4, 0.1)
   graph.add_edge(node3, node5, 0.1)
   graph.add_edge(node4, node5, 0.1)
   graph.add_edge(node4, node7, 0.90)
   graph.add_edge(node5, node7, 0.90)
   graph.add_edge(node6, node7, 0.1)
   graph.add_edge(node6, node5, 0.1)

   robots = RobotData(robot_id=1, position=[-3.0, 4.0], cur_node=node1)
#    plot_street_graph(nodes, graph.edges)
   dijkstra(graph, node7)

   return node1, node7, graph, robots


def s_graph_unc():
   """Generate a simple graph for testing purposes."""
   nodes = []
   node1 = Vertex(coord=(0.0, 0.0)) # start node
   nodes.append(node1)
   node2 = Vertex(coord=(4.0, 4.0))
   nodes.append(node2)
   node3 = Vertex(coord=(4.0, 0.0))
   nodes.append(node3)
   node4 = Vertex(coord=(8.0, 0.0)) # goal node
   nodes.append(node4)
   graph = Graph(nodes)
   graph.edges.clear()

   # adding edges
   graph.add_edge(node1, node2, 0.1)
   graph.add_edge(node1, node3, 0.1)
   graph.add_edge(node2, node3, 0.1)
   graph.add_edge(node2, node4, 0.1)
   graph.add_edge(node3, node4, 0.9)

   robots = RobotData(robot_id=1, position=[0.0, 0.0], cur_node=node1)
#    plot_street_graph(nodes, graph.edges)
   dijkstra(graph, node4)

   return node1, node4, graph, robots


def disjoint_unc():  # edge 34 is blocked

   # this disjoint graph have 4 nodes (1,2,3,4) and 4 edges: (1,4), (1,2), (3,4), (2,3)
   nodes = []
   node1 = Vertex(coord=(0.0, 0.0))
   nodes.append(node1)
   node2 =  Vertex(coord=(4.0, 0.0))
   nodes.append(node2)
   node3 =  Vertex(coord=(8.0, 0.0)) # goal node
   nodes.append(node3)
   node4 =  Vertex(coord=(4.0, 4.0))
   nodes.append(node4)

   graph = Graph(nodes)
   graph.edges.clear()
   graph.add_edge(node1, node2, 0.1)
   graph.add_edge(node3, node4, 0.1)
   graph.add_edge(node2, node3, 0.9)
   graph.add_edge(node1, node4, 0.2)
   node_new = copy.copy(node1)
   robots = RobotData(robot_id=1, position=[0.0, 0.0], cur_node=node_new)
#    plot_street_graph(nodes, graph.edges)
   dijkstra(graph, node3)
   return node1, node3, graph, robots


def linear_graph_unc():
   nodes = []
   node1 = Vertex(coord=(0.0, 0.0)) # start node
   nodes.append(node1)
   node2 = Vertex(coord=(5.0, 0.0))
   nodes.append(node2)
   node3 = Vertex(coord=(15.0, 0.0)) # goal node
   nodes.append(node3)
   graph = Graph(nodes)
   graph.edges.clear()
   graph.add_edge(node1, node2, 0.9)
   graph.add_edge(node2, node3, 0.0)
   robots = RobotData(robot_id=1, position=[0.0, 0.0], cur_node=node1)
   dijkstra(graph, node3)
   return node1, node3, graph, robots

def dijkstra(graph, goal):
    """Find the shortest paths to a goal from any node in a graph."""
    # Initialize the distance to all nodes as infinity
    dist = {node: float('inf') for node in graph.vertices}
    dist[goal] = 0.0
    parent = {node: None for node in graph.vertices}
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
            edge = [edge for edge in graph.edges if ((edge.v1.id == node.id and edge.v2.id == neighbor) \
                        or (edge.v1.id == neighbor and edge.v2.id == node.id))][0]
            neigh_vertex = [node for node in graph.vertices if node.id == neighbor][0]
            if dist[node] + edge.cost < dist[neigh_vertex]:
                dist[neigh_vertex] = dist[node] + edge.cost
                parent[neigh_vertex] = node
            heapq.heappush(queue, (dist[neigh_vertex], neigh_vertex.id, neigh_vertex))
    for node in graph.vertices:
        node.heur2goal = dist[node]
    

# helper functions
def plot_graph(nodes, edges, name="Testing Graph", path=None, 
               startID=None, goalID=None, seed=None):
    """Plot graph using matplotlib."""
    plt.figure(figsize=(10, 10))

    # Plot edges
    for edge in edges:
        x_values = [edge.v1.coord[0], edge.v2.coord[0]]
        y_values = [edge.v1.coord[1], edge.v2.coord[1]]
        plt.plot(x_values, y_values, 'b-', alpha=0.7)
        # Display block probability
        mid_x = (edge.v1.coord[0] + edge.v2.coord[0]) / 2
        mid_y = (edge.v1.coord[1] + edge.v2.coord[1]) / 2
        probs = f"{edge.block_prob:.2f}/" + f"{edge.cost:.1f}"
        plt.text(mid_x, mid_y, probs, color='red', fontsize=8)

    # Plot nodes
    for node in nodes:
        plt.scatter(node.coord[0], node.coord[1], color='black', s=50)
        plt.text(node.coord[0] + 0.2, node.coord[1] + 0.2, f"{node.id}", color='blue', fontsize=10)
        if startID is not None:
            if node.id == startID:
                plt.text(node.coord[0] - 0.2, node.coord[1] - 0.5, "S", color='blue', fontsize=15)
                if seed is not None:
                    plt.text(node.coord[0] - 0.2, node.coord[1] + 0.5, f"seed={seed}", color='black', fontsize=10)
        if goalID is not None:
            if node.id == goalID:
                plt.text(node.coord[0] + 0.4, node.coord[1] - 0.4, "G", color='red', fontsize=15)

    if path is not None:
        x_values = []
        y_values = []
        for a in path:
            x_values.append(nodes[a.start-1].coord[0])
            y_values.append(nodes[a.start-1].coord[1])
        x_values.append(nodes[path[-1].end-1].coord[0])
        y_values.append(nodes[path[-1].end-1].coord[1])
        plt.plot(x_values, y_values, color='orange', linewidth=2)
    plt.title(name)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis("equal")
    plt.show()


# def plot_found_path(nodes, edges, path, name="Testing Graph", startID=None, goalID=None):


def print_graph(nodes, edges, show_edge=False, show_node=False):
    """Print the graph details."""
    if show_node:
        for node in nodes:
            print(f"Node {node.id}: ({node.coord[0]:.2f}, {node.coord[1]:.2f}) with neighbors: {node.neighbors}")
            print(f"The neighbors features:")
            for n in node.neighbors:
                edge = [edge for edge in edges if ((edge.v1.id == node.id and edge.v2.id == n) or (edge.v1.id == n and edge.v2.id == node.id))][0]
                print(f"edge {edge.id}: block prob {edge.block_prob:.2f}, block status {edge.block_status}, cost {edge.cost}")
    if show_edge:
        for edge in edges:
            print(f"Edge {edge.id}: block prob {edge.block_prob:.2f}, block status {edge.block_status}, cost {edge.cost}")

if __name__ == "__main__":
    # Parameters for the street system
    grid_rows = 5  # Number of rows in the grid
    grid_cols = 5  # Number of columns in the grid
    edge_probability = 1.0  # Probability of creating an edge between nodes

    # Generate and plot the street-like graph
    # nodes, edges = generate_street_graph(grid_rows, grid_cols, edge_probability)
    # name = "Street Graph"
    # nodes, edges = simple_graph()
    # name = "Simple Graph"
    # nodes, edges = simple_disjoint_graph()
    # name = "Simple Disjoint Graph"
    # print_graph(nodes, edges, show_edge=True)
    # print("THe neighbors are:")
    # for node in nodes:
    #     print(f"Node {node.id}: with neighbors: {node.neighbors}")

    # plot_street_graph(nodes, edges, name)

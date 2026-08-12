import copy
import pytest
from basesctp import base_graphs as graphs


def test_sctpbase_graph_vertex_edge_check():
    start_node = graphs.Vertex(coord=(0.0, 0.0))
    goal_node = graphs.Vertex(coord=(15.0, 0.0))
    node1 = graphs.Vertex(coord=(5.0, 0.0))
    
    graph = graphs.Graph(vertices=[start_node, goal_node, node1])
    graph.add_edge(start_node, node1, 0.0)
    graph.add_edge(node1, goal_node, 0.0)
    assert start_node in graph.vertices
    assert goal_node in graph.vertices

    assert start_node != goal_node  # Start should not be equal to goal
    assert start_node == start_node  # Start should be equal to itself


    start_copy = copy.copy(start_node)
    assert start_copy == start_node  # Still the same because id and coords are the same

    new_start = graphs.Vertex(coord=(0.0, 0.0))
    assert new_start != start_node  # Although coords are same, id is different

    # edge test
    edge = graph.edges[0]
    new_edge = copy.copy(edge)
    assert edge == new_edge
    # edge with opposite direction
    out_edge = graphs.Edge(node1, start_node, 0.0)
    assert out_edge == edge

def test_sctpbase_linear_graph_check():
    start, goal, ln_graph, robots = graphs.linear_graph_unc()
    
    assert start in ln_graph.vertices
    assert goal in ln_graph.vertices

    assert len(ln_graph.vertices) == 3
    assert len(ln_graph.edges) == 2

    assert robots.cur_vertex == start.id
    assert robots.position != start.coord
    assert robots.position[0] == start.coord[0]
    assert robots.position[1] == start.coord[1]
    robots.position = [1.0, 1.0]
    assert robots.position != start.coord
    assert robots.position[0] != start.coord[0]
    assert robots.position[1] != start.coord[1]
    for vertex in ln_graph.vertices:
        if vertex.id == 1:
            assert vertex.heur2goal == 15.0
        elif vertex.id == 2:
            assert vertex.heur2goal == 10.0
        elif vertex.id == 3:
            assert vertex.heur2goal == 0.0
    
def test_sctpbase_disjoint_graph_check():
    start, goal, dj_graph, robots = graphs.disjoint_unc()
    
    assert start in dj_graph.vertices
    assert goal in dj_graph.vertices

    assert len(dj_graph.vertices) == 4
    assert len(dj_graph.edges) == 4

    assert robots.cur_vertex == start.id
    assert robots.position != start.coord
    assert robots.position[0] == start.coord[0]
    assert robots.position[1] == start.coord[1]
    for vertex in dj_graph.vertices:
        if vertex.id == 1:
            assert vertex.heur2goal == 8.0
        elif vertex.id == 2:
            assert vertex.heur2goal == 4.0
        elif vertex.id == 3:
            assert vertex.heur2goal == 0.0
        elif vertex.id == 4:
            assert vertex.heur2goal == pytest.approx(5.56, 0.1)

def test_sctpbase_sgraph_check():
    start, goal, s_graph, robot = graphs.s_graph_unc()
    
    assert start in s_graph.vertices
    assert goal in s_graph.vertices

    assert len(s_graph.vertices) == 4
    assert len(s_graph.edges) == 5

    assert robot.cur_vertex == start.id
    assert robot.position != start.coord
    assert robot.position[0] == start.coord[0]
    assert robot.position[1] == start.coord[1]
    for vertex in s_graph.vertices:
        if vertex.id == 1:
            assert vertex.heur2goal == 8.0
        elif vertex.id == 2:
            assert vertex.heur2goal == pytest.approx(5.56, 0.1)
        elif vertex.id == 4:
            assert vertex.heur2goal == 0.0
        elif vertex.id == 3:
            assert vertex.heur2goal == 4.0
    
def test_sctpbase_mgraph_check():
    start, goal, m_graph, robot = graphs.m_graph_unc()
    
    assert start in m_graph.vertices
    assert goal in m_graph.vertices

    assert len(m_graph.vertices) == 7
    assert len(m_graph.edges) == 12

    assert robot.cur_vertex == start.id
    assert robot.position != start.coord
    assert robot.position[0] == start.coord[0]
    assert robot.position[1] == start.coord[1]
    for vertex in m_graph.vertices:
        if vertex.id == 1:
            assert vertex.heur2goal == pytest.approx(4.0+4.5+3.6, 0.2)
        elif vertex.id == 2:
            assert vertex.heur2goal == pytest.approx(23.3, 0.2)
        elif vertex.id == 3:
            assert vertex.heur2goal == pytest.approx(8.5, 0.1)
        elif vertex.id == 4:
            assert vertex.heur2goal == pytest.approx(5.7, 0.05)
        elif vertex.id == 5:
            assert vertex.heur2goal == pytest.approx(4.0, 0.05)
        elif vertex.id == 6:
            assert vertex.heur2goal == pytest.approx(5.7, 0.05)
        elif vertex.id == 7:
            assert vertex.heur2goal == 0.0

    
def test_sctpbase_rangraph_check():
    start, goal, ran_graph, robot = graphs.random_graph(n_vertex=15)
    graphs.plot_graph(nodes=ran_graph.vertices, edges=ran_graph.edges, startID=start.id, goalID=goal.id)
    

   
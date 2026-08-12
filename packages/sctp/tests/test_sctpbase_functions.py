import pytest
from pouct_planner import core
from basesctp import base_pomdpstate, base_navigation
from basesctp import base_graphs as graphs
from sctp.param import EventOutcome

def test_robot_position():
    start_node = graphs.Vertex(coord=(0.0, 0.0))
    goal_node = graphs.Vertex(coord=(15.0, 0.0))
    node1 = graphs.Vertex(coord=(5.0, 0.0))
    
    graph = graphs.Graph(vertices=[start_node, goal_node, node1])
    graph.add_edge(start_node, node1, 0.0)
    graph.add_edge(node1, goal_node, 0.0)
    action1 = base_pomdpstate.Action(start_node=start_node.id, target_node=node1.id)
    robot = graphs.RobotData(robot_id = 1, position=[0.0, 0.0], cur_node=start_node)
    state = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal_node.id, robots=robot)
    state.robot_move(action1)
    assert state.robots.cur_vertex == node1.id
    assert state.robots.position == [5.0, 0.0]



def test_get_edge_from_action_function():
    start_node = graphs.Vertex(coord=(0.0, 0.0))
    goal_node = graphs.Vertex(coord=(15.0, 0.0))
    node1 = graphs.Vertex(coord=(5.0, 0.0))
    
    graph = graphs.Graph(vertices=[start_node, goal_node, node1])
    graph.add_edge(start_node, node1, 0.0)
    graph.add_edge(node1, goal_node, 0.0)
    action1 = base_pomdpstate.Action(start_node=start_node.id, target_node=node1.id)
    robot = graphs.RobotData(robot_id = 1, position=(0.0, 0.0), cur_node=start_node)
    state = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal_node.id, robots=robot)
    edge = base_pomdpstate.get_edge_from_action(state, action1)
    assert edge == graph.edges[0]
    action1 = base_pomdpstate.Action(start_node=node1.id, target_node=goal_node.id)
    edge2 = base_pomdpstate.get_edge_from_action(state, action1)
    assert edge2 == graph.edges[1]

def test_sctpbase_function_rollout():
    start, goal, l_graph, robots = graphs.s_graph_unc()
    # start, goal, l_graph, robots = graphs.m_graph_unc()
    init_state = base_pomdpstate.SCTPBaseState(graph=l_graph, goal=goal.id, robots=robots)
    all_actions = init_state.get_actions()
    assert len(all_actions) == 2

    action2 = base_pomdpstate.Action(start_node=start.id, target_node=2)
    action3 = base_pomdpstate.Action(start_node=start.id, target_node=3)
    for action in all_actions:
        assert action in [action2, action3]


    best_action, cost, path = core.po_mcts(init_state, n_iterations=10,C=15.0)

    # best_action, cost, path = core.po_mcts(init_state, n_iterations=10,
    #                             C=15.0, rollout_fn=base_pomdpstate.sctpbase_rollout)


    # assert best_action == action3
    # assert len(path[1]) == 4
    

import pytest, random
import numpy as np
from basesctp import base_pomdpstate, base_navigation
from pouct_planner import core
from basesctp import base_graphs as graphs

def test_sctpbase_planner_lineargraph():
    start_node, goal_node, graph, robot = graphs.linear_graph_unc()
    
    graph.edges[0].block_prob = 0.0
    graph.edges[1].block_prob = 0.0
    initial_state = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal_node.id, robots=robot)
    all_actions = initial_state.get_actions()
    assert len(all_actions) == 1
    action = all_actions[0]
    assert action.end == 2
        
    best_action, cost, _  = core.po_mcts(initial_state, n_iterations=1000,
                                         rollout_fn=base_pomdpstate.sctpbase_rollout)
    desired_action = base_pomdpstate.Action(start_node=start_node.id, target_node=graph.vertices[1].id)
    assert best_action == desired_action
    assert cost == pytest.approx(15.0, abs=0.2)

    graph.edges[0].block_prob = 0.4
    graph.edges[1].block_prob = 0.9
    initial_state2 = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal_node.id, robots=robot)        
    best_action, cost, _  = core.po_mcts(initial_state2, n_iterations=2000,
                                         rollout_fn=base_pomdpstate.sctpbase_rollout)
    desired_action = base_pomdpstate.Action(start_node=start_node.id, target_node=graph.vertices[1].id)
    assert best_action == desired_action
    assert cost == pytest.approx(5.0*0.6 + (0.4+0.9)*base_pomdpstate.BLOCK_COST + 0.1*10 , abs=2.5)

def test_sctpbase_planner_disjoint_noblock():
    explore_param = 15.0
    # start, goal, graph, robots = graphs.disjoint_unc()
    nodes = []
    node1 = graphs.Vertex(coord=(0.0, 0.0))
    nodes.append(node1)
    node2 =  graphs.Vertex(coord=(4.0, 0.0))
    nodes.append(node2)
    node3 =  graphs.Vertex(coord=(8.0, 0.0)) # goal node
    nodes.append(node3)
    node4 =  graphs.Vertex(coord=(4.0, 4.0))
    nodes.append(node4)

    graph = graphs.Graph(nodes)
    graph.edges.clear()
    graph.add_edge(node1, node2, 0.1)
    graph.add_edge(node3, node4, 0.1)
    graph.add_edge(node2, node3, 0.9)
    graph.add_edge(node1, node4, 0.2)
    # node_new = copy.copy(node1)
    robots = graphs.RobotData(robot_id=1, position=[0.0, 0.0], cur_node=node1)
    graphs.dijkstra(graph, node3)


    for edge in graph.edges:
        edge.block_prob = 0.0
    init_state1 = base_pomdpstate.SCTPBaseState(graph=graph,goal=node3.id, robots=robots)
    all_actions = init_state1.get_actions()
    assert len(all_actions) == 2
    
    action2 = base_pomdpstate.Action(start_node=node1.id, target_node=graph.vertices[1].id)
    action4 = base_pomdpstate.Action(start_node=node1.id, target_node=graph.vertices[3].id)
    for action in all_actions:
        assert action in [action2, action4]
    
    action = all_actions[0] # action 2
    outcome_states = init_state1.transition(action)
    assert len(outcome_states) == 1
    for state, (prob, cost) in outcome_states.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV
        assert prob == 1.0
        assert cost == pytest.approx(4.0, abs=0.1)

    action = all_actions[1] # action 4
    outcome_states2 = init_state1.transition(action)
    assert len(outcome_states2) == 1
    for state, (prob, cost) in outcome_states2.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV
        assert prob == 1.0
        assert cost == pytest.approx(5.56, abs=0.1)

    assert init_state1.history.get_data_length()==0
    best_action, cost, path  = core.po_mcts(init_state1,C=explore_param, n_iterations=1000,
                                            rollout_fn=base_pomdpstate.sctpbase_rollout)
    # print(best_action)
    assert best_action == action2
    assert cost == pytest.approx(8.0, abs=0.5)
    assert path[0][0].start == 1
    assert path[0][0].end == 2
    assert path[0][1].end == 3
    


def test_sctpbase_planner_disjoint_prob():
    explore_param = 15.0
    start, goal, graph, robots = graphs.disjoint_unc()
    init_state1 = base_pomdpstate.SCTPBaseState(graph=graph,goal=goal.id, robots=robots)
    all_actions = init_state1.get_actions()
    assert len(all_actions) == 2
    
    action2 = base_pomdpstate.Action(start_node=start.id, target_node=graph.vertices[1].id)
    action4 = base_pomdpstate.Action(start_node=start.id, target_node=graph.vertices[3].id)
    for action in all_actions:
        assert action in [action2, action4]
    
    action = all_actions[0] # action 2
    outcome_states = init_state1.transition(action)
    assert len(outcome_states) == 2

    action = all_actions[1] # action 4
    outcome_states2 = init_state1.transition(action)
    assert len(outcome_states2) == 2

    # assert init_state1.history.get_data_length()==0
    best_action, cost, path_cost  = core.po_mcts(init_state1,C=explore_param, n_iterations=1000,
                                            rollout_fn=base_pomdpstate.sctpbase_rollout)
    print([(p.start, p.end) for p in path_cost[0]])
    assert best_action == action4
    assert cost == pytest.approx((0.1+0.2)*base_pomdpstate.BLOCK_COST + 0.9*5.56 + 0.8*5.56, abs=8.0)
    assert path_cost[0][0].start == 1
    assert path_cost[0][0].end == 4
    assert path_cost[0][1].end == 3
   

def test_sctpbase_planner_sgraph_noblock():
    exp_param = 15.0
    start, goal, graph, robots = graphs.s_graph_unc()
    for edge in graph.edges:
        edge.block_prob = 0.0
    init_state1 = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal.id, robots=robots)
    all_actions = init_state1.get_actions()
    assert len(all_actions) == 2

    action2 = base_pomdpstate.Action(start_node=start.id, target_node=2)
    action3 = base_pomdpstate.Action(start_node=start.id, target_node=3)
    for action in all_actions:
        assert action in [action2, action3]
    
    action = all_actions[0] # action 2
    outcome_states = init_state1.transition(action)
    assert len(outcome_states) == 1
    for state, (prob, cost) in outcome_states.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV
        assert prob == 1.0
        assert cost == pytest.approx(5.66, abs=0.1)

    action = all_actions[1] # action 4
    outcome_states2 = init_state1.transition(action)
    assert len(outcome_states2) == 1
    for state, (prob, cost) in outcome_states2.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV
        assert prob == 1.0
        assert cost == pytest.approx(4.0, abs=0.1)

    best_action, exp_cost, path_cost  = core.po_mcts(init_state1, C=exp_param, n_iterations=1000,
                                                    rollout_fn=base_pomdpstate.sctpbase_rollout)
    print([(p.start, p.end) for p in path_cost[0]])
    assert best_action == action3
    # assert len(path) == 2
    assert exp_cost == pytest.approx(8.0, abs=2.5)
    assert path_cost[0][0].start == 1
    assert path_cost[0][0].end == 3
    assert path_cost[0][1].end == 4

def test_sctpbase_planner_sgraph_prob1():
    start, goal, graph, robots = graphs.s_graph_unc() # edge 3-4 is blocked
    exp_param = 10.0
    init_state1 = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal.id, robots=robots)
    all_actions = init_state1.get_actions()
    assert len(all_actions) == 2

    action2 = base_pomdpstate.Action(start_node=start.id, target_node=2)
    action3 = base_pomdpstate.Action(start_node=start.id, target_node=3)
    for action in all_actions:
        assert action in [action2, action3]
    
    action = all_actions[0] # action 2
    outcome_states = init_state1.transition(action)
    assert len(outcome_states) == 2
    for state, (prob, cost) in outcome_states.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV or  \
                state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.BLOCK
        assert prob == 0.1 or prob == 0.9
        assert cost == pytest.approx(5.66, abs=0.5) or cost == pytest.approx(base_pomdpstate.BLOCK_COST, abs=1.0)

    action = all_actions[1] # action 3
    outcome_states2 = init_state1.transition(action)
    assert len(outcome_states2) == 2
    for state, (prob, cost) in outcome_states2.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV or  \
                state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.BLOCK
        assert prob == 0.1 or prob == 0.9
        assert cost == pytest.approx(4.0, abs=0.5) or cost == pytest.approx(base_pomdpstate.BLOCK_COST, abs=1.0)

    best_action, exp_cost, path_cost  = core.po_mcts(init_state1, C=exp_param, n_iterations=1000,
                                                rollout_fn=base_pomdpstate.sctpbase_rollout)
    print([(p.start, p.end) for p in path_cost[0]])
    assert best_action == action2
    assert len(path_cost[0]) == 2
    assert exp_cost == pytest.approx(0.9*2*5.66 + 0.1*2*base_pomdpstate.BLOCK_COST, abs=10.0)
    assert path_cost[0][0].start == 1
    assert path_cost[0][0].end == 2
    assert path_cost[0][1].end == 4

def test_sctpbase_planner_sgraph_prob2():
    start, goal, graph, robots = graphs.s_graph_unc() # edge 3-4 is blocked
    exp_param = 100.0
    graph.edges[0].block_prob = 0.8

    edge = graph.get_edge(1, 2)
    assert edge.block_prob == 0.8
    edge = graph.get_edge(3, 4)
    assert edge.block_prob == 0.9

    init_state1 = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal.id, robots=robots)
    all_actions = init_state1.get_actions()
    assert len(all_actions) == 2

    action2 = base_pomdpstate.Action(start_node=start.id, target_node=2)
    action3 = base_pomdpstate.Action(start_node=start.id, target_node=3)
    for action in all_actions:
        assert action in [action2, action3]
    
    action = all_actions[0] # action 2
    outcome_states = init_state1.transition(action)
    assert len(outcome_states) == 2
    for state, (prob, cost) in outcome_states.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV or  \
                state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.BLOCK
        assert prob == pytest.approx(0.2, 0.01) or prob == pytest.approx(0.8, 0.01)
        assert cost == pytest.approx(5.66, abs=0.5) or cost == pytest.approx(base_pomdpstate.BLOCK_COST, abs=1.0)

    action = all_actions[1] # action 3
    outcome_states2 = init_state1.transition(action)
    assert len(outcome_states2) == 2
    for state, (prob, cost) in outcome_states2.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV or  \
                state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.BLOCK
        assert prob == pytest.approx(0.1, 0.01) or prob == pytest.approx(0.9, 0.01)
        assert cost == pytest.approx(4.0, abs=0.5) or cost == pytest.approx(base_pomdpstate.BLOCK_COST, abs=1.0)

    best_action, exp_cost, path_cost  = core.po_mcts(init_state1, C=exp_param, n_iterations=1000,
                                                rollout_fn=base_pomdpstate.sctpbase_rollout)
    print([(p.start, p.end) for p in path_cost[0]])
    assert best_action == action3
    assert len(path_cost[0]) == 3
    # assert exp_cost == pytest.approx(0.9*5.66 +0.9*2*4.0+ 0.1*3*base_pomdpstate.BLOCK_COST, abs=10.0)
    assert path_cost[0][0].start == 1
    assert path_cost[0][0].end == 3
    assert path_cost[0][1].end == 2
    assert path_cost[0][2].end == 4

def test_sctpbase_planner_mgraph_noblock():
    start, goal, graph, robots = graphs.m_graph_unc()
    exp_param=10.0
    for edge in graph.edges: # edge 1-2 and 3-4 are blocked
      edge.block_prob = 0.0

    init_state1 = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal.id, robots=robots)
    all_actions = init_state1.get_actions()
    assert len(all_actions) == 2


    action2 = base_pomdpstate.Action(start_node=start.id, target_node=2)
    action3 = base_pomdpstate.Action(start_node=start.id, target_node=3)
    for action in all_actions:
        assert action in [action2, action3]
    
    action = all_actions[0] # action 2
    outcome_states = init_state1.transition(action)
    assert len(outcome_states) == 1
    for state, (prob, cost) in outcome_states.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV
        assert prob == pytest.approx(1.0, 0.01)
        assert cost == pytest.approx(12.5, abs=0.5)

    action = all_actions[1] # action 3
    outcome_states2 = init_state1.transition(action)
    assert len(outcome_states2) == 1
    for state, (prob, cost) in outcome_states2.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV
        assert prob == pytest.approx(1.0, 0.01)
        assert cost == pytest.approx(3.61, abs=0.1)

    best_action, exp_cost, path_cost  = core.po_mcts(init_state1, C=exp_param, n_iterations=2000,
                                                rollout_fn=base_pomdpstate.sctpbase_rollout)
    print([(p.start, p.end) for p in path_cost[0]])
    assert best_action == action3
    assert len(path_cost[0]) == 3
    assert exp_cost == pytest.approx(3.6 +4.47+ 4.0, abs=5.0)
    assert path_cost[0][0].start == 1
    assert path_cost[0][0].end == 3
    assert path_cost[0][1].end == 5
    assert path_cost[0][2].end == 7

def test_sctpbase_planner_mgraph_prob():
    start, goal, graph, robots = graphs.m_graph_unc()
    # graphs.plot_street_graph(graph.vertices, graph.edges)
    assert graph.get_edge(4, 7).block_prob == 0.9
    assert graph.get_edge(5, 7).block_prob == 0.9
    exp_param=100.0

    init_state1 = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal.id, robots=robots)
    all_actions = init_state1.get_actions()
    assert len(all_actions) == 2


    action2 = base_pomdpstate.Action(start_node=start.id, target_node=2)
    action3 = base_pomdpstate.Action(start_node=start.id, target_node=3)
    for action in all_actions:
        assert action in [action2, action3]
    
    action = all_actions[0] # action 2
    outcome_states = init_state1.transition(action)
    assert len(outcome_states) == 2
    for state, (prob, cost) in outcome_states.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV or \
                state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.BLOCK
        # assert prob == pytest.approx(0.1, 0.01) or prob == pytest.approx(0.9, 0.01)
        assert cost == pytest.approx(12.5, abs=0.1) or \
                cost == pytest.approx(base_pomdpstate.BLOCK_COST, abs=0.1)

    action = all_actions[1] # action 3
    outcome_states = init_state1.transition(action)
    assert len(outcome_states) == 2
    for state, (prob, cost) in outcome_states.items():
        assert state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV or \
                state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.BLOCK
        # assert prob == pytest.approx(0.1, 0.01) or prob == pytest.approx(0.9, 0.01)
        assert cost == pytest.approx(3.6, abs=0.1) or \
                cost == pytest.approx(base_pomdpstate.BLOCK_COST, abs=0.1)

    best_action, exp_cost, path_cost  = core.po_mcts(init_state1, C=exp_param, n_iterations=5000,
                                                rollout_fn=base_pomdpstate.sctpbase_rollout)
    # print(best_action)
    print([(p.start, p.end) for p in path_cost[0]])
    assert best_action == action3
    
    
    assert len(path_cost[0]) == 4
    # print(exp_cost)
    assert exp_cost == pytest.approx((3.6 +4.47+ 4.0+5.66)*0.9+0.1*4*base_pomdpstate.BLOCK_COST, abs=10.0)
    assert path_cost[0][0].start == 1
    assert path_cost[0][0].end == 3
    assert path_cost[0][1].end == 5
    assert path_cost[0][2].end == 6
    assert path_cost[0][3].end == 7

def test_sctpbase_planner_rangraph_prob():
    # seed = np.random.randint(0, 1000)
    seed = 719
    np.random.seed(seed)
    n_vertex = 12
    start, goal, ran_graph, robots = graphs.random_graph(n_vertex=n_vertex)
    print("Done generating the random graph")
    exp_param=100.0

    init_state1 = base_pomdpstate.SCTPBaseState(graph=ran_graph, goal=goal.id, robots=robots)
    all_actions = init_state1.get_actions()
    
    
    best_action, exp_cost, path_cost  = core.po_mcts(init_state1, C=exp_param, n_iterations=10000,
                                                rollout_fn=base_pomdpstate.sctpbase_rollout)
    # print(best_action)
    print([(p.start, p.end) for p in path_cost[0]])
    print(f"the seed is: {seed}")
    graphs.plot_graph(ran_graph.vertices, ran_graph.edges, path=path_cost[0], 
                                    startID=start.id, goalID=goal.id, seed=seed)
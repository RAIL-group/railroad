import pytest
from basesctp import base_pomdpstate, base_navigation
from pouct_planner import core
from basesctp import base_graphs as graphs


def test_sctpbase_state_lineargraph_outcome_states():
    # Initialize nodes
    start = graphs.Vertex(coord=(0.0, 0.0))
    node1 = graphs.Vertex(coord=(5.0, 0.0))
    goal = graphs.Vertex(coord=(15.0, 0.0))
    nodes = [start, node1, goal]
    assert start.id == 1 and node1.id == 2 and goal.id == 3

    # Make graph and add connectivity
    graph = graphs.Graph(vertices=nodes)
    graph.edges.clear()
    graph.add_edge(start, node1, 0.0)
    graph.add_edge(node1, goal, 0.0)

    # Why is the position and current_vertex differently initialized?
    robots = graphs.RobotData(robot_id=1, position=(0.0, 0.0), cur_node=start)
    initial_state = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal.id, robots=robots)

    all_actions = initial_state.get_actions()
    assert len(all_actions) == 1
    action = base_pomdpstate.Action(start_node=start.id, target_node=node1.id)
    assert action == all_actions[0]
    assert initial_state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.CHANCE
    outcome_states = initial_state.transition(action)
    # outcome states as taking action with certain probability
    assert len(outcome_states) == 1
    new_state = list(outcome_states.keys())[0]
    assert outcome_states[new_state] == (1.0, 5.0)
    assert len(new_state.get_actions()) == 1 # moving back action is not allow
    # the robot current vertex should be updated
    assert new_state.robots.cur_vertex != initial_state.robots.cur_vertex
    assert new_state.history.get_action_outcome(action) == base_pomdpstate.EventOutcome.TRAV
    assert new_state.history.get_data_length() == 2
    # the robot takes action of moving back
    action2 = base_pomdpstate.Action(start_node=node1.id, target_node=start.id)
    outcome_states = new_state.transition(action2)
    assert len(outcome_states) == 1
    new_state2 = list(outcome_states.keys())[0]
    assert new_state2.history == new_state.history
    assert len(new_state2.get_actions()) == 1
    assert new_state2.robots.cur_vertex == initial_state.robots.cur_vertex
    assert new_state2.robots.cur_vertex != new_state.robots.cur_vertex
    # the robot takes action of moving to goal
    action3 = base_pomdpstate.Action(start_node=node1.id, target_node=goal.id)
    outcome_states = new_state.transition(action3)
    new_state3 = list(outcome_states.keys())[0]
    assert new_state3.is_goal_state == True
    assert new_state3.history.get_data_length() == 4
    assert new_state3.history.get_action_outcome(action3) == base_pomdpstate.EventOutcome.TRAV
    assert len(new_state3.get_actions()) == 1


def test_sctpbase_state_lineargraph_with_history():
    # Initialize nodes
    start = graphs.Vertex(coord=(0.0, 0.0))
    node1 = graphs.Vertex(coord=(5.0, 0.0))
    goal = graphs.Vertex(coord=(15.0, 0.0))
    nodes = [start, node1, goal]
    assert start.id == 1 and node1.id == 2 and goal.id == 3

    # Make graph and add connectivity
    graph = graphs.Graph(vertices=nodes)
    graph.edges.clear()
    graph.add_edge(start, node1, 0.8)
    graph.add_edge(node1, goal, 0.5)

    # Why is the position and current_vertex differently initialized?
    robots = graphs.RobotData(robot_id=1, position=(0.0, 0.0), cur_node=start)
    action1 = base_pomdpstate.Action(start_node=start.id, target_node=node1.id)
    node1_blocked_history = base_pomdpstate.History()
    node1_blocked_history.add_history(action1, base_pomdpstate.EventOutcome.BLOCK)
    initial_state = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal.id, robots=robots, history=node1_blocked_history)

    all_actions = initial_state.get_actions()
    assert len(all_actions) == 1
    action1 = all_actions[0]
    # Since node1 is blocked and is in history, we should get only one state in transition where the probability is 1.0
    # and the cost is 10e5
    # WITH HISTORY
    outcome_states = initial_state.transition(action1)
    assert len(outcome_states) == 1
    new_state = list(outcome_states.keys())[0]
    assert new_state.robots.cur_vertex != initial_state.robots.cur_vertex
    assert outcome_states[new_state] == (1.0, base_pomdpstate.BLOCK_COST)

    # WITHOUT HISTORY
    initial_state2 = base_pomdpstate.SCTPBaseState(graph=graph, goal=goal.id, robots=robots) 
    outcome_states = initial_state2.transition(action1)
    new_state1 = list(outcome_states.keys())[0]
    new_state2 = list(outcome_states.keys())[1]
    assert len(outcome_states) == 2
    assert outcome_states[new_state2] == (0.8, base_pomdpstate.BLOCK_COST)
    assert outcome_states[new_state1] == (pytest.approx(0.2, 0.01), 5.0)
    assert new_state1.robots.cur_vertex == new_state.robots.cur_vertex
    mb_action = base_pomdpstate.Action(start_node=node1.id, target_node=start.id)
    outcome_states_1 = new_state2.transition(mb_action)
    assert len(outcome_states_1) == 1
    new_state3 = list(outcome_states_1.keys())[0]
    assert new_state3.history == new_state2.history
    assert outcome_states_1[new_state3] == (1.0, base_pomdpstate.BLOCK_COST)

    outcome_states_2 = new_state1.transition(mb_action)
    assert len(outcome_states_2) == 1
    new_state4 = list(outcome_states_2.keys())[0]
    assert new_state4.history == new_state1.history
    edge = base_pomdpstate.get_edge_from_action(new_state1, mb_action)
    assert outcome_states_2[new_state4] == (1.0, edge.cost)


def test_sctpbase_state_disjointgraph_outcome_states():
    node1 = graphs.Vertex(coord=(0.0, 0.0))
    node2 = graphs.Vertex(coord=(0.0, 4.0))
    node3 = graphs.Vertex(coord=(4.0, 4.0))
    node4 = graphs.Vertex(coord= (5.0, 4.0))
    nodes = [node1, node2, node3, node4]

    graph = graphs.Graph(vertices=nodes)
    graph.edges.clear()
    graph.add_edge(node1, node2, 0.1)
    graph.add_edge(node3, node4, 0.0)
    graph.add_edge(node1, node4, 0.0)
    graph.add_edge(node2, node3, 0.0)

    robots = graphs.RobotData(robot_id=1, position=(0.0, 0.0), cur_node=node1)
    initial_state = base_pomdpstate.SCTPBaseState(graph=graph, goal=node3.id, robots=robots)
    all_actions = initial_state.get_actions()

    assert len(all_actions) == 2
    for action in all_actions:
        assert action.end in [node2.id, node4.id]

    # Try with action to node 4
    action1 = all_actions[1]
    outcome_states = initial_state.transition(action1)
    assert len(outcome_states) == 1
    # Lets say we choose node2 action
    action2 = all_actions[0]
    outcome_states = initial_state.transition(action2)
    assert len(outcome_states) == 2

    # One is success history where node2 is not blocked (TRAV)
    success_history = base_pomdpstate.History()
    success_history.add_history(action2, base_pomdpstate.EventOutcome.TRAV)
    new_state_traversable = base_pomdpstate.get_state_from_history(outcome_states, success_history)
    assert outcome_states[new_state_traversable][0] == 0.9
    edge = base_pomdpstate.get_edge_from_action(initial_state, action2)
    assert outcome_states[new_state_traversable][1] == edge.cost

    # another is failure history where node2 is blocked (BLOCK)
    failure_history = base_pomdpstate.History()
    failure_history.add_history(action2, base_pomdpstate.EventOutcome.BLOCK)
    new_state_failure = base_pomdpstate.get_state_from_history(outcome_states, failure_history)
    assert outcome_states[new_state_failure][0] == 0.1
    assert outcome_states[new_state_failure][1] == base_pomdpstate.BLOCK_COST


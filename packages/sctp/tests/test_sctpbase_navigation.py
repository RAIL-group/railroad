import pytest
from pouct_planner import core
from basesctp import base_graphs as graphs
from basesctp import base_navigation, base_pomdpstate

graph_types = ['linear_det', 'linear_unc', 'disjoint_det', 'disjoint_unc',
               's_regular_det', 'm_regular_unc', 's_streetgraph']

def test_sctpbase_nav_sense_update_disjointgraph():
   # testing on a simple disjoint graph
   
   start, goal, nodes, edges, robots = graphs.disjoint_unc()
   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs, 
                     goasl=goal, vertices=nodes, edges=edges, robots=robots)

   observed_status = base_navigation.sense(initial_state) # sense the env 
   assert observed_status == {(1, 2): 0.0, (1, 4): 1.0}
   state, _ = base_navigation.update_belief_state(initial_state, observed_status)
   for key, value in observed_status.items():
      for edge in state.edges:
         if key == edge.id:
            assert state.edge_probs[key] ==1.0 or state.edge_probs[key] ==0.0
            if state.edge_probs[key] == 1.0:
               assert state.edge_costs[key] == 10e5
            else:
               assert state.edge_costs[key] == edge_costs[key]
         else:
            assert state.edge_probs[key] == edge_probs[key]
            assert state.edge_costs[key] == edge_costs[key]

def test_sctpbase_nav_sense_update_lineargraph():
   # testing on a linear_graph
   start, goal, nodes, edges, robots = graphs.linear_graph_unc()

   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs, 
                     goal=goal, vertices=nodes, edges=edges, robots=robots)

   observed_status = base_navigation.sense(initial_state) # sense the env 
   assert observed_status == {(1, 2): 1.0}
   state, _ = base_navigation.update_belief_state(initial_state, observed_status)
   for key, value in observed_status.items():
      for edge in state.edges:
         if key == edge.id:
            assert state.edge_probs[key] ==1.0 or state.edge_probs[key] ==0.0
            if state.edge_probs[key] == 1.0:
               assert state.edge_costs[key] == 10e5
            else:
               assert state.edge_costs[key] == edge_costs[key]
         else:
            assert state.edge_probs[key] == edge_probs[key]
            assert state.edge_costs[key] == edge_costs[key]

def test_sctpbase_nav_move_sense_update_disjointgraph():
   # testing on a disjoint graph
   start, goal, nodes, edges, robots = graphs.disjoint_unc() # edge 34 is blocked.

   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs, 
                     goal=goal, vertices=nodes, edges=edges, robots=robots)

   state = base_pomdpstate.SCTPBaseState(last_state=initial_state, edge_probs=edge_probs, 
                                    edge_costs=edge_costs, history=initial_state.history)
   nav_cost = 0.0
   observed_status = base_navigation.sense(initial_state) # sense the env 
   assert observed_status == {(1, 2): 0.0, (1, 4): 0.0}
   state, _ = base_navigation.update_belief_state(state, observed_status)
   for key, value in observed_status.items():
      for edge in state.edges:
         if key == edge.id:
            assert state.edge_probs[key] ==1.0 or state.edge_probs[key] ==0.0
            if state.edge_probs[key] == 1.0:
               assert state.edge_costs[key] == 10e5
            else:
               assert state.edge_costs[key] == edge_costs[key]
         else:
            assert state.edge_probs[key] == edge_probs[key]
            assert state.edge_costs[key] == edge_costs[key]
   action, exp_cost = core.po_mcts(state) # apply pomcp algorithm to get the next action
   assert action == 4
   state, move_cost = base_navigation.move(state, action) # move the robot
   nav_cost += move_cost
   assert move_cost == 4.0
   assert state.robots.cur_vertex == 2
   observed_status = base_navigation.sense(state) # sense the env
   assert observed_status == {(2, 3): 1.0}
   state, _ = base_navigation.update_belief_state(state, observed_status)
   for key, value in observed_status.items():
      for edge in state.edges:
         if key == edge.id:
            assert state.edge_probs[key] ==1.0 or state.edge_probs[key] ==0.0
            if state.edge_probs[key] == 1.0:
               assert state.edge_costs[key] == 10e5
            else:
               assert state.edge_costs[key] == edge_costs[key]
         else:
            assert state.edge_probs[key] == edge_probs[key]
            assert state.edge_costs[key] == edge_costs[key]
   action, exp_cost = core.po_mcts(state) # apply pomcp algorithm to get the next action
   assert action == 3
   state, move_cost = base_navigation.move(state, action) # move the robot
   nav_cost += move_cost
   assert move_cost == 4.0
   assert state.robots.cur_vertex == 3
   observed_status = base_navigation.sense(state) # sense the env
   assert observed_status == {(3, 4): 1.0}
   assert nav_cost == 8.0

def test_sctpbase_nav_move_sense_update_sgraph():
   # testing on a disjoint graph
   start, goal, nodes, edges, robots = graphs.s_graph_unc() # edge 34 is blocked.
   for edge in edges:
      edge.block_prob = 0.0
      edge.block_status = 0
   for edge in edges:
      print(f"the edge id is {edge.id} and block probs {edge.block_prob} and status are {edge.block_status} and cost {edge.cost}")
   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs, 
                     goal=goal, vertices=nodes, edges=edges, robots=robots)

   state = base_pomdpstate.SCTPBaseState(last_state=initial_state, edge_probs=edge_probs, 
                                    edge_costs=edge_costs, history=initial_state.history)
   nav_cost = 0.0
   C1 = 40
   assert len(state.get_actions()) == 2
   observed_status = base_navigation.sense(initial_state) # sense the env 
   # assert observed_status == {(1, 2): 0.0, (1, 3): 0.0}
   state, _ = base_navigation.update_belief_state(state, observed_status)
   for key, value in observed_status.items():
      for edge in state.edges:
         if key == edge.id:
            assert state.edge_probs[key] ==1.0 or state.edge_probs[key] ==0.0
            if state.edge_probs[key] == 1.0:
               assert state.edge_costs[key] == 10e5
            else:
               assert state.edge_costs[key] == edge_costs[key]
         else:
            assert state.edge_probs[key] == edge_probs[key]
            assert state.edge_costs[key] == edge_costs[key]
   action, exp_cost = core.po_mcts(state, n_iterations=5000, C=C1) # apply pomcp algorithm to get the next action
   print(f"the action and exp cost are {action} {exp_cost}")
   assert action == 3
   assert exp_cost == pytest.approx(12.06, 0.1)
   
   state, move_cost = base_navigation.move(state, action) # move the robot
   nav_cost += move_cost
   assert move_cost == pytest.approx(3.61, 0.1)
   assert state.robots.cur_vertex == 3
   assert len(state.get_actions()) == 4
   observed_status = base_navigation.sense(state) # sense the env
   # assert observed_status == {(2, 3): 0.0, (3, 4): 1.0, (3, 5): 0.0}
   state, _ = base_navigation.update_belief_state(state, observed_status)
   for key, value in observed_status.items():
      for edge in state.edges:
         if key == edge.id:
            assert state.edge_probs[key] ==1.0 or state.edge_probs[key] ==0.0
            if state.edge_probs[key] == 1.0:
               assert state.edge_costs[key] == 10e5
            else:
               assert state.edge_costs[key] == edge_costs[key]
         else:
            assert state.edge_probs[key] == edge_probs[key]
            assert state.edge_costs[key] == edge_costs[key]
   action, exp_cost = core.po_mcts(state, C=C1) # apply pomcp algorithm to get the next action
   assert action == 5
   # state, move_cost = base_navigation.move(state, action) # move the robot
   # nav_cost += move_cost
   # assert move_cost == 4.0
   # assert state.robots.cur_vertex == 3
   # observed_status = base_navigation.sense(state) # sense the env
   # assert observed_status == {(3, 4): 1.0}
   # assert nav_cost == 8.0


def test_sctpbase_nav_lineargraph():
   # testing on a linear_graph
   start, goal, nodes, edges, robots = graphs.linear_graph_unc()
   edges[0].block_status = 0.0
   edges[0].block_prob = 0.0
   foundPath, exe_path, nav_cost = base_navigation.sctpbase_navigating(nodes, edges, robots, start, goal)
   assert foundPath == True
   assert len(exe_path) == 3
   assert nav_cost == pytest.approx(15.0)

def test_sctpbase_nav_disjoint():
   # testing on a simple disjoint graph
   start, goal, nodes, edges, robots = graphs.disjoint_unc()
   foundPath, exe_path, nav_cost = base_navigation.sctpbase_navigating(nodes, edges, robots, start, goal)
   print(f"the path is {exe_path}")
   assert foundPath == True
   assert len(exe_path) == 3
   assert exe_path[0] == 1
   assert exe_path[1] == 4
   assert exe_path[2] == 3
   assert nav_cost == pytest.approx(8.0)

def test_sctpbase_nav_sgraph():
   # testing on a simple disjoint graph
   start, goal, nodes, edges, robots = graphs.s_graph_unc()
   assert start == 1
   assert goal == 7
   assert len(nodes) == 7
   assert len(edges) == 12
   block_edges = [edge for edge in edges if edge.block_status == 1]
   assert len(block_edges) == 3
   for i in range(len(block_edges)):
      assert block_edges[i].id == (3, 4) or block_edges[i].id == (5, 7) or block_edges[i].id == (6, 7)
   foundPath, exe_path, nav_cost = base_navigation.sctpbase_navigating(nodes, edges, robots, start, goal)
   print(f"the path is {exe_path} with navigation cost {nav_cost}" )
   assert foundPath == True
   # assert len(exe_path) == 5
   # assert exe_path[0] == 1
   # assert exe_path[1] == 3
   # assert exe_path[2] == 5
   # assert exe_path[3] == 4
   # assert exe_path[4] == 7
   # assert nav_cost == pytest.approx(20.0)
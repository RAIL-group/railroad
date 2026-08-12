import copy
from basesctp import base_pomdpstate
from pouct_planner import core   

def sctpbase_navigating(nodes, edges, robots, start, goal):
   exe_path = [start]
   nav_cost = 0.0
   count = 0
   edge_probs = {edge.id: edge.block_prob for edge in edges}
   edge_costs = {edge.id: edge.cost for edge in edges}
   initial_state = base_pomdpstate.SCTPBaseState(edge_probs=edge_probs, edge_costs=edge_costs, 
                     goal=goal, vertices=nodes, edges=edges, robots=robots)
   state = base_pomdpstate.SCTPBaseState(last_state=initial_state, edge_probs=edge_probs, 
                                    edge_costs=edge_costs, history=initial_state.history)
   observed_status = sense(state) # move then sense
   state, _ = update_belief_state(state, observed_status)
   while True:
      if state.is_goal_state:
         return True, exe_path, nav_cost

      action, exp_cost = core.po_mcts(state) # apply pomcp algorithm to get the next action
      exe_path.append(action)
      state, move_cost = move(state, action) # move the robot
      nav_cost += move_cost
      state, new_obser = update_belief_state(state, observed_status)
      count += 1
      if count > 10:
         break 
   return False, exe_path, nav_cost 

def move(state, action):
   cost = 0.0
   if action is None:
      raise ValueError("Action cannot be None")
   
   edge_id = tuple(sorted((state.robots.cur_vertex, action)))
   # edge = [edge for edge in state.edges if edge.id == edge_id][0]
   cost = state.edge_costs[edge_id] # edge.get_cost()
   state.robot_move(action)
   
   return state, cost

def sense(state):
   neighbors = [node for node in state.vertices if node.id == state.robots.cur_vertex][0].neighbors
   observed_status = {}
   for n in neighbors:
      e_id = tuple(sorted((state.robots.cur_vertex, n)))
      next_edge = [edge for edge in state.edges if edge.id == e_id][0]
      if state.edge_probs[e_id] != 1.0 and state.edge_probs[e_id] != 0.0:
         observed_status[e_id] = float(next_edge.block_status)
   return observed_status

def update_belief_state(state, observed_status):
   need_update = False
   for key, value in observed_status.items():
      if state.edge_probs[key] != value:
         need_update = True
      state.edge_probs[key] = value
      if state.edge_probs[key] == 1.0:
         state.edge_costs[key] = 10e5
   return state, need_update

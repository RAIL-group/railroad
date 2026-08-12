from enum import Enum
from basesctp import base_graphs as graphs
from sctp import param
import numpy as np


class Action(object):
    def __init__(self, start_node, target_node):
        self.start = start_node
        self.end = target_node
    def __eq__(self, other):
        return self.start == other.start and self.end == other.end
    def __hash__(self):
        return hash((self.start, self.end))
    def __str__(self):
        return f'Action({self.start} -> {self.end})'
class History(object):
    def __init__(self, data=None):
        self._data = data if data is not None else dict()

    def add_history(self, action, outcome):
        assert outcome == param.EventOutcome.TRAV or outcome == param.EventOutcome.BLOCK
        self._data[action] = outcome
        # the outcome is the same for the inverse action
        invert_action = Action(start_node=action.end, target_node=action.start)
        self._data[invert_action] = outcome


    def get_action_outcome(self, action):
        # return the history or, it it doesn't exist, return CHANCE
        return self._data.get(action, param.EventOutcome.CHANCE)

    def copy(self):
        return History(data=self._data.copy())
    
    def get_data_length(self):
        return len(self._data)

    def __eq__(self, other):
        if not isinstance(other, History):
            return False
        return self._data == other._data

    def __str__(self):
        return f'{self._data}'

    def __hash__(self):
        return hash(tuple(self._data.items()))

def get_state_from_history(outcome_states, history):
    for state in outcome_states:
        if state.history == history:
            return state


class SCTPBaseState(object):
    def __init__(self, graph=None, last_state=None, history=None, 
                goal=None, robots=None):
        self.action_cost = 0.0
        if history is None:
            self.history = History()
        else:
            self.history = history
        if last_state is None:
            self.edge_probs = {edge: edge.block_prob for edge in graph.edges}
            self.edge_costs = {edge: edge.cost for edge in graph.edges}
            self.vertices = graph.vertices
            self.edges = graph.edges
            self.goalID = goal
            self.robots = robots
        else:
            self.edge_probs = last_state.edge_probs
            self.edge_costs = last_state.edge_costs
            self.vertices = last_state.vertices
            self.edges = last_state.edges
            self.goalID = last_state.goalID
            self.robots = graphs.RobotData(last_robot=last_state.robots)
        self.heuristic = 0.0
        self.update_heuristic()
        neighbors = [node for node in self.vertices if node.id == self.robots.cur_vertex][0].neighbors
        self.actions = [Action(start_node=self.robots.cur_vertex, target_node=neighbor)
                        for neighbor in neighbors]
        self.depth = 0
        self.max_depth = 10
    def get_actions(self):
        return self.actions
    
    def update_heuristic(self):
        for node in self.vertices:
            if node.id == self.robots.cur_vertex:
                self.heuristic = node.heur2goal
                break

    @property
    def is_goal_state(self):
        return self.robots.cur_vertex == self.goalID


    def transition(self, action, nav=False):
        return advance_state(self, action)


    def robot_move(self, action):
        next_vertex = action.end
        node = [node for node in self.vertices if node.id == next_vertex][0]
        
        self.robots.last_vertex = self.robots.cur_vertex
        self.robots.cur_vertex = next_vertex
        self.robots.position = [node.coord[0], node.coord[1]]
        self.update_heuristic()
        neighbors = node.neighbors
        self.actions = [Action(start_node=self.robots.cur_vertex, target_node=neighbor)
                        for neighbor in neighbors]
        
    def hash_graph(self):
        edges_tuple = tuple(self.edge_probs)
        # Compute the hash
        return hash(edges_tuple)

    def hash_robot(self):
        return hash(self.robots.cur_vertex)

    def hash_state(self):
        graph_hash = self.hash_graph()
        robot_hash = self.hash_robot()
        # Combine the two hashes
        combined_hash = hash((graph_hash, robot_hash, self.history))
        return combined_hash

    def __hash__(self):
        self.hash_id = self.hash_state()
        return self.hash_id

    def __eq__(self, other):
        return self.hash_id == other.hash_id

   # def __repr__(self):
   #    return f'{self.edge_probs, self.robots.cur_vertex}'

def advance_state(state, action):
    edge_status = state.history.get_action_outcome(action)
    edge = get_edge_from_action(state, action)
    block_prob = state.edge_probs[edge]
    # if edge_status is blocked, return action blocked (state) with blocked cost
    if edge_status == param.EventOutcome.BLOCK or block_prob == 1.0:
        if edge_status == param.EventOutcome.BLOCK:
            new_state_block = SCTPBaseState(last_state=state, history=state.history)
        else:
            block_history = state.history.copy()
            block_history.add_history(action, param.EventOutcome.BLOCK)
            new_state_block = SCTPBaseState(last_state=state, history=block_history)
        new_state_block.robot_move(action)
        new_state_block.action_cost = param.BLOCK_COST
        return {new_state_block: (1.0, param.BLOCK_COST)}

    # if edge_status is traversable, return action traversable (state) with traversable cost
    elif edge_status == param.EventOutcome.TRAV or block_prob == 0.0:
        if edge_status == param.EventOutcome.TRAV:            
            new_state_trav = SCTPBaseState(last_state=state, history=state.history)
        else:
            trav_history = state.history.copy()
            trav_history.add_history(action, param.EventOutcome.TRAV)
            new_state_trav = SCTPBaseState(last_state=state, history=trav_history)
        new_state_trav.robot_move(action)
        new_state_trav.action_cost = state.edge_costs[edge]
        return {new_state_trav: (1.0, new_state_trav.action_cost)}

    # if edge_status is 'CHANCE', we don't know the outcome.
    elif edge_status == param.EventOutcome.CHANCE:        
        # TRAVERSABLE
        trav_history = state.history.copy()
        trav_history.add_history(action, param.EventOutcome.TRAV)
        new_state_trav = SCTPBaseState(last_state=state, history=trav_history)
        new_state_trav.robot_move(action)
        new_state_trav.action_cost = state.edge_costs[edge]
        # BLOCKED
        block_history = state.history.copy()
        block_history.add_history(action, param.EventOutcome.BLOCK)
        new_state_block = SCTPBaseState(last_state=state, history=block_history)
        new_state_block.robot_move(action)
        new_state_block.action_cost = param.BLOCK_COST

        return {new_state_trav: (1.0-block_prob, new_state_trav.action_cost),
                    new_state_block: (block_prob, param.BLOCK_COST)}

def get_edge_from_action(state, action):
   v1 = [v for v in state.vertices if v.id == action.start][0]
   v2 = [v for v in state.vertices if v.id == action.end][0]
   edge = [e for e in state.edges if (e.hash_id == hash(v1) + hash(v2))][0]
   return edge


def get_action_traversability_from_history(history, action):
   return history.get_action_outcome(action)

def sctpbase_rollout(state):
    cost = 0.0
    while not (state.is_goal_state or len(state.get_actions()) == 0):
        actions = state.get_actions()
        action_cost = [(action, state.transition(action)) for action in actions]
        best_action = min(action_cost, key=lambda x: list(x[1].keys())[0].heuristic)
        node_action_transition = {}
        node_action_transition_cost = {}
        for state, (prob, cost) in best_action[1].items():
            node_action_transition[state] = prob
            node_action_transition_cost[state] = cost
        prob = [p for p in node_action_transition.values()]
        state = np.random.choice(list(node_action_transition.keys()), p=prob)
        cost += node_action_transition_cost[state]
    return cost

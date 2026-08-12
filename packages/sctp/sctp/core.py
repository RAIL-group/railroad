from enum import Enum
import random
import matplotlib.pyplot as plt
from sctp import sctp_graphs  as graphs
from sctp import graph as g
from sctp import param
from sctp.utils import paths, plotting
import numpy as np
from sctp.param import EventOutcome, APPROX_TIME, STUCK_COST, RobotType, REVISIT_PEN, VEL_RATIO


class Action(object):
    def __init__(self, target, rtype=RobotType.Ground, start_pose = (0.0,0.0)):
        self.target = target # vertex idsub
        self.rtype=rtype
        self.start_pose = start_pose
    def update_pose(self, pose):
        self.start_pose = pose
    def __eq__(self, other):
        return self.target == other.target
    def __hash__(self):
        return hash(self.target)
    def __str__(self):
        if self.rtype == RobotType.Ground:
            return f'Robot goes from ({self.start_pose[0]:.2f}, {self.start_pose[1]:.2f}) to V{self.target}'
        return f'Drone goes from ({self.start_pose[0]:.2f}, {self.start_pose[1]:.2f}) to V{self.target}'

class History(object):
    def __init__(self, data=None):
        self._data = data if data is not None else dict()

    def add_history(self, action, outcome):
        assert outcome == EventOutcome.TRAV or outcome == EventOutcome.BLOCK
        self._data[action] = outcome

    def get_action_outcome(self, action):
        # return the history or, it it doesn't exist, return CHANCE
        return self._data.get(action, EventOutcome.CHANCE)
    def copy(self):
        return History(data=self._data.copy())
    def get_data_length(self):
        return len(self._data)
    
    def get_data(self):
        return self._data

    def __eq__(self, other):
        if not isinstance(other, History):
            return False
        return self._data == other._data

    def __str__(self):
        return f'{self._data}'

    def __hash__(self):
        return hash(tuple(self._data.items()))

def get_edge(edges, v1_id, v2_id):
    for edge in edges:
        if (edge.v1.id == v1_id and edge.v2.id == v2_id) or (edge.v1.id == v2_id and edge.v2.id == v1_id):
            return edge
    raise ValueError(f'Edge between {v1_id} and {v2_id} not found')
    

class SCTPState(object):
    def __init__(self, graph=None, goalID=None, robot=None, drones=[], iscopy=False, n_maps=100):
        self.action_cost = 0.0
        self.heuristic = -1.0
        self.noway2goal = False
        self.depth = 0
        self.vertices_map = dict() # map vertex id to vertex object
        self.n_samples = n_maps
        self.uav_action_values = dict() # map action to its value
              
        if not iscopy:
            self.graph = graph
            self.goalID = goalID
            self.history = History()
            self.vertices_map = {v.id: v for v in self.graph.vertices + self.graph.pois}
            for vertex in graph.vertices+graph.pois:
                action = Action(target=vertex.id)
                if vertex.block_prob == 1.0:
                    self.history.add_history(action, EventOutcome.BLOCK)
                elif vertex.block_prob == 0.0:
                    self.history.add_history(action, EventOutcome.TRAV)
            self.assigned_pois = set()
            self.v_vertices = dict()
            self.heuristic_vertices = dict()
            # define robot
            self.robot = robot
            self.robot.need_action = True
            if self.robot.at_node:
                neighbors = [node for node in self.graph.vertices+self.graph.pois if node.id == self.robot.last_node][0].neighbors
                self.robot_actions = [Action(target=neighbor, start_pose=self.robot.cur_pose) for neighbor in neighbors]
                self.v_vertices[self.robot.last_node] = 1
                if self.history.get_action_outcome(Action(target=self.robot.last_node))==EventOutcome.BLOCK:
                    self.robot_actions = [action for action in self.robot_actions if action.target ==self.robot.pl_vertex]                
            else:
                self.robot_actions = [Action(target=self.robot.edge[0], start_pose=self.robot.cur_pose), 
                                      Action(target=self.robot.edge[1],start_pose=self.robot.cur_pose)]
            
            self.robot_actions = [action for action in self.robot_actions \
                                  if self.history.get_action_outcome(action) != EventOutcome.BLOCK]
            assert len(self.robot_actions) > 0
            self.state_actions = [action for action in self.robot_actions ]
            self.update_heuristic2()
            # define drones
            self.uavs = drones 
            self.uav_actions = []
            if self.uavs != []:
                self.uav_actions = [Action(target=poi.id, rtype=RobotType.Drone) for poi in self.graph.pois]                
                self.uav_actions = [action for action in self.uav_actions \
                                    if self.history.get_action_outcome(action) == EventOutcome.CHANCE] # list unexplored pois
                continue_actions = []
                for uav in self.uavs:
                    if uav.unfinished_action is not None and uav.unfinished_action in self.uav_actions:
                        uav.action = uav.unfinished_action
                        continue_actions.append(uav.unfinished_action)
                        uav.unfinished_action = None
                        uav.need_action = False
                        distance, direction = self.get_distance_direction(uav.cur_pose, uav.action.target)
                        uav.action.update_pose((uav.cur_pose[0], uav.cur_pose[1]))
                        uav.retarget(uav.action, distance, direction)
                        # print(f"THe uav {uav.id} has targeted to POI: {uav.action.target}")
                    else:
                        uav.need_action = True
                
                if param.ADD_IV:
                    # print("Something wrong - it should not be here")
                    for act in self.uav_actions: # only for value information gain 118-127
                        if act in continue_actions:
                            continue
                        act_value = get_action_value(graph=self.graph, action=act, robot_edge=[self.robot.last_node, self.robot.last_node],
                                                    d0=0.0, d1=0.0, goalID=self.goalID, atNode=self.robot.at_node, drone_pose=self.robot.cur_pose,
                                                    cur_heuristic=self.heuristic, n_samples=param.IV_SAMPLE_SIZE)
                        self.uav_action_values[act] = act_value
                    self.uav_action_values = dict(sorted(self.uav_action_values.items(), key=lambda item: item[1], reverse=True))
                    self.uav_actions = list(self.uav_action_values.keys())[:min(param.MAX_UAV_ACTION, len(self.uav_action_values))]
                    
                    for _ in range(min(param.MAX_UAV_ACTION, len(self.uav_action_values))):
                        first_key = next(iter(self.uav_action_values))
                        self.uav_action_values.pop(first_key)
                    
                if len(self.uav_actions) == 0:
                    # for uav_index in range(len(self.uavs)): # need consider as having more drones
                    self.uav_actions = [Action(target=self.goalID, rtype=RobotType.Drone, 
                                    start_pose = (self.uavs[0].cur_pose[0], self.uavs[0].cur_pose[1]))]
        
                assert isinstance(self.uav_actions[0], Action)
                assert len(self.uav_actions) > 0
                if any([uav.need_action for uav in self.uavs]):
                    self.state_actions = [action for action in self.uav_actions]
            stuck = is_robot_stuck(self)
            
    def get_actions(self):
        return self.state_actions

    def update_heuristic2(self):
        redge = [self.robot.last_node, self.robot.pl_vertex]
        block_pois = [key.target for key, value in self.history.get_data().items() if value == EventOutcome.BLOCK]
        new_graph = g.modify_graph(graph=self.graph, robot_edge=redge, poiIDs=block_pois)        
        min_dist1, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[0], goal=self.goalID)
        min_dist2, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[1], goal=self.goalID)
        assert (min_dist1 < 0) == (min_dist2 < 0)
        if min_dist1 < 0.0 and min_dist2 < 0.0:
            self.heuristic = STUCK_COST
            return self.heuristic
        d2 = np.linalg.norm(np.array(self.robot.cur_pose)-np.array(self.vertices_map[redge[1]].coord))
        d1 = np.linalg.norm(np.array(self.robot.cur_pose)-np.array(self.vertices_map[redge[0]].coord))
        self.heuristic = sampling_rollout(new_graph, redge, d1, d2, self.goalID, self.robot.at_node, 
                                          startNode=self.robot.last_node, n_maps=self.n_samples)        
        return self.heuristic
        
    def update_uav_actionvalue(self):
        redge = [self.robot.last_node, self.robot.pl_vertex]
        block_pois = [key.target for key, value in self.history.get_data().items() if value == EventOutcome.BLOCK]
        new_graph = g.modify_graph(graph=self.graph, robot_edge=redge, poiIDs=block_pois)        
        min_dist1, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[0], goal=self.goalID)
        min_dist2, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[1], goal=self.goalID)
        assert (min_dist1 < 0) == (min_dist2 < 0)
        uav_actions_left = [Action(target=poi.id, rtype=RobotType.Drone) for poi in new_graph.pois]
        uav_actions_left = [action for action in uav_actions_left if self.history.get_action_outcome(action) == EventOutcome.CHANCE]
        self.uav_action_values.clear()
        for action in uav_actions_left:
            action_value = get_action_value(graph=new_graph, action=action, robot_edge=redge,
                                            d0=min_dist1, d1=min_dist2, goalID=self.goalID, atNode=self.robot.at_node,
                                            drone_pose=self.robot.cur_pose, cur_heuristic=self.heuristic,
                                            n_samples=param.IV_SAMPLE_SIZE)
            self.uav_action_values[action] = action_value
        for action in self.uav_actions:
            if action in self.uav_action_values:
                del self.uav_action_values[action]
        
    @property
    def is_goal_state(self):
        return self.robot.last_node == self.goalID or self.noway2goal

    @property
    def is_block_state(self):
        return self.noway2goal

    def transition(self, action):
        temp_state = self.copy()
        anyrobot_action = False        
        if action.rtype == RobotType.Drone:
            uav_needs_action = [uav.need_action for uav in temp_state.uavs]
            assert any(uav_needs_action) == True
            assert action in temp_state.uav_actions
            uav_idx = uav_needs_action.index(True)
            start_pos = temp_state.uavs[uav_idx].cur_pose
            if np.isnan(start_pos[0]) or np.isnan(start_pos[1]):
                ValueError("Start position is NaN") 
            action.update_pose(start_pos)
            distance, direction = temp_state.get_distance_direction(start_pos, action.target)            
            # print(f"state depth is: {self.depth} with distance {distance} and direction {direction}")
            # print(f"the time remaining is: {self.uavs[uav_idx].remaining_time} -- {temp_state.uavs[uav_idx].remaining_time}")
            temp_state.uavs[uav_idx].retarget(action, distance, direction)
            # print(f"state depth is: {self.depth} =for checking")
            if action.target not in temp_state.assigned_pois:
                temp_state.assigned_pois.add(action.target)
            anyrobot_action = True
        elif action.rtype == RobotType.Ground:
            assert temp_state.robot.need_action == True
            start_pos = temp_state.robot.cur_pose
            action.update_pose(start_pos)
            distance, direction = temp_state.get_distance_direction(start_pos, action.target)
            temp_state.robot.retarget(action, distance, direction)
            anyrobot_action = True
        assert anyrobot_action == True
        return advance_state(temp_state, action)

    def copy(self):
        new_state = SCTPState(iscopy=True)
        new_state.vertices_map = self.vertices_map.copy()
        new_state.depth = self.depth
        new_state.history = self.history.copy()
        new_state.graph = self.graph
        new_state.goalID = self.goalID
        new_state.action_cost = 0.0
        new_state.assigned_pois = self.assigned_pois.copy() # [poi for poi in self.assigned_pois]
        new_state.state_actions = []
        new_state.v_vertices = self.v_vertices.copy()
        new_state.noway2goal = self.noway2goal
        new_state.heuristic = self.heuristic
        # copy the robot
        new_state.robot = self.robot.copy()
        new_state.robot_actions = [Action(target=action.target, rtype=action.rtype, start_pose=action.start_pose) \
                                   for action in self.robot_actions]
        if self.uavs != []:
            new_state.uavs = [uav.copy() for uav in self.uavs]
            new_state.uav_actions = [Action(target=action.target, rtype=action.rtype, start_pose=action.start_pose) \
                                 for action in self.uav_actions]
        else:
            new_state.uavs = []
            new_state.uav_actions = []
        return new_state
            
    def get_distance_direction(self, start_pos, target):
        end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == target][0].coord
        distance = np.linalg.norm(np.array(start_pos) - np.array(end_pos))
        if distance != 0.0:
            direction = (np.array([end_pos[0], end_pos[1]]) - start_pos)/distance
        else:
            direction = np.array([1.0, 1.0])
        return distance, direction

def advance_state(state1, action):
    state = state1.copy()
    # 1. if any robot needs action, determine its actions then return
    if state.uavs != [] and any([uav.need_action for uav in state.uavs]):
        state.state_actions = [action for action in state.uav_actions]
        assert len(state.state_actions) > 0
        state.depth += 1
        return {state: (1.0, 0.0)}
    if state.robot.need_action:
        # set action for this state
        state.robot_actions = [action for action in state.robot_actions \
                            if state.history.get_action_outcome(action) != EventOutcome.BLOCK]
        state.state_actions = [action for action in state.robot_actions]
        state.depth += 1
        assert len(state.state_actions) > 0
        return {state: (1.0, 0.0)}
    
    # 2. Find the robot that finishes its action first.
    robot_reach_first, uav_index, time_advance = _get_robot_that_finishes_first(state)
    assert time_advance >= 0.0
    state.action_cost = time_advance
    # save some data before moving
    last_node = state.robot.last_node
    edge = state.robot.edge.copy()
    # move the robots
    state.robot.advance_time(time_advance)
    for uav in state.uavs:
        if uav.last_node != state.goalID:
            uav.advance_time(time_advance)
        else:
            uav.need_action = False
    if robot_reach_first:
        # print("Ground robot reaches its target first")
        return get_new_nodes_grobot(state, last_node=last_node, last_edge=edge)
    else:
        # print(f"Drone {uav_index} reaches its target first")
        return get_new_nodes_drone(state=state, uav_index=uav_index)
        
def get_new_nodes_grobot(state, last_node, last_edge):
    vertex_status = state.history.get_action_outcome(state.robot.action)
    vertex = [node for node in state.graph.vertices+state.graph.pois if node.id == state.robot.action.target][0]
    state.robot.visited_vertices.append(state.robot.last_node)
    # update the uav action set.
    state.uav_actions = [action for action in state.uav_actions if action.target != state.robot.last_node]
    action = Action(target=state.robot.last_node)
    if param.ADD_IV and len(state.uavs) > 0 and action in state.uav_action_values:
        del state.uav_action_values[action]
    if param.ADD_IV:
        if len(state.uav_actions) == 0 and len(state.uavs) >0 and len(state.uav_action_values) == 0:
            state.uav_actions = [Action(target=state.goalID, rtype=RobotType.Drone)]
    else:
        if len(state.uav_actions) == 0 and len(state.uavs) >0:
            state.uav_actions = [Action(target=state.goalID, rtype=RobotType.Drone)]
    state.v_vertices[state.robot.last_node] = state.v_vertices.get(state.robot.last_node, 0) + 1
    assert state.robot.at_node == True
    if vertex_status == EventOutcome.BLOCK:
        state.robot_actions = [Action(target=last_node, start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1]))]
        state.state_actions = [action for action in state.robot_actions]
        state.depth += 1
        state.update_heuristic2()
        if param.ADD_IV and len(state.uavs) > 0:
            # state.update_uav_actionvalue()
            if len(state.uav_actions) ==0 and len(state.uav_action_values) >0:
                state.uav_actions = [list(state.uav_action_values.keys())[0]]
                state.uav_action_values.pop(state.uav_actions[0])
        return {state: (1.0, state.action_cost)}
    # if edge_status is traversable, return action traversable (state) with traversable cost
    elif vertex_status == EventOutcome.TRAV: 
        cur_node = [node for node in state.graph.vertices+state.graph.pois if node.id == state.robot.last_node][0]
        neighbors = cur_node.neighbors
        if cur_node.id in state.v_vertices:
            state.robot_actions = [Action(target=neighbor, start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1])) \
                                                for neighbor in neighbors]
        else:
            state.robot_actions = [Action(target=neighbor, start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1])) \
                               for neighbor in neighbors if neighbor !=state.robot.pl_vertex]
        state.robot_actions = [action for action in state.robot_actions \
                                    if state.history.get_action_outcome(action) != EventOutcome.BLOCK]
        assert state.robot.last_node == state.robot.action.target
        state.state_actions = [action for action in state.robot_actions ]
        stuck = is_robot_stuck(state)
        state.depth += 1
        # update the cost if revisiting the vertex
        state.action_cost += (state.v_vertices.get(state.robot.last_node, 0)-1) * REVISIT_PEN
        state.update_heuristic2()
        if param.ADD_IV and len(state.uavs) > 0:
            if cur_node in state.graph.vertices: # only reculcate action values if robots is at vertex
                state.update_uav_actionvalue()
            if len(state.uav_actions) ==0 and len(state.uav_action_values) >0:
                state.uav_actions = [list(state.uav_action_values.keys())[0]]
                state.uav_action_values.pop(state.uav_actions[0])
        return {state: (1.0, state.action_cost)}
    # if edge_status is 'CHANCE', we don't know the outcome.action
    elif vertex_status == EventOutcome.CHANCE:
        if len(state.uavs) > 0:
            reset_uavs_action(state)
        # TRAVERSABLE
        new_state_trav = state.copy()
        new_state_trav.action_cost = state.action_cost
        new_state_trav.history.add_history(state.robot.action, EventOutcome.TRAV)
        neighbors = [node for node in state.graph.vertices+state.graph.pois if node.id == state.robot.last_node][0].neighbors
        new_state_trav.robot_actions = [Action(target=neighbor,start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1])) \
                                        for neighbor in neighbors if neighbor != state.robot.pl_vertex]
        stuck = is_robot_stuck(new_state_trav)
        new_state_trav.state_actions = [action for action in new_state_trav.robot_actions]
        new_state_trav.depth += 1
        new_state_trav.update_heuristic2()
        if param.ADD_IV and len(new_state_trav.uav_actions) ==0 and len(new_state_trav.uav_action_values) >0:
                new_state_trav.uav_actions = [list(new_state_trav.uav_action_values.keys())[0]]
                new_state_trav.uav_action_values.pop(new_state_trav.uav_actions[0])
        # BLOCKED
        new_state_block = state.copy()
        new_state_block.action_cost = state.action_cost
        new_state_block.history.add_history(state.robot.action, EventOutcome.BLOCK)
        if last_node != new_state_block.robot.last_node:
            target = last_node
        else:
            target = new_state_block.robot.pl_vertex
        new_state_block.robot_actions = [Action(target=target,start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1]))]
        new_state_block.state_actions = [action for action in new_state_block.robot_actions]
        stuck= is_robot_stuck(new_state_block)
        new_state_block.depth += 1
        new_state_block.update_heuristic2()
        if param.ADD_IV and len(new_state_block.uav_actions) ==0 and len(new_state_block.uav_action_values) >0:
                new_state_block.uav_actions = [list(new_state_block.uav_action_values.keys())[0]]
                new_state_block.uav_action_values.pop(new_state_block.uav_actions[0])
        assert new_state_block.depth == new_state_trav.depth
        return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
                    new_state_block: (vertex.block_prob, new_state_block.action_cost)}

def get_new_nodes_drone(state, uav_index):
    assert len(state.uavs) > 0
    vertex_status = state.history.get_action_outcome(state.uavs[uav_index].action)
    vertex = [node for node in state.graph.pois+state.graph.vertices if node.id == state.uavs[uav_index].action.target][0]
    # determine all actions related to the poi and remove it from the set.
    poi_id = state.uavs[uav_index].last_node
    assert poi_id == vertex.id
    state.uav_actions = [action for action in state.uav_actions if action.target != poi_id]
    action = Action(target=poi_id, rtype=RobotType.Drone)
    # assert action not in state.uav_action_values
        
    if param.ADD_IV: # adding information gain
        if len(state.uav_actions) == 0 and len(state.uav_action_values) == 0:
            state.uav_actions = [Action(target=state.goalID, rtype=RobotType.Drone, 
                                        start_pose = (state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))]
            state.state_actions = [action for action in state.uav_actions]
        elif len(state.uav_actions) > 0:
            for action in state.uav_actions:
                action.update_pose((state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))
            state.state_actions = [action for action in state.uav_actions if action.target not in state.assigned_pois]
    else:
        if len(state.uav_actions) == 0:
            state.uav_actions = [Action(target=state.goalID, rtype=RobotType.Drone, 
                                        start_pose = (state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))]
            state.state_actions = [action for action in state.uav_actions]
        else:
            for action in state.uav_actions:
                action.update_pose((state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))
            state.state_actions = [action for action in state.uav_actions if action.target not in state.assigned_pois]
        
    # if some uavs also finish theirs actions, reset need_action for the next iteration
    for i, uav in enumerate(state.uavs):
        uav.need_action = False if i != uav_index and uav.need_action else True
    # if the robot is at the node, it needs new action, so delay it for the next state/iteration
    if state.robot.at_node:
        state.robot.need_action = False  # I think we don't need this, but just in case
    
    if vertex_status == EventOutcome.BLOCK: # should not go here
        ValueError("Drones should never visit this node")
        assert 1==0
        state.depth += 1
        return {state: (1.0, state.action_cost)}
    elif vertex_status == EventOutcome.TRAV:  # only at goal
        assert vertex.id == state.goalID
        state.depth += 1
        state.update_heuristic2()
        return {state: (1.0, state.action_cost)}
    elif vertex_status == EventOutcome.CHANCE:
        if not state.robot.at_node: # reset if it is in middle of action
            state.robot.need_action = True # you don't want it to go to get_new_nodes_grobot (no node reached)
            state.robot.remaining_time = 0.0
            state.robot_actions = [Action(target=state.robot.edge[0], start_pose = (state.robot.cur_pose[0],state.robot.cur_pose[1])), 
                                   Action(target=state.robot.edge[1], start_pose = (state.robot.cur_pose[0],state.robot.cur_pose[1]))]
        # TRAVERSABLE
        new_state_trav = state.copy()
        new_state_trav.action_cost = state.action_cost
        new_state_trav.history.add_history(state.uavs[uav_index].action, EventOutcome.TRAV)
        stuck = is_robot_stuck(new_state_trav)
        new_state_trav.depth += 1
        new_state_trav.update_heuristic2()
        if param.ADD_IV and len(new_state_trav.uav_actions) ==0 and len(new_state_trav.uav_action_values) >0:
            new_state_trav.uav_actions = [list(new_state_trav.uav_action_values.keys())[0]]
            new_state_trav.uav_action_values.pop(new_state_trav.uav_actions[0])
        new_state_trav.state_actions = [action for action in new_state_trav.uav_actions]
        # BLOCKED
        new_state_block = state.copy()
        new_state_block.action_cost = state.action_cost
        new_state_block.history.add_history(state.uavs[uav_index].action, EventOutcome.BLOCK)
        new_state_block.robot_actions = [action for action in new_state_block.robot_actions \
                            if new_state_block.history.get_action_outcome(action) != EventOutcome.BLOCK]
        stuck = is_robot_stuck(new_state_block)
        new_state_block.depth += 1
        new_state_block.update_heuristic2()
        if param.ADD_IV and len(new_state_block.uav_actions) ==0 and len(new_state_block.uav_action_values) >0:
            new_state_block.uav_actions = [list(new_state_block.uav_action_values.keys())[0]]
            new_state_block.uav_action_values.pop(new_state_block.uav_actions[0])
        new_state_block.state_actions = [action for action in new_state_block.uav_actions]
        assert new_state_trav.depth == new_state_block.depth
        return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
                    new_state_block: (vertex.block_prob, new_state_block.action_cost)}


def is_robot_stuck(state):
    if state.robot.last_node == state.goalID:
        state.noway2goal = False
        return False
    if state.robot.at_node:
        robot_edge = [state.robot.last_node, state.robot.pl_vertex]
    else:
        robot_edge = [state.robot.edge[0],state.robot.edge[1]]
    if (not _is_robot_goal_connected(state.graph, state.history, robot_edge, state.goalID))\
        or (len(state.robot_actions) == 0):
        state.noway2goal = True
        return True
    return False

def reset_uavs_action(state):
    for i, uav in enumerate(state.uavs):
        if not uav.need_action and uav.action not in state.uav_actions and uav.last_node != state.goalID:
            uav.need_action = True 
            uav.remaining_time = 0.0

def _get_robot_that_finishes_first(state):
    time_remaining_uavs = []
    if len(state.uavs) > 0:
        for uav in state.uavs:
            if uav.last_node == state.goalID:
                continue
            if uav.remaining_time >APPROX_TIME:
                time_remaining_uavs.append(uav.remaining_time)
    robot_reach_first = False
    if len(time_remaining_uavs)==0 or state.robot.remaining_time < min(time_remaining_uavs):
        robot_reach_first = True
        return robot_reach_first, len(time_remaining_uavs), state.robot.remaining_time
    else:
        assert len(state.uavs) > 0
        min_time = min(time_remaining_uavs)
        remaining_times = [uav.remaining_time for uav in state.uavs]
        uav_index = remaining_times.index(min_time)
        return robot_reach_first, uav_index, min_time

def _is_robot_goal_connected(graph, history, redge, goalID):
    block_pois = []
    for key, value in history.get_data().items():
        if value == EventOutcome.BLOCK:
            block_pois.append(key.target)
    new_graph = g.modify_graph(graph=graph, robot_edge=redge, poiIDs=block_pois)
    reach1 = paths.is_reachable(graph=new_graph, start=redge[0], goal=goalID)
    reach2 = paths.is_reachable(graph=new_graph, start=redge[1], goal=goalID)
    assert reach1 == reach2
    return reach2

def sctp_rollout3(state):
    if state.is_goal_state and not state.is_block_state:
        return 0.0
    if state.heuristic >= 0.0:
        return state.heuristic
    return state.update_heuristic2()


def sampling_rollout(graph, robot_edge, d0, d1, goalID, atNode, startNode, n_maps=100):
    # noway_penalty = 200.0
    total_cost = 0.0
    for _ in range(n_maps):
        block_pois = [poi.id for poi in graph.pois if random.random() <= poi.block_prob ] 
        modified_graph = g.modify_graph(graph=graph, robot_edge=robot_edge, poiIDs=block_pois)
        if atNode:
            cost, _ = paths.get_shortestPath_cost(modified_graph, startNode, goalID)
            total_cost += cost if (cost >= 0.0) else param.NOWAY_PEN
        else:
            cost0, _ = paths.get_shortestPath_cost(modified_graph, robot_edge[0], goalID)
            cost1, _ = paths.get_shortestPath_cost(modified_graph, robot_edge[1], goalID)
            assert (cost1 < 0) == (cost0 < 0)
            if cost0 >= 0:
                total_cost += (cost0+d0) if (cost0+d0)<(cost1+d1) else (cost1+d1)
            else:
                total_cost += param.NOWAY_PEN
    return total_cost/n_maps

def get_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, 
                     drone_pose, cur_heuristic, n_samples=100):
    dist = np.linalg.norm(np.array(drone_pose)-np.array(graph.get_poi(action.target).coord))/VEL_RATIO
    # value if the action is passable
    block_value = 0.0
    pass_value = 0.0
    num_pois = len(graph.pois)
    num_vertices = len(graph.vertices)
    num_edges = len(graph.edges)
    for _ in range(n_samples):
        assert num_pois == len(graph.pois)
        assert num_vertices == len(graph.vertices)
        assert num_edges == len(graph.edges)
        pass_value += sampling_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, block_edge=False)
        block_value += sampling_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, block_edge=True)
    pass_value /= n_samples
    block_value /= n_samples
    aver_block = graph.get_poi(action.target).block_prob * block_value
    aver_pass = (1-graph.get_poi(action.target).block_prob) * pass_value
    value = aver_block + aver_pass
    bc = cur_heuristic - value
    return bc, bc - dist 


def sampling_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, block_edge=False):
    # noway_penalty = 200.0
    block_pois = [poi.id for poi in graph.pois if poi.id != action.target and random.random() <= poi.block_prob ] 
    if block_edge:
        modified_graph = g.modify_graph(graph=graph, robot_edge=robot_edge, poiIDs=block_pois+[action.target])
    else:
        modified_graph = g.modify_graph(graph=graph, robot_edge=robot_edge, poiIDs=block_pois)
    if atNode:
        cost, _ = paths.get_shortestPath_cost(modified_graph, start=robot_edge[0], goal=goalID)
        return cost if cost >= 0.0 else param.NOWAY_PEN
    else:
        cost0, _ = paths.get_shortestPath_cost(modified_graph, start=robot_edge[0], goal=goalID)
        cost1, _ = paths.get_shortestPath_cost(modified_graph, start=robot_edge[1], goal=goalID)
        assert (cost1 < 0) == (cost0 < 0)
        return min(cost0+d0, cost1+d1) if cost0 >= 0 else param.NOWAY_PEN
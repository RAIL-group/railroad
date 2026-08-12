from enum import Enum
import random
import matplotlib.pyplot as plt
from sctp import sctp_graphs  as graphs
from sctp import graph as g
from sctp.utils import paths, plotting
import numpy as np
from sctp import param, core
    

class SCTPState(object):
    def __init__(self, graph=None, goalIDs=[], robots=[], drones=[], iscopy=False, n_maps=100):
        self.action_cost = 0.0
        self.heuristic = -1.0
        self.noway2goal = False
        self.depth = 0
        self.vertices_map = dict() # map vertex id to vertex object
        self.n_maps = n_maps
        self.uav_action_values = dict() # map action to its value
        self.robots_actions = []
              
        if not iscopy:
            self.graph = graph
            self.goalIDs = goalIDs
            self.history = core.History()
            # self.vertices_map = {v.id: v for v in self.graph.vertices + self.graph.pois}
            self.init_history()
            self.assigned_pois = set()
            self.v_vertices = dict()
            # self.heuristic_vertices = dict()
            # define robot
            self.robots = robots
            for robot in self.robots:
                robot.need_action = True
                if robot.at_node:
                    neighbors = [node for node in self.graph.vertices+self.graph.pois if node.id == robot.last_node][0].neighbors
                    robot_actions = [core.Action(target=neighbor, start_pose=robot.cur_pose) for neighbor in neighbors]
                    self.v_vertices[robot.last_node] = 1
                    if self.history.get_action_outcome(core.Action(target=robot.last_node))==param.EventOutcome.BLOCK:
                        robot_actions = [action for action in robot_actions if action.target ==robot.pl_vertex]                
                else:
                    robot_actions = [core.Action(target=robot.edge[0], start_pose=robot.cur_pose), 
                                          core.Action(target=robot.edge[1],start_pose=robot.cur_pose)]
                robot_actions = [action for action in robot_actions \
                                  if self.history.get_action_outcome(action) != core.EventOutcome.BLOCK]
                assert len(robot_actions) > 0
                self.robots_actions.append(robot_actions)
            self.state_actions = [action for action in self.robots_actions[0]]
            self.update_heuristic()
            # define drones
            self.uavs = drones 
            self.uav_actions = []
            if self.uavs != []:
                self.uav_actions = [core.Action(target=poi.id, rtype=param.RobotType.Drone) for poi in self.graph.pois]                
                self.uav_actions = [action for action in self.uav_actions \
                                    if self.history.get_action_outcome(action) == param.EventOutcome.CHANCE] # list unexplored pois
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
                        self.assigned_pois.add(uav.action.target)
                        self.uav_actions.remove(uav.action)
                    else:
                        uav.need_action = True
                
            #     if param.ADD_IV:
            #         # print("Something wrong - it should not be here")
            #         for act in self.uav_actions: # only for value information gain 118-127
            #             if act in continue_actions:
            #                 continue
            #             act_value = get_action_value(graph=self.graph, action=act, robot_edge=[self.robot.last_node, self.robot.last_node],
            #                                         d0=0.0, d1=0.0, goalID=self.goalID, atNode=self.robot.at_node, drone_pose=self.robot.cur_pose,
            #                                         cur_heuristic=self.heuristic, n_samples=param.IV_SAMPLE_SIZE)
            #             self.uav_action_values[act] = act_value
            #         self.uav_action_values = dict(sorted(self.uav_action_values.items(), key=lambda item: item[1], reverse=True))
            #         self.uav_actions = list(self.uav_action_values.keys())[:min(param.MAX_UAV_ACTION, len(self.uav_action_values))]
                    
            #         for _ in range(min(param.MAX_UAV_ACTION, len(self.uav_action_values))):
            #             first_key = next(iter(self.uav_action_values))
            #             self.uav_action_values.pop(first_key)
                    
                if len(self.uav_actions) == 0:
                    self.uav_actions = [core.Action(target=self.goalIDs[0], rtype=param.RobotType.Drone)]
        
                assert isinstance(self.uav_actions[0], core.Action)
                assert len(self.uav_actions) > 0
                if any([uav.need_action for uav in self.uavs]):
                    self.state_actions = [action for action in self.uav_actions]
            self.noway2goal = are_robots_stuck(self)
            
    def init_history(self):
        for vertex in self.graph.vertices+self.graph.pois:
            action = core.Action(target=vertex.id)
            if vertex.block_prob == 1.0:
                self.history.add_history(action, param.EventOutcome.BLOCK)
            elif vertex.block_prob == 0.0:
                self.history.add_history(action, param.EventOutcome.TRAV)
                
    def get_actions(self):
        return self.state_actions

    def copy(self):
        new_state = SCTPState(iscopy=True)
        new_state.depth = self.depth
        new_state.history = self.history.copy()
        new_state.graph = self.graph
        new_state.goalIDs = self.goalIDs
        new_state.action_cost = 0.0
        new_state.assigned_pois = self.assigned_pois.copy() # [poi for poi in self.assigned_pois]
        new_state.state_actions = []
        new_state.v_vertices = self.v_vertices.copy()
        new_state.noway2goal = self.noway2goal
        new_state.heuristic = self.heuristic
        # copy the robot
        new_state.robots = [robot.copy() for robot in self.robots]
        
        new_state.robot_actions = []
        for i, actions in enumerate(self.robots_actions):
            if self.robots[i].last_node == self.goalIDs[i]:
                new_state.robots_actions.append([core.Action(target=self.goalIDs[i], start_pose=self.robots[i].cur_pose)])
            else:
                acts = []
                for act in actions:
                    acts.append(core.Action(target=act.target, rtype=act.rtype, start_pose=self.robots[i].cur_pose))
                new_state.robot_actions.append(acts)
                
        if self.uavs != []:
            new_state.uavs = [uav.copy() for uav in self.uavs]
            new_state.uav_actions = [core.Action(target=action.target, rtype=action.rtype) \
                                 for action in self.uav_actions]
        else:
            new_state.uavs = []
            new_state.uav_actions = []
        return new_state

    def update_heuristic(self):
        block_pois = [key.target for key, value in self.history.get_data().items() if value == core.EventOutcome.BLOCK]
        # we need to modify the graph first
        all_edges = set()
        edges = []
        for robot in self.robots:
            all_edges.add(robot.last_node)
            all_edges.add(robot.pl_vertex)
            if robot.last_node in block_pois or robot.pl_vertex in block_pois:
                edges.append([robot.last_node, robot.pl_vertex])
        new_pois = [poi for poi in block_pois if poi not in all_edges]        
        new_graph = g.remove_pois(self.graph, new_pois)
        new_graph = g.remove_edges(new_graph, edges)
        for i, robot in enumerate(self.robots):
            redge = [robot.last_node, robot.pl_vertex]
            
            d2 = np.linalg.norm(np.array(robot.cur_pose)-np.array(self.vertices_map[redge[1]].coord))
            d1 = np.linalg.norm(np.array(robot.cur_pose)-np.array(self.vertices_map[redge[0]].coord))
            self.heuristic += core.sampling_rollout(new_graph, redge, d1, d2, self.goalID[i], robot.at_node,
                                                startNode=robot.last_node, n_samples=self.n_maps)        
        return self.heuristic
        
    def update_uav_actionvalue(self):
        pass
        # redge = [self.robot.last_node, self.robot.pl_vertex]
        # block_pois = [key.target for key, value in self.history.get_data().items() if value == EventOutcome.BLOCK]
        # new_graph = g.modify_graph(graph=self.graph, robot_edge=redge, poiIDs=block_pois)        
        # min_dist1, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[0], goal=self.goalID)
        # min_dist2, _ = paths.get_shortestPath_cost(graph=new_graph, start=redge[1], goal=self.goalID)
        # assert (min_dist1 < 0) == (min_dist2 < 0)
        # uav_actions_left = [Action(target=poi.id, rtype=RobotType.Drone) for poi in new_graph.pois]
        # uav_actions_left = [action for action in uav_actions_left if self.history.get_action_outcome(action) == EventOutcome.CHANCE]
        # self.uav_action_values.clear()
        # for action in uav_actions_left:
        #     action_value = get_action_value(graph=new_graph, action=action, robot_edge=redge,
        #                                     d0=min_dist1, d1=min_dist2, goalID=self.goalID, atNode=self.robot.at_node,
        #                                     drone_pose=self.robot.cur_pose, cur_heuristic=self.heuristic,
        #                                     n_samples=param.IV_SAMPLE_SIZE)
        #     self.uav_action_values[action] = action_value
        # for action in self.uav_actions:
        #     if action in self.uav_action_values:
        #         del self.uav_action_values[action]
        
    @property
    def is_goal_state(self):
        return all([robot.last_node == self.goalIDs[i] for i, robot in enumerate(self.robots)]) or self.noway2goal

    @property
    def is_block_state(self):
        return self.noway2goal

    def get_distance_direction(self, start_pos, target):
        end_pos = [node for node in self.graph.vertices+self.graph.pois if node.id == target][0].coord
        distance = np.linalg.norm(np.array(start_pos) - np.array(end_pos))
        if distance != 0.0:
            direction = (np.array([end_pos[0], end_pos[1]]) - start_pos)/distance
        else:
            direction = np.array([1.0, 1.0])
        return distance, direction

    def transition(self, action):
        temp_state = self.copy()
        anyrobot_action = False        
        if action.rtype == param.RobotType.Drone:
            uav_needs_action = [uav.need_action for uav in temp_state.uavs]
            assert any(uav_needs_action) == True
            assert action in temp_state.uav_actions
            if action.target != temp_state.goalIDs[0]:
                assert action.target not in temp_state.assigned_pois
            uav_idx = uav_needs_action.index(True)
            start_pos = temp_state.uavs[uav_idx].cur_pose
            if np.isnan(start_pos[0]) or np.isnan(start_pos[1]):
                ValueError("Start position is NaN") 
            action.update_pose(start_pos)
            distance, direction = temp_state.get_distance_direction(start_pos, action.target)            
            temp_state.uavs[uav_idx].retarget(action, distance, direction)            
            temp_state.assigned_pois.add(action.target)
            anyrobot_action = True
        elif action.rtype == param.RobotType.Ground:
            robots_need_action = [robot.need_action for robot in temp_state.robots]
            assert any(robots_need_action) == True
            robot_idx = robots_need_action.index(True)
            start_pos = temp_state.robots[robot_idx].cur_pose
            action.update_pose(start_pos)
            distance, direction = temp_state.get_distance_direction(start_pos, action.target)
            temp_state.robot.retarget(action, distance, direction)
            anyrobot_action = True
        assert anyrobot_action == True
        return advance_state(temp_state, action)     

def advance_state(state1):
    state = state1.copy()
    # 1. if any robot needs action, determine its actions then return
    if state.uavs != [] and any([uav.need_action for uav in state.uavs]):
        state.state_actions = [action for action in state.uav_actions]
        assert len(state.state_actions) > 0
        state.depth += 1
        return {state: (1.0, 0.0)}
    
    robots_need_action = [robot.need_action for robot in state.robots]
    if any(robots_need_action):
        # set action for this state        
        robot_idx = robots_need_action.index(True)
        state.robots_actions[robot_idx] = [action for action in state.robot_actions[robot_idx] \
                            if state.history.get_action_outcome(action) != param.EventOutcome.BLOCK]
        state.state_actions = [action for action in state.robots_actions[robot_idx]]
        state.depth += 1
        assert len(state.state_actions) > 0
        return {state: (1.0, 0.0)}
    
    # 2. Find the robot that finishes its action first.
    robot_reach_first, rd_index, time_advance = _get_robot_that_finishes_first(state)
    assert time_advance >= 0.0
    state.action_cost = time_advance
    # save some data before moving
    last_nodes = [robot.last_node for robot in state.robots]
    edges = [robot.edge for robot in state.robots]
    # move the robots
    for i, robot in enumerate(state.robots):
        if robot.last_node != state.goalIDs[i]:
            robot.advance_time(time_advance)
        else:
            robot.need_action = False
    for uav in state.uavs:
        if uav.last_node != state.goalIDs[0]:
            uav.advance_time(time_advance)
        else:
            uav.need_action = False
    if robot_reach_first:
        return get_new_nodes_grobot(state, last_nodes=last_nodes, robot_idx = rd_index, last_edges=edges)
    else:
        return get_new_nodes_drone(state=state, uav_index=rd_index)
        
def get_new_nodes_grobot(state, last_nodes, robot_idx, last_edges):
    pass
    # vertex_status = state.history.get_action_outcome(state.robot.action)
    # vertex = [node for node in state.graph.vertices+state.graph.pois if node.id == state.robot.action.target][0]
    # state.robot.visited_vertices.append(state.robot.last_node)
    # # update the uav action set.
    # state.uav_actions = [action for action in state.uav_actions if action.target != state.robot.last_node]
    # action = Action(target=state.robot.last_node)
    # if param.ADD_IV and len(state.uavs) > 0 and action in state.uav_action_values:
    #     del state.uav_action_values[action]
    # if param.ADD_IV:
    #     if len(state.uav_actions) == 0 and len(state.uavs) >0 and len(state.uav_action_values) == 0:
    #         state.uav_actions = [Action(target=state.goalID, rtype=RobotType.Drone)]
    # else:
    #     if len(state.uav_actions) == 0 and len(state.uavs) >0:
    #         state.uav_actions = [Action(target=state.goalID, rtype=RobotType.Drone)]
    # state.v_vertices[state.robot.last_node] = state.v_vertices.get(state.robot.last_node, 0) + 1
    # assert state.robot.at_node == True
    # if vertex_status == EventOutcome.BLOCK:
    #     state.robot_actions = [Action(target=last_node, start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1]))]
    #     state.state_actions = [action for action in state.robot_actions]
    #     state.depth += 1
    #     state.update_heuristic2()
    #     if param.ADD_IV and len(state.uavs) > 0:
    #         # state.update_uav_actionvalue()
    #         if len(state.uav_actions) ==0 and len(state.uav_action_values) >0:
    #             state.uav_actions = [list(state.uav_action_values.keys())[0]]
    #             state.uav_action_values.pop(state.uav_actions[0])
    #     return {state: (1.0, state.action_cost)}
    # # if edge_status is traversable, return action traversable (state) with traversable cost
    # elif vertex_status == EventOutcome.TRAV: 
    #     cur_node = [node for node in state.graph.vertices+state.graph.pois if node.id == state.robot.last_node][0]
    #     neighbors = cur_node.neighbors
    #     if cur_node.id in state.v_vertices:
    #         state.robot_actions = [Action(target=neighbor, start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1])) \
    #                                             for neighbor in neighbors]
    #     else:
    #         state.robot_actions = [Action(target=neighbor, start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1])) \
    #                            for neighbor in neighbors if neighbor !=state.robot.pl_vertex]
    #     state.robot_actions = [action for action in state.robot_actions \
    #                                 if state.history.get_action_outcome(action) != EventOutcome.BLOCK]
    #     assert state.robot.last_node == state.robot.action.target
    #     state.state_actions = [action for action in state.robot_actions ]
    #     stuck = is_robot_stuck(state)
    #     state.depth += 1
    #     # update the cost if revisiting the vertex
    #     state.action_cost += (state.v_vertices.get(state.robot.last_node, 0)-1) * REVISIT_PEN
    #     state.update_heuristic2()
    #     if param.ADD_IV and len(state.uavs) > 0:
    #         if cur_node in state.graph.vertices: # only reculcate action values if robots is at vertex
    #             state.update_uav_actionvalue()
    #         if len(state.uav_actions) ==0 and len(state.uav_action_values) >0:
    #             state.uav_actions = [list(state.uav_action_values.keys())[0]]
    #             state.uav_action_values.pop(state.uav_actions[0])
    #     return {state: (1.0, state.action_cost)}
    # # if edge_status is 'CHANCE', we don't know the outcome.action
    # elif vertex_status == EventOutcome.CHANCE:
    #     if len(state.uavs) > 0:
    #         reset_uavs_action(state)
    #     # TRAVERSABLE
    #     new_state_trav = state.copy()
    #     new_state_trav.action_cost = state.action_cost
    #     new_state_trav.history.add_history(state.robot.action, EventOutcome.TRAV)
    #     neighbors = [node for node in state.graph.vertices+state.graph.pois if node.id == state.robot.last_node][0].neighbors
    #     new_state_trav.robot_actions = [Action(target=neighbor,start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1])) \
    #                                     for neighbor in neighbors if neighbor != state.robot.pl_vertex]
    #     stuck = is_robot_stuck(new_state_trav)
    #     new_state_trav.state_actions = [action for action in new_state_trav.robot_actions]
    #     new_state_trav.depth += 1
    #     new_state_trav.update_heuristic2()
    #     if param.ADD_IV and len(new_state_trav.uav_actions) ==0 and len(new_state_trav.uav_action_values) >0:
    #             new_state_trav.uav_actions = [list(new_state_trav.uav_action_values.keys())[0]]
    #             new_state_trav.uav_action_values.pop(new_state_trav.uav_actions[0])
    #     # BLOCKED
    #     new_state_block = state.copy()
    #     new_state_block.action_cost = state.action_cost
    #     new_state_block.history.add_history(state.robot.action, EventOutcome.BLOCK)
    #     if last_node != new_state_block.robot.last_node:
    #         target = last_node
    #     else:
    #         target = new_state_block.robot.pl_vertex
    #     new_state_block.robot_actions = [Action(target=target,start_pose=(state.robot.cur_pose[0],state.robot.cur_pose[1]))]
    #     new_state_block.state_actions = [action for action in new_state_block.robot_actions]
    #     stuck= is_robot_stuck(new_state_block)
    #     new_state_block.depth += 1
    #     new_state_block.update_heuristic2()
    #     if param.ADD_IV and len(new_state_block.uav_actions) ==0 and len(new_state_block.uav_action_values) >0:
    #             new_state_block.uav_actions = [list(new_state_block.uav_action_values.keys())[0]]
    #             new_state_block.uav_action_values.pop(new_state_block.uav_actions[0])
    #     assert new_state_block.depth == new_state_trav.depth
    #     return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
    #                 new_state_block: (vertex.block_prob, new_state_block.action_cost)}

def get_new_nodes_drone(state, uav_index):
    assert len(state.uavs) > 0
    vertex_status = state.history.get_action_outcome(state.uavs[uav_index].action)
    vertex = [node for node in state.graph.pois+state.graph.vertices if node.id == state.uavs[uav_index].action.target][0]
    # determine all actions related to the poi and remove it from the set.
    poi_id = state.uavs[uav_index].last_node
    assert poi_id == vertex.id
    state.uav_actions = [action for action in state.uav_actions if action.target != poi_id]
    using_uav_action_values(state, uav_index) # if using action values, update the uav actions    
    # if some uavs also finish theirs actions, reset need_action for the next iteration
    for i, uav in enumerate(state.uavs):
        uav.need_action = False if i != uav_index and uav.need_action else True
    # if the robot is at the node, it needs new action, so delay it for the next state/iteration
    for i, robot in enumerate(state.robots):
        robot.need_action = False
    
    if vertex_status == param.EventOutcome.BLOCK: # should not go here
        ValueError("Drones should never visit this node")
        state.depth += 1
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.TRAV:  # only at goal
        assert vertex.id == state.goalID
        state.depth += 1
        state.update_heuristic()
        return {state: (1.0, state.action_cost)}
    elif vertex_status == param.EventOutcome.CHANCE:
        if not state.robot.at_node: # reset if it is in middle of action
            state.robot.need_action = True # you don't want it to go to get_new_nodes_grobot (no node reached)
            state.robot.remaining_time = 0.0
            state.robot_actions = [core.Action(target=state.robot.edge[0], start_pose = (state.robot.cur_pose[0],state.robot.cur_pose[1])), 
                                   core.Action(target=state.robot.edge[1], start_pose = (state.robot.cur_pose[0],state.robot.cur_pose[1]))]
        new_state_trav = get_new_uav_node(state, uav_index, blocked=False) # TRAVERSABLE
        new_state_block = get_new_uav_node(state, uav_index, blocked=True) # BLOCKED
        assert new_state_trav.depth == new_state_block.depth
        return {new_state_trav: (1.0-vertex.block_prob, new_state_trav.action_cost),
                    new_state_block: (vertex.block_prob, new_state_block.action_cost)}

def using_uav_action_values(state, uav_index):
    if param.ADD_IV: # adding information gain
        if len(state.uav_actions) == 0 and len(state.uav_action_values) == 0:
            state.uav_actions = [core.Action(target=state.goalID, rtype=param.RobotType.Drone, 
                                        start_pose = (state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))]
            state.state_actions = [action for action in state.uav_actions]
        elif len(state.uav_actions) > 0:
            for action in state.uav_actions:
                action.update_pose((state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))
            state.state_actions = [action for action in state.uav_actions if action.target not in state.assigned_pois]
    else:
        if len(state.uav_actions) == 0:
            state.uav_actions = [core.Action(target=state.goalID, rtype=param.RobotType.Drone, 
                                        start_pose = (state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))]
            state.state_actions = [action for action in state.uav_actions]
        else:
            for action in state.uav_actions:
                action.update_pose((state.uavs[uav_index].cur_pose[0],state.uavs[uav_index].cur_pose[1]))
            state.state_actions = [action for action in state.uav_actions if action.target not in state.assigned_pois]
    

def get_new_uav_node(state, uav_index, blocked=False):
    new_state = state.copy()
    new_state.action_cost = state.action_cost
    if blocked:
        new_state.history.add_history(state.uavs[uav_index].action, param.EventOutcome.BLOCK)
    else:
        new_state.history.add_history(state.uavs[uav_index].action, param.EventOutcome.TRAV)
    new_state.noway2goal = are_robots_stuck(new_state)
    new_state.depth += 1
    new_state.update_heuristic()
    if param.ADD_IV and len(new_state.uav_actions) ==0 and len(new_state.uav_action_values) >0:
        new_state.uav_actions = [list(new_state.uav_action_values.keys())[0]]
        new_state.uav_action_values.pop(new_state.uav_actions[0])
    new_state.state_actions = [action for action in new_state.uav_actions]
    return new_state


def are_robots_stuck(state):
    if all([robot.last_node == state.goalIDs[i] for i, robot in enumerate(state.robots)]):
        return False
    robots_edges = []
    for robot in state.robots:
        if state.robot.at_node:
            robots_edges.append([state.robot.last_node, state.robot.pl_vertex])
        else:
            robots_edges.append([state.robot.edge[0],state.robot.edge[1]])
    for i, robot in enumerate(state.robots):
        if (not _is_robot_goal_connected(state.graph, state.history, robots_edges[i], state.goalIDs[i]))\
            or (len(state.robots_actions[i]) == 0):
            return True
    return False

def reset_uavs_action(state):
    for i, uav in enumerate(state.uavs):
        if not uav.need_action and uav.action not in state.uav_actions and uav.last_node != state.goalIDs[0]:
            uav.need_action = True 
            uav.remaining_time = 0.0

def _get_robot_that_finishes_first(state):
    time_remaining_uavs = []
    if len(state.uavs) > 0:
        for uav in state.uavs:
            if uav.last_node == state.goalIDs[0]:
                continue
            # if uav.remaining_time > param.APPROX_TIME:
            time_remaining_uavs.append(uav.remaining_time)
    robots_remaining_times = [robot.remaining_time for robot in state.robots]
    min_robot_time = min(robots_remaining_times)
    if len(time_remaining_uavs)==0 or min_robot_time < min(time_remaining_uavs):
        return True, robots_remaining_times.index(min_robot_time), min_robot_time
    else:
        assert len(state.uavs) > 0
        min_uav_time = min(time_remaining_uavs)
        remaining_times = [uav.remaining_time for uav in state.uavs]
        return False, remaining_times.index(min_uav_time), min_uav_time

def _is_robot_goal_connected(graph, history, redge, goalID):
    block_pois = []
    for key, value in history.get_data().items():
        if value == param.EventOutcome.BLOCK:
            block_pois.append(key.target)
    new_graph = g.modify_graph(graph=graph, robot_edge=redge, poiIDs=block_pois)
    reach = paths.is_reachable(graph=new_graph, start=redge[0], goal=goalID)
    return reach

# def sctp_rollout3(state):
#     if state.is_goal_state and not state.is_block_state:
#         return 0.0
#     if state.heuristic >= 0.0:
#         return state.heuristic
#     return state.update_heuristic2()


# def sampling_rollout(graph, robot_edge, d0, d1, goalID, atNode, startNode, n_samples=100):
#     # noway_penalty = 200.0
#     total_cost = 0.0
#     for _ in range(n_samples):
#         block_pois = [poi.id for poi in graph.pois if random.random() <= poi.block_prob ] 
#         modified_graph = g.modify_graph(graph=graph, robot_edge=robot_edge, poiIDs=block_pois)
#         if atNode:
#             cost, _ = paths.get_shortestPath_cost(modified_graph, startNode, goalID)
#             total_cost += cost if (cost >= 0.0) else param.NOWAY_PEN
#         else:
#             cost0, _ = paths.get_shortestPath_cost(modified_graph, robot_edge[0], goalID)
#             cost1, _ = paths.get_shortestPath_cost(modified_graph, robot_edge[1], goalID)
#             assert (cost1 < 0) == (cost0 < 0)
#             if cost0 >= 0:
#                 total_cost += (cost0+d0) if (cost0+d0)<(cost1+d1) else (cost1+d1)
#             else:
#                 total_cost += param.NOWAY_PEN
#     return total_cost/n_samples

# def get_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, 
#                      drone_pose, cur_heuristic, n_samples=100):
#     dist = np.linalg.norm(np.array(drone_pose)-np.array(graph.get_poi(action.target).coord))/VEL_RATIO
#     # value if the action is passable
#     block_value = 0.0
#     pass_value = 0.0
#     num_pois = len(graph.pois)
#     num_vertices = len(graph.vertices)
#     num_edges = len(graph.edges)
#     for _ in range(n_samples):
#         assert num_pois == len(graph.pois)
#         assert num_vertices == len(graph.vertices)
#         assert num_edges == len(graph.edges)
#         pass_value += sampling_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, block_edge=False)
#         block_value += sampling_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, block_edge=True)
#     pass_value /= n_samples
#     block_value /= n_samples
#     aver_block = graph.get_poi(action.target).block_prob * block_value
#     aver_pass = (1-graph.get_poi(action.target).block_prob) * pass_value
#     value = aver_block + aver_pass
#     bc = cur_heuristic - value
#     return bc, bc - dist 


# def sampling_action_value(graph, action, robot_edge, d0, d1, goalID, atNode, block_edge=False):
#     # noway_penalty = 200.0
#     block_pois = [poi.id for poi in graph.pois if poi.id != action.target and random.random() <= poi.block_prob ] 
#     if block_edge:
#         modified_graph = g.modify_graph(graph=graph, robot_edge=robot_edge, poiIDs=block_pois+[action.target])
#     else:
#         modified_graph = g.modify_graph(graph=graph, robot_edge=robot_edge, poiIDs=block_pois)
#     if atNode:
#         cost, _ = paths.get_shortestPath_cost(modified_graph, start=robot_edge[0], goal=goalID)
#         return cost if cost >= 0.0 else param.NOWAY_PEN
#     else:
#         cost0, _ = paths.get_shortestPath_cost(modified_graph, start=robot_edge[0], goal=goalID)
#         cost1, _ = paths.get_shortestPath_cost(modified_graph, start=robot_edge[1], goal=goalID)
#         assert (cost1 < 0) == (cost0 < 0)
#         return min(cost0+d0, cost1+d1) if cost0 >= 0 else param.NOWAY_PEN
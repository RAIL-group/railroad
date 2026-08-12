import pytest
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.param import EventOutcome, RobotType
import numpy as np
import random
from sctp.robot import Robot
from sctp.utils import plotting

def test_sctp_transition_lg_noblock():
    start, goal, l_graph, robots = graphs.linear_graph_unc()
    for poi in l_graph.pois:
        poi.block_prob = 0.0
    init_state = core.SCTPState(graph=l_graph, goal=goal.id, robots=robots)
    state_actions = init_state.get_actions()
    assert init_state.robot.cur_pose[0] == 0.0 and init_state.robot.cur_pose[1] ==0.0
    assert init_state.uavs[0].cur_pose[0] == 0.0 and init_state.uavs[0].cur_pose[1] ==0.0
    assert len(state_actions) == 2
    assert len(init_state.uav_actions) == len(state_actions)
    assert init_state.robot.need_action == True
    assert init_state.robot.last_node ==1
    # the first transition - assign action 4 to the drone
    state_prob_cost = init_state.transition(state_actions[0])
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    assert state1.robot.cur_pose[0] == 0.0 and state1.robot.cur_pose[1] ==0.0
    assert state1.uavs[0].cur_pose[0] == 0.0 and state1.uavs[0].cur_pose[1] ==0.0
    assert state1.action_cost == state_prob_cost[state1][1]
    assert state1.action_cost == 0.0
    assert state1.uavs[0].need_action == False
    assert state1.uavs[0].remaining_time == pytest.approx(2.5/2.0, 0.1)
    assert state1.robot.need_action == True
    assert state1.robot.remaining_time == 0.0
    assert state1.robot_actions == state1.state_actions
    assert len(state1.state_actions) == 1
    assert state1.robot.last_node ==1
    # the second transtion - assign action 4 to the robot then move
    state_prob_cost1 = state1.transition(state1.state_actions[0])
    assert len(state_prob_cost1) == 2
    prob = list(state_prob_cost1.values())[0][0]
    assert prob == 1.0 
    assert list(state_prob_cost1.values())[0][1] == pytest.approx(1.25, 0.01)
    state2 = list(state_prob_cost1.keys())[0]
    #=====================   # block situation ===================
    state2_2 = list(state_prob_cost1.keys())[1]
    actions = state2_2.get_actions()
    # assign action 5 to the drone
    state_prob_cost = state2_2.transition(actions[0])
    assert len(state_prob_cost) == 1
    state3_2 = list(state_prob_cost.keys())[0]
    actions = state3_2.get_actions()
    assert len(actions) == 1
    assert actions[0].target == 1
    # assign action 1 to ground robot then move then get stuck
    state_prob_cost = state3_2.transition(actions[0])
    assert len(state_prob_cost) == 1
    assert list(state_prob_cost.values())[0][1] == 31.25
    state4_2 = list(state_prob_cost.keys())[0]
    actions = state4_2.get_actions()
    assert len(actions) == 0
    assert state4_2.robot.need_action == True 
    assert state4_2.uavs[0].need_action == False
    assert state4_2.noway2goal == True 
    assert state4_2.is_goal_state == True
    # ==================== end of block situation ======================
    action = state2.get_actions()
    assert len(action) == 1
    # 3rd transition - assign action 5 to drone
    assert action[0].target == 5
    state_prob_cost2 = state2.transition(action[0])
    assert len(state_prob_cost2) == 1
    state3 = list(state_prob_cost2.keys())[0]
    action = state3.get_actions()
    assert len(action) == 2
    assert state3.robot.edge is not None 
    assert state3.robot.at_node == False
    # 4th transition - assign action 1 to the ground robot and move
    assert action[0].target == 1
    state_prob_cost3 = state3.transition(action[0])
    assert len(state_prob_cost3) == 1
    assert list(state_prob_cost3.values())[0][1] == 1.25
    assert list(state_prob_cost3.values())[0][0] == 1.00
    state4 = list(state_prob_cost3.keys())[0]
    action = state4.get_actions()
    assert len(action) == 1
    # 5th transition - assign action 4 to the ground robot and move
    assert action[0].target == 4
    state_prob_cost4 = state4.transition(action[0])
    assert len(state_prob_cost4) == 2
    assert list(state_prob_cost4.values())[0][1] == 2.5
    state = list(state_prob_cost4.keys())[0]
    assert state.robot.need_action == False 
    assert state.robot.at_node == True 
    assert state.robot.edge == None 
    assert state.uavs[0].need_action == True 
    assert state.uavs[0].at_node == True
    action = state.get_actions()
    assert action[0].target == 3
    assert len(action) == 1
    # six transition - assign action 3 to the drone
    assert action[0].target == 3
    state_prob_cost5 = state.transition(action[0])
    assert len(state_prob_cost5) ==1
    assert list(state_prob_cost5.values())[0][1] == 0.0
    assert list(state_prob_cost5.values())[0][0] == 1.0
    state = list(state_prob_cost5.keys())[0]
    action = state.get_actions()
    assert len(action) == 2
    assert action[0].target == 1
    assert action[1].target == 2
    assert np.array_equal(state.uavs[0].cur_pose, np.array([10.0, 0.0])) == True
    assert np.array_equal(state.robot.cur_pose, np.array([2.5, 0.0])) == True
    assert state.robot.need_action == True
    assert state.uavs[0].need_action == False
    # the seven transition - assign action 2 to the ground robot then move
    assert action[1].target == 2
    state_prob_cost = state.transition(action[1])
    assert len(state_prob_cost) == 1
    assert list(state_prob_cost.values())[0][0] == 1.0
    assert list(state_prob_cost.values())[0][1] == 2.5
    state = list(state_prob_cost.keys())[0]
    actions = state.get_actions()
    print(f"{state.robot_actions[0]} {state.robot_actions[1]}")
    assert len(actions) == 1
    assert np.array_equal(state.uavs[0].cur_pose, np.array([15.0, 0.0])) == True
    assert state.uavs[0].last_node == 3
    assert state.robot.last_node == 2 and state.robot.at_node == True
    assert np.array_equal(state.robot.cur_pose, np.array([5.0, 0.0])) == True
    # the eighth transition - reassign action 3 to the drone
    assert actions[0].target == 3
    state_prob_cost = state.transition(actions[0])
    assert len(state_prob_cost) == 1
    state = list(state_prob_cost.keys())[0]
    assert state.history.get_action_outcome(actions[0]) == EventOutcome.TRAV
    assert state.robot.at_node == True
    assert state.robot.last_node == 2
    actions = state.get_actions()
    assert len(actions) == 2
    assert actions[0].target == 4 or actions[0].target == 5
    # the ninth call of transition - assign action 5 for the ground robot
    state_prob_cost = state.transition(actions[1])
    state = list(state_prob_cost.keys())[0]
    print(state.history.get_action_outcome(actions[1]))
    actions = state.get_actions()
    assert len(actions) == 2
    # last transition
    state_prob_cost = state.transition(actions[1])
    state = list(state_prob_cost.keys())[0]
    assert state.is_goal_state == True



def test_sctp_transition_lg_prob():
    start, goal, l_graph, robots = graphs.linear_graph_unc()
    init_state = core.SCTPState(graph=l_graph, goal=goal.id, robots=robots)
    state_actions = init_state.get_actions()
    assert len(state_actions) == 2
    # the first transitiopn - assign action 4 to the drone
    state_prob_cost = init_state.transition(state_actions[0])
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    assert state1.action_cost == state_prob_cost[state1][1] and state1.action_cost == 0.0
    assert state1.uavs[0].remaining_time == pytest.approx(2.5/2.0, 0.1)
    assert len(state1.state_actions) == 1
    # the second transtion - assign action 4 to the robot then move
    state_prob_cost = state1.transition(state1.state_actions[0])
    assert len(state_prob_cost) == 2
    assert list(state_prob_cost.values())[0][0] == 0.5
    state2_1 = list(state_prob_cost.keys())[0]
    state2_2 = list(state_prob_cost.keys())[1]

    assert len(state2_1.get_actions()) == len(state2_2.get_actions())
    assert state2_1.get_actions()[0].target == state2_2.get_actions()[0].target
    # assign action 5 to the drone
    state_prob_cost1 = state2_1.transition(state2_1.get_actions()[0])
    state_prob_cost2 = state2_2.transition(state2_2.get_actions()[0])
    assert len(state_prob_cost1) == 1 and len(state_prob_cost1) == len(state_prob_cost2)
    state3_1 = list(state_prob_cost1.keys())[0]
    state3_2 = list(state_prob_cost2.keys())[0]
    assert len(state3_1.get_actions()) != len(state3_2.get_actions())
    assert len(state3_1.get_actions()) == 2
    assert state3_1.get_actions()[0].target == 1 and state3_1.get_actions()[1].target == 4
    assert state3_2.get_actions()[0].target == 1
    # +++++++++++++++++++++++++++++++++++++++++++++++
    # assign the action 1 for the robot, then go into a stuck/terminal state
    state_prob_cost2 = state3_2.transition(state3_2.get_actions()[0])
    assert len(state_prob_cost2) == 1
    state4_2 = list(state_prob_cost2.keys())[0]
    assert state4_2.is_goal_state == True
    # +++++++++++++++++++++++++++++++++++++++++++
    #assign action 4 to the ground robot, then move
    state_prob_cost1 = state3_1.transition(state3_1.get_actions()[1])
    assert len(state_prob_cost1) == 1
    state4_1 = list(state_prob_cost1.keys())[0]
    state4_1.robot.last_node == 4
    assert state4_1.robot.need_action == True
    assert state4_1.uavs[0].need_action == False 
    assert len(state4_1.get_actions()) == 2
    assert state4_1.get_actions()[0].target ==1 and state4_1.get_actions()[1].target == 2
    # assign action 2 to the ground robot, then move.
    state_prob_cost1 = state4_1.transition(state4_1.get_actions()[1])
    assert len(state_prob_cost1) == 2
    state5_1 = list(state_prob_cost1.keys())[0]
    state5_2 = list(state_prob_cost1.keys())[1]

def test_sctp_transition_dg_noblock():
    # when you test transition, we focus on the number of children state, prob, and cost
    start, goal, dgraph, robots = graphs.disjoint_unc()
    for poi in dgraph.pois:
        poi.block_prob = 0.0
    init_state = core.SCTPState(graph=dgraph, goal=goal.id, robots=robots)
    actions = init_state.get_actions()
    assert len(actions) == 4
    assert actions[0].rtype == RobotType.Drone
    assert len(init_state.v_vertices) == 1
    assert len(init_state.gateway) == 2
    assert actions[0].target == 5 and actions[1].target == 6 and actions[2].target == 7 and actions[3].target == 8
    ###### calling transition 1: assign action 7 to drone
    state_prob_cost = init_state.transition(actions[2])
    assert len(state_prob_cost) == 1
    state = list(state_prob_cost.keys())[0]
    assert list(state_prob_cost.values())[0][0] == 1.0
    assert list(state_prob_cost.values())[0][1] == 0.0
    assert state.robot.need_action == True and state.uavs[0].need_action == False
    ######  calling transition 2: assign action 8 to the robot
    actions = state.get_actions()
    assert len(actions) == 2
    assert actions[0].target == 5 and actions[1].target == 8
    state_prob_cost = state.transition(actions[1])
    assert len(state_prob_cost) == 2
    assert list(state_prob_cost.values())[0][0] == 1.0 and list(state_prob_cost.values())[0][1] == pytest.approx(2.83, 0.01)
    assert list(state_prob_cost.values())[1][0] == 0.0 and list(state_prob_cost.values())[1][1] == pytest.approx(2.83, 0.01)
    state = list(state_prob_cost.keys())[0]
    assert state.robot.need_action == True and state.uavs[0].need_action == False
    assert state.robot.last_node == 8 and state.robot.at_node == True and state.uavs[0].at_node ==False
    ##### transition call 3: if the edge is passable, keep going - assign action 4 to the ground robot:
    actions = state.get_actions()
    assert len(actions) == 1
    assert actions[0].target == 4
    state_prob_cost = state.transition(actions[0]) # drone reach its goal first
    assert len(state_prob_cost) == 2
    assert list(state_prob_cost.values())[0][0] == 1.0 and list(state_prob_cost.values())[0][1] == pytest.approx(0.17, 0.01) 
    state = list(state_prob_cost.keys())[0]
    assert state.robot.need_action == True and state.uavs[0].need_action == True 
    assert state.robot.at_node == False and state.uavs[0].at_node == True
    ###### transition call 4: assign action 5 to the drone
    actions = state.get_actions()
    assert len(actions) == 2 and actions[0].target == 5 and actions[1].target == 6
    state_prob_cost = state.transition(actions[0])
    assert len(state_prob_cost) == 1
    assert list(state_prob_cost.values())[0][0] == 1.0 and list(state_prob_cost.values())[0][1] == 0.0
    state = list(state_prob_cost.keys())[0]
    assert state.uavs[0].need_action == False and state.robot.need_action == True 
    assert state.robot.at_node == False and state.robot.edge is not None and state.uavs[0].at_node == True
    ###### transition call 5: assign action 8 to the robot, then move
    actions = state.get_actions()
    assert len(actions) == 2 and actions[0].target == 8 and actions[1].target == 4
    state_prob_cost = state.transition(actions[0])




def test_sctp_transition_graph_stuck1():
    seed = 2001
    np.random.seed(seed)
    random.seed(seed)
    exp_param=30.0
    num_iters = 2000
    start, goal, graph = graphs.graph_stuck()
    drones = []
    v2 = [v for v in graph.vertices if v.id==2][0]
    s = v2
    robot = Robot(position=[s.coord[0],s.coord[1]], cur_node=s.id, at_node=True)
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)

    actions = init_state.get_actions()
    assert len(actions) == 2
    assert actions[0].target == 6
    assert actions[1].target == 7

    # Taking action 7
    state_prob_cost = init_state.transition(action=actions[1])
    assert len(state_prob_cost) == 2
    prob_cost = list(state_prob_cost.values())
    assert prob_cost[0][0] == 0.85
    assert prob_cost[0][1] == 4.0
    assert prob_cost[1][0] == 0.15
    assert prob_cost[1][1] == 4.0    
    states7 = list(state_prob_cost.keys())
    assert states7[0].history.get_action_outcome(core.Action(target=7)) == EventOutcome.TRAV
    assert states7[1].history.get_action_outcome(core.Action(target=7)) == EventOutcome.BLOCK
    assert states7[0].history.get_action_outcome(core.Action(target=6)) == EventOutcome.CHANCE
    assert states7[1].history.get_action_outcome(core.Action(target=6)) == EventOutcome.CHANCE


    # Taking action 6
    state_prob_cost = init_state.transition(action=actions[0])
    assert len(state_prob_cost) == 2
    prob_cost = list(state_prob_cost.values())
    assert prob_cost[0][0] == 0.75
    assert prob_cost[0][1] == 1.25
    assert prob_cost[1][0] == 0.25
    assert prob_cost[1][1] == 1.25    
    states6 = list(state_prob_cost.keys())
    assert states6[0].history.get_action_outcome(core.Action(target=6)) == EventOutcome.TRAV
    assert states6[1].history.get_action_outcome(core.Action(target=6)) == EventOutcome.BLOCK
    assert states6[0].history.get_action_outcome(core.Action(target=7)) == EventOutcome.CHANCE
    assert states6[1].history.get_action_outcome(core.Action(target=7)) == EventOutcome.CHANCE
    actions = states6[0].get_actions()
    assert len(actions) == 1
    assert actions[0].target == 1
    actions = states6[1].get_actions()
    assert len(actions) == 1
    assert actions[0].target == 2
    # select TRAV state at P6, going to V1
    state_prob_cost = states6[0].transition(states6[0].get_actions()[0])
    assert len(state_prob_cost) == 1
    prob_cost = list(state_prob_cost.values())
    assert prob_cost[0][0] == 1.0
    assert prob_cost[0][1] == 1.25
    states = list(state_prob_cost.keys())
    actions = states[0].get_actions()
    assert len(actions) == 1
    assert actions[0].target == 10
    state_prob_cost = states[0].transition(states[0].get_actions()[0])
    assert len(state_prob_cost) == 2
    states = list(state_prob_cost.keys())
    assert states[0].robot.at_node == True 
    assert states[0].robot.last_node == 10
    prob_cost = list(state_prob_cost.values())
    assert states[0].history.get_action_outcome(core.Action(target=10)) == EventOutcome.TRAV
    assert states[1].history.get_action_outcome(core.Action(target=10)) == EventOutcome.BLOCK
    prob_cost[0][0] == 0.23
    prob_cost[0][1] == 4.0
    prob_cost[1][0] == 0.77
    prob_cost[1][1] == 4.0
    plotting.plot_policy(graph, actions=[], startID=s.id, \
                               goalID=goal.id, seed=seed, verbose=True)

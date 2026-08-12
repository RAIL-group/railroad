import pytest
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.robot import Robot
from sctp.param import RobotType, VEL_RATIO
from sctp.param import EventOutcome
  
def test_sctp_heuristic_lg():
    start, goal, l_graph, robots = graphs.linear_graph_unc()
    init_state = core.SCTPState(graph=l_graph, goal=goal.id, robots=robots)
    state_actions = init_state.get_actions()
    # the 1st transition - assign action 4 to the drone
    state_prob_cost = init_state.transition(state_actions[0])
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == 15.0
    # the 2nd transtion - assign action 4 to the robot then move
    actions = state.get_actions()
    assert actions[0].target == 4
    state_prob_cost = state.transition(actions[0])
    assert len(state_prob_cost) == 2
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == 13.75
    state2 = list(state_prob_cost.keys())[1]
    assert state2.heuristic == 13.75
    ###### third transition - assign action 5 to the drone - ground robot waits
    actions = state.get_actions()
    assert actions[0].target == 5
    state_prob_cost = state.transition(actions[0])
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == 13.75
    ###### fourth transition -- assign action 4 to the ground robot and move
    actions = state.get_actions()
    assert actions[1].target == 4
    assert actions[1].rtype == RobotType.Ground
    state_prob_cost = state.transition(actions[1])
    state = list(state_prob_cost.keys())[0]
    # print(f"The robot position: {state.robot.last_node}")
    assert state.robot.at_node == True
    assert state.robot.last_node == 4
    assert state.heuristic == 12.5
    ###### fifth transition -- assign action 2 to the ground robot and move
    actions = state.get_actions()
    assert actions[1].target == 2
    state_prob_cost = state.transition(actions[1])
    assert len(state_actions) == 2
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == 10.0
    ###### sixth transition -- assign action 3 to the drone
    actions = state.get_actions()
    state_prob_cost = state.transition(actions[0])
    assert len(state_prob_cost) == 1
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == 10.0
    ###### seven transition -- assign action 5 for ground robot, drone reaches its goal
    actions = state.get_actions()
    state_prob_cost = state.transition(actions[1])
    assert len(state_prob_cost) == 1
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == 7.5
    ###### eight transition -- assign wait action for drone
    actions = state.get_actions()
    state_prob_cost = state.transition(actions[0])
    assert len(state_prob_cost) == 1
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == 5.0
    ######## nine transition
    actions = state.get_actions()
    state_prob_cost = state.transition(actions[1])
    assert len(state_prob_cost) == 1
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == 0.0

def test_sctp_heuristic_dg():
    start, goal, l_graph, robots = graphs.disjoint_unc()
    init_state = core.SCTPState(graph=l_graph, goal=goal.id, robots=robots)
    init_actions = init_state.get_actions()
    assert len(init_actions) == 4
    assert init_actions[0].target == 5 
    assert init_actions[1].target == 6 
    assert init_actions[2].target == 7
    assert init_actions[3].target == 8
    
    ####### Assign action 5 to the drone, and action 8 to the ground robot
    state_prob_cost = init_state.transition(init_actions[0])
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == 8.0
    actions = state.get_actions()
    assert actions[1].target == 8 
    state_prob_cost = state.transition(actions[1])
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == 9.0 
    # Assign action 7 to the drone and action 8 to robot
    actions = state.get_actions()
    assert actions[1].target == 7
    state_prob_cost = state.transition(actions[1])
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == 9.0
    actions = state.get_actions()
    assert actions[1].target == 8
    state_prob_cost = state.transition(actions[1])
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == pytest.approx(3*2.83, 0.01)
    # Assign action 4 for the ground robot, drone is still moving
    actions = state.get_actions()
    assert actions[0].target == 4
    state_prob_cost = state.transition(actions[0])
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == pytest.approx(4*2.83-3.0, 0.01)

    ###### Assign action 6 to the drone and action 8 to the ground robot
    state_prob_cost = init_state.transition(init_actions[1])
    state = list(state_prob_cost.keys())[0]
    actions = state.get_actions()
    assert actions[1].target == 8 
    state_prob_cost = state.transition(actions[1])
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == pytest.approx(3*2.83, 0.01)
    # Assign action 4 to the robot, drone still keeps the same action
    assert state.uavs[0].need_action == False and state.uavs[0].remaining_time > 0.0
    actions = state.get_actions()
    assert actions[0].target == 4
    state_prob_cost = state.transition(actions[0])
    state = list(state_prob_cost.keys())[0]
    assert state.heuristic == pytest.approx(4*2.83-0.5*6.32, 0.01)


def test_heuristic_sampling_lg():
    start, goal, graph = graphs.linear_graph_unc()
    robot_edge = [start.id, start.id]
    d0 = 0.0
    d1 = 0.0
    num_samples = 30000
    heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                        goalID=goal.id, atNode=False, startNode=start.id,
                                        n_samples=num_samples)
    assert heuristic == pytest.approx(0.35*15.0+0.65*200, abs=0.5)

def test_heuristic_sampling_dj():
    start, goal, graph = graphs.disjoint_unc()
    robot_edge = [start.id, start.id]
    d0 = 0.0
    d1 = 0.0
    num_samples = 40000
    heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                        goalID=goal.id, atNode=False, startNode=start.id,
                                        n_samples=num_samples)
    true1 = 0.1*(0.72*8*1.41421356+0.28*200)
    true2 = 0.9*(0.1*8.0 +0.9*(0.72*8*1.41421356+0.28*200))
    assert heuristic == pytest.approx(true1+true2, abs=0.5)

def test_heuristic_sampling_sgraph():
    start, goal, graph = graphs.s_graph_unc()
    robot_edge = [start.id, start.id]
    d0 = 0.0
    d1 = 0.0
    num_samples = 40000
    heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                        goalID=goal.id, atNode=False, startNode=start.id,
                                        n_samples=num_samples)
    b1 = 0.1*(0.1*200 +0.9*(0.9*8*1.41421356+ 0.1*(0.1*200 + 0.9*(0.9*200 +0.1*(4+8*1.41421356)))))
    b21 = 0.9*(0.9*8*1.41421356+ 0.1*200 )
    b22 = 0.1*(0.9*(0.9*(4+8*1.41421356)+0.1*200) + 0.1*200)
    b2 = 0.9*(0.1*8+ 0.9*(b21 + b22))
    print(f"heuristic: {heuristic} --- true value: {b1+b2}")
    assert heuristic == pytest.approx(b1+b2, abs=0.5)
    
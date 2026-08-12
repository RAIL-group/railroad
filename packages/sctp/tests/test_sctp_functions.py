import pytest
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.robot import Robot
from sctp.param import RobotType, VEL_RATIO, EventOutcome
from sctp.utils import plotting, paths
import matplotlib.pyplot as plt

def test_sctp_actions():
    start_node = graphs.Vertex(coord=(0.0, 0.0))
    goal_node = graphs.Vertex(coord=(15.0, 0.0))
    node1 = graphs.Vertex(coord=(5.0, 0.0))
    
    graph = graphs.Graph(vertices=[start_node, goal_node, node1])
    graph.add_edge(start_node, node1, 0.0)
    graph.add_edge(node1, goal_node, 0.0)
    action1 = core.Action(target=node1.id)
    action2 = core.Action(target=goal_node.id)
    assert action1 != action2
    action3 = core.Action(target=node1.id)
    assert action1 == action3

def test_sctp_metric():
    metric1 = core.sctp_metric(10.0, 5.0)
    metric2 = core.sctp_metric(5.0, 10.0)
    assert metric1 > metric2
    metric3 = core.sctp_metric(10.0, 10.0)
    assert metric1 < metric3
    assert metric2 < metric3
    metric4 = core.sctp_metric(10.0, 5.0)
    assert metric1 == metric4
    assert metric3 != metric4
    assert metric1 + metric2 == core.sctp_metric(15.0, 15.0)

def test_sctp_function_init():
    start, goal, l_graph, robots = graphs.linear_graph_unc()

    action1 = core.Action(target=start.neighbors[0])
    state = core.SCTPState(graph=l_graph, goal=goal.id, robots=robots)
    assert state.history.get_data_length() == len(l_graph.vertices)
    assert len(state.uav_actions) == 2
    assert len(state.robot_actions) == 1
    assert all(uav.need_action == True for uav in state.uavs)
    assert state.robot.need_action == True
    assert state.state_actions == state.uav_actions
    assert state.robot == robots[0]
    assert state.uavs == robots[1:]
    assert len(state.assigned_pois) == 0

    state2 = state.copy()
    assert state2 != state
    assert state2.history == state.history
    assert state2.robot != state.robot
    for i in range(len(state2.uavs)):
        assert state2.uavs[i] != state.uavs[i]
    assert state2.uav_actions == state.uav_actions
    assert state2.robot_actions == state.robot_actions
    assert state2.state_actions == []
    assert state2.assigned_pois == state.assigned_pois

def test_sctp_function_advance_assign_task():
    start, goal, l_graph, robots = graphs.linear_graph_unc()
    init_state = core.SCTPState(graph=l_graph, goal=goal.id, robots=robots)
    state_actions = init_state.get_actions()
    assert init_state.robot.cur_pose[0] == 0.0 and init_state.robot.cur_pose[1] ==0.0
    assert init_state.uavs[0].cur_pose[0] == 0.0 and init_state.uavs[0].cur_pose[1] ==0.0
    assert len(state_actions) == 2
    assert len(init_state.uav_actions) == len(state_actions)
    assert init_state.robot.need_action == True
    assert len(init_state.robot_actions) == 1
    assert init_state.robot.last_node ==1
    # the first transition - assign action to the drone
    state_prob_cost = init_state.transition(state_actions[0])
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    assert state1.robot.cur_pose[0] == 0.0 and state1.robot.cur_pose[1] ==0.0
    assert state1.uavs[0].cur_pose[0] == 0.0 and state1.uavs[0].cur_pose[1] ==0.0
    assert state1.action_cost == state_prob_cost[state1][1]
    # assert state1.action_cost == core.sctp_metric(0.0, 0.0)
    assert state1.uavs[0].need_action == False
    assert state1.uavs[0].remaining_time == pytest.approx(2.5/VEL_RATIO, 0.1)
    assert state1.robot.need_action == True
    assert state1.robot.remaining_time == 0.0
    assert state1.robot_actions == state1.state_actions
    assert len(state1.state_actions) == 1
    assert state1.robot.last_node ==1
    # check whether the robot on move or at a node
    assert state1.robot.at_node == True 
    assert state1.robot.edge == None
    assert state1.uavs[0].at_node == True
    # the second transition - assign action to the ground robot then move
    state1_action = state1.get_actions()
    state_prob_cost2 = state1.transition(state1_action[0])
    assert len(state_prob_cost2) == 2
    states_list = list(state_prob_cost2.keys())
    state2_1 = states_list[0]
    assert state2_1.robot.need_action == True
    assert state2_1.robot.cur_pose[0] == pytest.approx(2.5/VEL_RATIO, 0.05)
    assert state2_1.robot.cur_pose[1] == pytest.approx(0.0, 0.05)
    assert state2_1.uavs[0].need_action == True
    assert state2_1.uavs[0].cur_pose[0] == 2.5
    assert state2_1.uavs[0].cur_pose[1] == 0.0
    assert state2_1.history.get_action_outcome(state1_action[0]) == EventOutcome.TRAV
    state2_1_actions = state2_1.get_actions()
    assert len(state2_1_actions) == 1
    assert state2_1_actions[0].target == 5
    assert state2_1.robot.edge == [1,4]
    assert state2_1.robot.at_node == False
    assert state2_1.uavs[0].at_node == True
    # third transition -- assigning action to drones
    state_prob_cost3 = state2_1.transition(state2_1_actions[0])
    assert len(state_prob_cost3) ==1
    state3 = list(state_prob_cost3.keys())[0]
    # robot is on an edge
    assert state3.robot.at_node == False
    assert state3.robot.edge == [1,4]
    # uav is at a node
    assert state3.uavs[0].at_node == True
    state3_actions = state3.get_actions()
    assert len(state3_actions) == 2
    assert state3_actions[0].target != state3_actions[1].target
    assert state3_actions[0].target == 4 or state3_actions[1].target == 4
    assert state3_actions[0].target == 1 or state3_actions[1].target == 1
    # forth transition -- assign action to robot and move
    print(f"Action 0: {state3_actions[0]}") 
    # move back to node 1
    state_prob_cost4 = state3.transition(state3_actions[0])
    assert len(state_prob_cost4) == 1
    state4 = list(state_prob_cost4.keys())[0]
    assert state4.robot.at_node == True 
    assert state4.robot.edge == None
    assert state4.uavs[0].at_node == False
    assert len(state4.get_actions()) == 1
    state4_action = state4.get_actions()[0]
    assert state4_action.target == 4
    if state4.history.get_action_outcome(state4_action) == EventOutcome.BLOCK:
        print("The node 4 is blocked")

def test_sctp_gateway_lg_prob():
    start, goal, l_graph, robots = graphs.linear_graph_unc()
    init_state = core.SCTPState(graph=l_graph, goal=goal.id, robots=robots)
    state_actions = init_state.get_actions()
    assert len(state_actions) == 2
    assert len(init_state.v_vertices) == 1
    assert len(init_state.gateway) == 1
    # the 1st transition - assign action 4 to the drone
    state_prob_cost = init_state.transition(state_actions[0])
    state1 = list(state_prob_cost.keys())[0]
    assert state1.action_cost == state_prob_cost[state1][1] and state1.action_cost == 0.0
    assert state1.uavs[0].remaining_time == pytest.approx(2.5/VEL_RATIO, 0.1)
    assert len(state1.state_actions) == 1
    # the 2nd transtion - assign action 4 to the robot then move
    state_prob_cost = state1.transition(state1.state_actions[0])
    assert len(state_prob_cost) == 2
    assert list(state_prob_cost.values())[0][0] == 0.5
    state2_1 = list(state_prob_cost.keys())[0]
    state2_2 = list(state_prob_cost.keys())[1]
    assert len(state2_1.get_actions()) == len(state2_2.get_actions())
    assert state2_1.get_actions()[0].target == state2_2.get_actions()[0].target
    action4 = core.Action(target=4)
    action1 = core.Action(target=1)
    assert state2_1.history.get_action_outcome(action4) == EventOutcome.TRAV
    assert state2_2.history.get_action_outcome(action4) == EventOutcome.BLOCK
    assert state2_2.is_goal_state == True
    assert len(state2_2.gateway) == 0
    assert len(state2_2.v_vertices) == 1
    assert state2_1.is_goal_state != True
    assert state2_1.robot.last_node == 1
    assert state2_1.uavs[0].last_node == 4
    actions = state2_1.get_actions()
    assert state2_1.uavs[0].need_action == True
    assert state2_1.robot.need_action == True
    assert actions[0] == core.Action(target=5)
    # assign action 5 to the drone
    state_prob_cost1 = state2_1.transition(actions[0])
    assert len(state_prob_cost1) == 1
    state3 = list(state_prob_cost1.keys())[0]
    assert state3.robot.last_node == 1
    assert state3.robot.need_action == True
    assert state3.uavs[0].need_action == False
    actions = state3.get_actions()
    assert len(actions) == 2
    assert actions[0].target == 1
    assert actions[1].target == 4
    state3.uavs[0].at_node = True
    assert len(state3.gateway) == 1
    assert len(state3.v_vertices) == 1
    ####### reassign action 4 and move the robot.
    state_prob_cost2 = state3.transition(actions[1])
    state4 = list(state_prob_cost2.keys())[0]
    assert state4.robot.need_action == True 
    assert state4.uavs[0].need_action == False
    assert len(state4.gateway) == 1
    assert len(state4.v_vertices) == 2
    
    ###### assign action 2 for robot then move
    actions = state4.get_actions()
    assert len(actions) == 2
    assert actions[1].target == 2
    ## drone reaches v5 and robot reach v2 at the same time after doing this action
    state_prob_cost3 = state4.transition(actions[1])
    assert len(state_prob_cost3) == 2
    act = core.Action(target=5)
    state5_1 = list(state_prob_cost3.keys())[0]
    state5_2 = list(state_prob_cost3.keys())[1]
    assert state5_1.history.get_action_outcome(act) == EventOutcome.TRAV 
    assert state5_2.history.get_action_outcome(act) == EventOutcome.BLOCK 
    assert state5_1.robot.need_action == state5_2.robot.need_action and state5_1.robot.need_action == False
    assert state5_2.robot.remaining_time == 0.0
    assert state5_2.uavs[0].need_action == True
    assert len(state5_1.gateway) == 1
    assert len(state5_1.v_vertices) == 2
    actions1 = state5_1.get_actions()
    actions2 = state5_2.get_actions()
    assert actions1[0] == actions2[0]
    
    ###### drone reaches v5 (robot reaches v2), assign action 3 to the drone
    state_prob_cost4_1 = state5_1.transition(actions1[0])
    state_prob_cost4_2 = state5_2.transition(actions2[0])
    assert len(state_prob_cost4_1) == len(state_prob_cost4_2) and len(state_prob_cost4_1) == 1
    state6_1 = list(state_prob_cost4_1.keys())[0]
    state6_2 = list(state_prob_cost4_2.keys())[0]
    assert state6_1.robot.need_action == state6_2.robot.need_action and state6_2.robot.need_action == True 
    assert state6_1.robot.last_node ==2
    assert state6_1.uavs[0].last_node ==5
    assert state6_1.uavs[0].need_action == False 
    assert state6_1.uavs[0].action.target == 3
    assert len(state6_1.gateway) == 1
    assert len(state6_1.v_vertices) == 3
    assert state6_1.is_goal_state == False
    assert len(state6_2.gateway) == 0
    assert len(state6_2.v_vertices) == 3
    assert state6_2.is_goal_state == True
    assert state6_2.robot.need_action == True
 
    ##### assign action 5 to ground robot then move
    actions = state6_1.get_actions()
    assert len(actions) == 2
    assert actions[1].target == 5 
    state_prob_cost5 = state6_1.transition(actions[1])
    assert len(state_prob_cost5) == 1
    state7 = list(state_prob_cost5.keys())[0]
    assert state7.robot.at_node == False 
    assert state7.robot.cur_pose[0] == 7.5 and state7.robot.cur_pose[1] == 0.0
    assert state7.robot.last_node == 2
    assert state7.uavs[0].at_node == True 
    assert state7.uavs[0].last_node == 3
    assert state7.robot.need_action == False
    assert state7.uavs[0].need_action == True
    #### moving back due to block at v5 and it is still terminate state
    actions = state6_2.get_actions()
    assert len(actions) == 1
    actions[0].target ==2
    state_prob_cost5_2 = state6_2.transition(actions[0])
    assert len(state_prob_cost5_2) == 1
    state7_2 = list(state_prob_cost5_2.keys())[0]
    assert state7_2.robot.at_node == True 
    assert state7_2.uavs[0].at_node == True
    assert state7_2.uavs[0].need_action == True
    assert state7_2.robot.need_action == False
    # print(f"number of gateways: {len(state7_2.gateway)}")
    assert state7_2.is_goal_state == True
    
    ##### ground robot keep moving
    actions = state7.get_actions()
    assert len(actions) == 1
    assert actions[0].target == 3
    state_prob_cost6 = state7.transition(actions[0])
    assert len(state_prob_cost6) == 1
    state8 = list(state_prob_cost6.keys())[0]
    assert state8.uavs[0].last_node == 3 and state8.uavs[0].need_action == False
    assert state8.robot.last_node == 5 and state8.robot.need_action == True
    actions = state8.get_actions()
    assert len(actions) == 2
    #### assign action 3 to the robot and move
    state_prob_cost6 = state8.transition(actions[1])
    assert len(state_prob_cost6) == 1
    state9 = list(state_prob_cost6.keys())[0]
    assert state9.is_goal_state == True
    assert state9.robot.need_action == True 
    assert state9.robot.last_node == 3

    
def test_sctp_gateway_splitting_sametime_dg():
    nodes = []
    node1 = graphs.Vertex(coord=(0.0, 0.0))
    nodes.append(node1)
    node2 =  graphs.Vertex(coord=(8.0, 0.0))
    nodes.append(node2)
    node3 =  graphs.Vertex(coord=(0.0, 4.0)) # goal node
    nodes.append(node3)
    node4 =  graphs.Vertex(coord=(8.0, 4.0))
    nodes.append(node4)

    graph = graphs.Graph(nodes)
    graph.edges.clear()
    graph.add_edge(node1, node2, 0.1)
    graph.add_edge(node1, node3, 0.2)
    graph.add_edge(node2, node4, 0.9)
    graph.add_edge(node3, node4, 0.2)
    G_robot = Robot(position=[0.0, 0.0], cur_node=node1.id)
    D_robot = Robot(position=[0.0, 0.0], cur_node=node1.id, robot_type=RobotType.Drone)
    robots = [G_robot, D_robot]
    vertices = graph.vertices + graph.pois
    graphs.dijkstra(vertices=vertices, edges=graph.edges, goal=node3)
    init_state = core.SCTPState(graph=graph, goal=node4.id, robots=robots)
    
    actions = init_state.get_actions()
    assert len(actions) == 4
    assert len(init_state.v_vertices) == 1
    assert len(init_state.gateway) == 2
    # the first transition - assign action 5 to the drone
    state_prob_cost = init_state.transition(actions[0])
    assert actions[0].target == 5
    assert len(state_prob_cost) == 1
    state1 = list(state_prob_cost.keys())[0]
    actions = state1.get_actions()
    assert len(actions) == 2
    ####### 2nd transtion - assign action 6 to the robot then move
    state_prob_cost = state1.transition(actions[1])
    assert len(state_prob_cost) == 2
    state1_1 = list(state_prob_cost.keys())[0]
    state1_2 = list(state_prob_cost.keys())[1]
    action5 = core.Action(target=5)
    assert state1_1.history.get_action_outcome(action5) == EventOutcome.TRAV
    assert state1_2.history.get_action_outcome(action5) == EventOutcome.BLOCK
    assert state1_1.robot.need_action == False and state1_1.robot.need_action == state1_2.robot.need_action
    assert state1_1.uavs[0].need_action == True and state1_1.uavs[0].need_action == state1_2.uavs[0].need_action
    assert state1_1.history.get_action_outcome(core.Action(target=6)) == EventOutcome.CHANCE
    actions = state1_1.get_actions()
    assert len(actions) == 3
    ###### 3rd transition - assign action 6 to drone
    state_prob_cost = state1_2.transition(actions[0])
    assert actions[0].target == 6
    assert len(state_prob_cost) == 2 
    state2_1 = list(state_prob_cost.keys())[0]
    state2_2 = list(state_prob_cost.keys())[1]
    assert state2_1.robot.need_action == state2_2.robot.need_action and state2_2.robot.need_action == True
    assert state2_1.uavs[0].need_action == state2_2.uavs[0].need_action and \
            state2_1.uavs[0].need_action == True
    assert state2_2.is_goal_state == True
    assert len(state2_2.get_actions()) == 1 and len(state2_2.gateway) == 0
    assert state2_2.action_cost == 30.0
    actions = state2_1.get_actions()
    assert len(actions) == 2
    assert actions[0].rtype == RobotType.Ground and actions[0].target == 1
    assert actions[1].rtype == RobotType.Ground and actions[1].target == 3
    ###### 4th transition -- assign action 3 to ground robot, reassign an action to drone
    state_prob_cost = state2_1.transition(actions[1])
    assert len(state_prob_cost) == 1
    state3 = list(state_prob_cost.keys())[0]
    assert state3.robot.last_node == 6 and state3.robot.need_action == False 
    assert state3.uavs[0].need_action == True
    actions = state3.get_actions()
    assert len(actions) == 2
    assert actions[0].rtype == RobotType.Drone and actions[0].target == 7
    assert actions[1].rtype == RobotType.Drone and actions[1].target == 8
    ###### 5th transition -- assign action 8 to the drone then move     
    state_prob_cost = state3.transition(actions[1])
    assert len(state_prob_cost) == 2
    state4_1 = list(state_prob_cost.keys())[0]
    state4_2 = list(state_prob_cost.keys())[1]
    assert state4_1.robot.last_node == 3
    assert state4_1.uavs[0].last_node == 8
    assert state4_1.robot.need_action==False and state4_1.robot.at_node == True 
    assert state4_1.uavs[0].need_action == True and state4_1.uavs[0].at_node == True
    assert state4_1.is_goal_state == False 
    ###### 6th transition -- assign action 7 to the drone
    actions = state4_1.get_actions()
    assert len(actions) == 1
    state_prob_cost = state4_1.transition(actions[0])
    assert len(state_prob_cost) == 1 # need to assign action for ground robot
    state5 = list(state_prob_cost.keys())[0]
    actions = state5.get_actions()
    assert len(actions) == 2
    assert actions[1].target == 8
    assert len(state5.gateway) == 1
    assert len(state5.v_vertices) == 3
    ###### 7th transition --- assign action 8 to the ground robot then move
    state_prob_cost = state5.transition(actions[1])
    assert len(state_prob_cost ) == 2
    state6_1 = list(state_prob_cost.keys())[0]
    assert state6_1.uavs[0].need_action == True 
    assert state6_1.robot.need_action == True 
    assert state6_1.uavs[0].at_node == True and state6_1.robot.at_node == False
    actions = state6_1.get_actions()
    assert len(actions) == 1 and actions[0].target == 4
    ###### 8th transition -- assign action 4 to the drone
    state_prob_cost = state6_1.transition(actions[0])
    assert len(state_prob_cost) == 1
    state7 = list(state_prob_cost.keys())[0]
    actions = state7.get_actions()
    assert len(actions) == 2 and actions[0].target == 3 and actions[1].target == 8
    ###### 9th transition -- reassign action 8 to the ground robot
    state_prob_cost = state7.transition(actions[1])
    assert len(state_prob_cost) == 1
    state8 = list(state_prob_cost.keys())[0]
    assert state8.uavs[0].at_node == True and state8.uavs[0].last_node == 4 and \
        state8.uavs[0].need_action == True
    assert state8.robot.at_node == False and state8.robot.last_node == 3 and \
        state8.robot.need_action == False
    ###### 10th transition -- wait action for drone
    actions = state8.get_actions()
    assert len(actions) == 1 and actions[0].target == 4
    state_prob_cost = state8.transition(actions[0])
    assert len(state_prob_cost) == 1
    state9 = list(state_prob_cost.keys())[0]
    assert state9.robot.need_action == True and state9.robot.at_node == True and \
        state9.robot.last_node == 8
    assert state9.uavs[0].need_action == False and state9.uavs[0].last_node == 4 and \
        state9.uavs[0].at_node == True
    ##### 11th transition -- assign action 4 to the robot
    actions = state9.get_actions()
    assert len(actions) == 2
    state_prob_cost = state9.transition(actions[1])
    assert len(state_prob_cost) == 1
    state10 = list(state_prob_cost.keys())[0]
    assert state10.robot.at_node == True and state10.robot.need_action == True 
    assert state10.uavs[0].at_node == True and state10.uavs[0].need_action == False 
    assert state10.is_goal_state == True

def test_sctp_get_poi_value_dg():
    start, goal, graph = graphs.disjoint_unc()
    poi_value = graphs.get_poi_value(graph=graph, poiID=7, startID=start.id, goalID=goal.id)
    assert poi_value == pytest.approx(3.31, 0.02)
    graph.pois[3].block_prob = 1.0
    poi_value = graphs.get_poi_value(graph=graph, poiID=7, startID=start.id, goalID=goal.id)
    assert poi_value < 0.0
        

def test_sctp_get_poi_value_sg():
    start, goal, graph = graphs.s_graph_unc()
    poi_value = graphs.get_poi_value(graph=graph, poiID=8, startID=start.id, goalID=goal.id)
    assert poi_value == 0.0
    poi_value = graphs.get_poi_value(graph=graph, poiID=9, startID=start.id, goalID=goal.id)
    assert poi_value == pytest.approx(3.31, 0.02)
    graph.pois[1].block_prob = 1.0
    poi_value = graphs.get_poi_value(graph=graph, poiID=8, startID=start.id, goalID=goal.id)
    assert poi_value == pytest.approx(2.34, 0.02)
    poi_value = graphs.get_poi_value(graph=graph, poiID=5, startID=start.id, goalID=goal.id)
    assert poi_value < 0.0

def test_sctp_get_poi_value_mg():
    start, goal, graph = graphs.m_graph_unc()
    poi_value = graphs.get_poi_value(graph=graph, poiID=13, startID=start.id, goalID=goal.id)
    assert poi_value == 0.0
    poi_value = graphs.get_poi_value(graph=graph, poiID=14, startID=start.id, goalID=goal.id)
    assert poi_value == pytest.approx(1.66, 0.02)
    poi_value = graphs.get_poi_value(graph=graph, poiID=16, startID=start.id, goalID=goal.id)
    assert poi_value == pytest.approx(0.0, 0.02)
    
    graph.pois[8].block_prob = 1.0
    poi_value = graphs.get_poi_value(graph=graph, poiID=17, startID=start.id, goalID=goal.id)
    assert poi_value == pytest.approx(5.66, 0.02)
    graph.pois[9].block_prob = 1.0
    poi_value = graphs.get_poi_value(graph=graph, poiID=18, startID=start.id, goalID=goal.id)
    assert poi_value < 0.0

def test_sctp_remove_pois_mg():
    start, goal, graph = graphs.m_graph_unc()

    pois_remove = [11,12,16]
    new_graph = graphs.remove_pois(graph=graph, poiIDs=pois_remove)
    assert 11 not in new_graph.vertices[1].neighbors
    assert 11 not in new_graph.vertices[4].neighbors
    assert 12 not in new_graph.vertices[1].neighbors
    assert 12 not in new_graph.vertices[5].neighbors
    assert 16 not in new_graph.vertices[3].neighbors
    assert 16 not in new_graph.vertices[6].neighbors
    assert len(new_graph.pois) == 9
    assert len(new_graph.edges) == 18
    
    plt.figure(figsize=(10, 10), dpi=300)
    plotting.plot_sctpgraph(graph=new_graph, plt=plt)
    plt.show()

def test_sctp_reachable_dg():
    start, goal, graph = graphs.disjoint_unc()
    pois_remove = [6,7]
    new_graph = graphs.remove_pois(graph=graph, poiIDs=pois_remove)
    is_connected = paths.is_reachable(new_graph, start.id, goal.id)
    assert is_connected == False

def test_sctp_reachable_mg_at_start():
    start, goal, graph = graphs.m_graph_unc()
    pois_remove = [16,17,18]
    new_graph = graphs.remove_pois(graph=graph, poiIDs=pois_remove)
    is_connected = paths.is_reachable(new_graph, start.id, goal.id)
    assert is_connected == False
    is_connected = paths.is_reachable(new_graph, start.id, graph.vertices[5].id)
    # plt.figure(figsize=(10, 10), dpi=300)
    # plotting.plot_sctpgraph(graph=new_graph, plt=plt, verbose=True)
    # plt.show()
    assert is_connected == True

def test_sctp_start_goal_connected_mg_at_start():
    start, goal, graph = graphs.m_graph_unc()
    
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    drones =[]
    init_state = core.SCTPState(graph=graph, robot=robot, drones=drones, goalID=goal.id)
    pois_remove = [16,17,18]
    for i in range(len(pois_remove)):
        init_state.history.add_history(core.Action(target=pois_remove[i]), EventOutcome.BLOCK)

    assert init_state.history.get_data_length() == 3 + 7 #vertices
    robot_edge = [start.id, start.id]
    is_connected = core._is_robot_goal_connected(graph=init_state.graph, \
                                                history=init_state.history, redge=robot_edge, goalID=goal.id)    
    assert is_connected == False

def test_sctp_start_goal_connected_mg_middle():
    start, goal, graph = graphs.m_graph_unc()
    poi = graph.pois[1]
    
    # robot = Robot(position=[poi.coord[0], poi.coord[1]], cur_node=poi.id, at_node=True)
    robot = Robot(position=[poi.coord[0], poi.coord[1]], cur_node=poi.id, edge=[1,9])
    drones =[]
    init_state = core.SCTPState(graph=graph, robot=robot, drones=drones, goalID=goal.id)
    pois_remove = [9, 18]
    for i in range(len(pois_remove)):
        init_state.history.add_history(core.Action(target=pois_remove[i]), EventOutcome.BLOCK)

    assert init_state.history.get_data_length() == len(pois_remove) + 7 #vertices
    robot_edge = [start.id, poi.id]
    is_connected = core._is_robot_goal_connected(graph=init_state.graph, \
                                                history=init_state.history, redge=robot_edge, goalID=goal.id)    
    assert is_connected == True

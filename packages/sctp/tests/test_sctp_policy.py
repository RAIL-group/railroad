import random
import numpy as np
from sctp import sctp_graphs as graphs
from sctp import core
from sctp.utils import plotting
from pouct_planner import core as policy
from sctp.robot import Robot
from sctp.param import EventOutcome, RobotType

def test_sctp_policy_lg():
    exp_param=50.0
    start, goal, l_graph = graphs.linear_graph_unc()
    robot = Robot(position=[0.0, 0.0], cur_node=start.id)
    drones = [Robot(position=[0.0, 0.0], cur_node=start.id, robot_type=RobotType.Drone)]
    init_state = core.SCTPState(graph=l_graph, goalID=goal.id, robot=robot, drones=drones)
    # ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=10000)
    
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=100,\
                                                rollout_fn=core.sctp_rollout)
    for p in pc[0]:
        print(p)
    for c in pc[1]:
        print(c)

def test_sctp_policy_dg():
    baseline = True
    seed = 2000
    np.random.seed(seed)
    random.seed(seed)
    exp_param=300.0
    num_iters = 20000
    print("")
    start, goal, graph = graphs.disjoint_unc()
    # graph.pois[0].block_prob = 0.9
    # graph.pois[1].block_prob = 0.5
    # graph.pois[2].block_prob = 0.9
    # graph.pois[3].block_prob = 0.9
    poi5 = [poi for poi in graph.pois+graph.vertices if poi.id==5][0]
    poi6 = [poi for poi in graph.pois+graph.vertices if poi.id==6][0]
    poi7 = [poi for poi in graph.pois+graph.vertices if poi.id==7][0]
    poi8 = [poi for poi in graph.pois+graph.vertices if poi.id==8][0]
    
    poi5.block_prob = 0.9
    poi6.block_prob = 0.5
    poi7.block_prob = 0.9
    poi8.block_prob = 0.9
    
    
    # robot = Robot(position=[graph.pois[0].coord[0], graph.pois[0].coord[1]], cur_node=graph.pois[0].id, at_node=True)
    robot = Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, at_node=True)
    if baseline:
        drones = []
    else:
        drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone)]
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    # assert init_state.history.get_data_length() == 4
    assert init_state.robot.need_action == True 
    tree_depth = 15
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=num_iters,\
                                    depth=tree_depth, rollout_fn=core.sctp_rollout3)
    # assert pc[0][0].target == 5
    reach_goal = False
    for p in pc[0]:
        print(p)
        if p.rtype==RobotType.Ground and p.target == goal.id:
            reach_goal = True 
    if reach_goal:
        pc[0].append(core.Action(target=goal.id, rtype=RobotType.Ground, start_pose=goal.coord))
        print("The ground robot reaches its goal")
    plotting.plot_policy(graph, actions=pc[0], startID=start.id, \
                               goalID=goal.id, seed=seed, verbose=True)
    # plotting.plot_firstAction(graph, action=pc[0][0], startID=start.id, \
    #                            goalID=goal.id, seed=seed, verbose=True)


def test_sctp_policy_sg():
    baseline = True
    seed = 2000
    np.random.seed(seed)
    random.seed(seed)
    exp_param=30.0
    num_iters = 1000
    start, goal, graph = graphs.s_graph_unc()
    poi5 = graph.pois[0]
    poi6 = graph.pois[1]
    poi7 = graph.pois[2]
    poi8 = graph.pois[3]
    poi9 = graph.pois[4]
    poi7.block_prob = random.random()
    poi8.block_prob = random.random()
    poi6.block_prob = random.random()
    poi5.block_prob = random.random()

    s = start
    robot = Robot(position=[s.coord[0], s.coord[1]], cur_node=s.id, at_node=True)
    if baseline:
        drones = []
    else:
        drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone)]
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=num_iters,\
                                             rollout_fn=core.sctp_rollout3)
    reach_goal = False
    for p in pc[0]:
        print(p)
        if p.rtype==RobotType.Ground and p.target == goal.id:
            reach_goal = True 
    if reach_goal:
        print("The ground robot reaches its goal")
        pc[0].append(core.Action(target=goal.id, rtype=RobotType.Ground, start_pose=goal.coord))
    plotting.plot_policy(graph, actions=pc[0], startID=s.id, \
                               goalID=goal.id, seed=seed, verbose=True)

    
def test_sctp_policy_mg():
    baseline = True
    seed = 2000
    np.random.seed(seed)
    random.seed(seed)
    exp_param=30.0
    num_iters = 1000
    start, goal, graph = graphs.m_graph_unc()
    robot = Robot(position=[start.coord[0],start.coord[1]], cur_node=start.id, at_node=True)
    if baseline:
        drones = []
    else:
        drones = [Robot(position=[start.coord[0], start.coord[1]], cur_node=start.id, robot_type=RobotType.Drone, at_node=True)]    
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=num_iters,\
                                             rollout_fn=core.sctp_rollout3)
    reach_goal = False
    for p in pc[0]:
        print(p)
        if p.rtype==RobotType.Ground and p.target == goal.id:
            reach_goal = True 
    if reach_goal:
        print("The ground robot reaches its goal")
        pc[0].append(core.Action(target=goal.id, rtype=RobotType.Ground, start_pose=goal.coord))
    plotting.plot_policy(graph, actions=pc[0], startID=start.id, \
                               goalID=goal.id, seed=seed, verbose=True)


def test_baseline_policy_dg_case():
    seed = 2002
    np.random.seed(seed)
    random.seed(seed)
    exp_param=30.0
    num_iters = 500
    start, goal, graph = graphs.disjoint_unc()
    v1 = [poi for poi in graph.vertices if poi.id==1][0]
    poi7 = [poi for poi in graph.pois if poi.id==7][0]
    poi5 = [poi for poi in graph.pois if poi.id==5][0]
    poi6 = [poi for poi in graph.pois if poi.id==6][0]
    poi7.block_prob = 0.55
    poi5.block_prob = 0.2
    poi6.block_prob = 0.9
    robot = Robot(position=[start.coord[0],start.coord[1]], cur_node=start.id, at_node=True)
    # robot = Robot(position=[v1.coord[0],v1.coord[1]], cur_node=v1.id, at_node=True)
    drones = []
    
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    
    
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=num_iters,\
                                             rollout_fn=core.sctp_rollout3)
    reach_goal = False
    for p in pc[0]:
        print(p)
        if p.rtype==RobotType.Ground and p.target == goal.id:
            reach_goal = True 
    if reach_goal:
        print("The ground robot reaches its goal")
        pc[0].append(core.Action(target=goal.id, rtype=RobotType.Ground, start_pose=goal.coord))
    plotting.plot_policy(graph, actions=pc[0], startID=start.id, \
                               goalID=goal.id, seed=seed, verbose=True)    


def test_sctp_policy_rg_2007():
    baseline = True
    seed = 2007
    np.random.seed(seed)
    random.seed(seed)
    exp_param=30.0
    num_iters = 1000
    start, goal, graph = graphs.random_graph(n_vertex=8)
    # graph = graphs.modify_graph(graph=graph, robot_edge=[6,6], poiIDs=[12])
    poi187 = [poi for poi in graph.pois if poi.id==187][0]
    poi181 = [poi for poi in graph.pois if poi.id==181][0]
    poi182 = [poi for poi in graph.pois if poi.id==182][0]
    poi179 = [poi for poi in graph.pois if poi.id==179][0]
    poi184 = [poi for poi in graph.pois if poi.id==184][0]
    poi185 = [poi for poi in graph.pois if poi.id==185][0]
    poi192 = [poi for poi in graph.pois if poi.id==192][0]
    
    poi174 = [poi for poi in graph.pois+graph.vertices if poi.id==174][0]
    poi171 = [poi for poi in graph.pois+graph.vertices if poi.id==171][0]
    
    poi181.block_prob = 0.0
    poi185.block_prob = 0.0
    # poi187.block_prob = 1.0
    # poi182.block_prob = 1.0
    poi179.block_prob = 1.0
    poi184.block_prob = 1.0
    poi192.block_prob = 1.0
    s = poi171
    robot = Robot(position=[s.coord[0], s.coord[1]], cur_node=s.id, at_node=True)
    if baseline:
        drones = []
    else:
        drones = [Robot([6.152, 8.16], cur_node=11, robot_type=RobotType.Drone, at_node=True)]
    
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    # action21 = core.Action(target=21)
    # action4 = core.Action(target=4)
    # action11 = core.Action(target=11)
    
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=num_iters,\
                                             rollout_fn=core.sctp_rollout3)
    reach_goal = False
    # assert pc[0][0] == action21
    # assert pc[0][1] == action4
    for p in pc[0]:
        print(p)
        if p.rtype==RobotType.Ground and p.target == goal.id:
            reach_goal = True 
    if reach_goal:
        print("The ground robot reaches its goal")
        pc[0].append(core.Action(target=goal.id, rtype=RobotType.Ground, start_pose=goal.coord))
    # plotting.plot_policy(graph, actions=pc[0], startID=s.id, \
    #                            goalID=goal.id, seed=seed, verbose=True)
    plotting.plot_firstAction(graph, action=pc[0][0], startID=s.id, \
                               goalID=goal.id, seed=seed, verbose=True)

def test_sctp_policy_rg_3000():
    baseline = False
    seed = 3000
    np.random.seed(seed)
    random.seed(seed)
    exp_param=300.0
    num_iters = 500
    heuristic_samples = 400
    tree_depth = 25
    n_vertex = 6
    start, goal, graph = graphs.random_graph(n_vertex=n_vertex)
    # poi203 = [poi for poi in graph.pois+graph.vertices if poi.id==203][0]
    # poi185 = [poi for poi in graph.pois+graph.vertices if poi.id==185][0]
    # poi194 = [poi for poi in graph.pois+graph.vertices if poi.id==194][0]
    # poi208 = [poi for poi in graph.pois+graph.vertices if poi.id==208][0]
    # poi193/5 = [poi for poi in graph.pois+graph.vertices if poi.id==224][0]
    # poi193 = [poi for poi in graph.pois+graph.vertices if poi.id==193][0]
    # poi196 = [poi for poi in graph.pois+graph.vertices if poi.id==196][0]
        
    
    # poi185.block_prob = 0.0
    # poi196.block_prob = 0.0
    # poi194.block_prob = 1.0
    # poi208.block_prob = 1.0
    # poi193.block_prob = 1.0
    # poi203.block_prob = 1.0
    # v4 = [v for v in graph.vertices if v.id==4][0]
    # v8 = [v for v in graph.pois if v.id==8][0]
    s = start
    robot = Robot(position=[s.coord[0], s.coord[1]], cur_node=s.id, at_node=True)
    if baseline:
        drones = []
    else:
        drones = [Robot(position=[s.coord[0], s.coord[1]], cur_node=s.id, robot_type=RobotType.Drone, at_node=True)]
    
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    
    # tree_depth = 50
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=num_iters,\
                                 depth=tree_depth, rollout_fn=core.sctp_rollout3)
    reach_goal = False
    for p in pc[0]:
        print(p)
        if p.rtype==RobotType.Ground and p.target == goal.id:
            reach_goal = True 
    verbose = True
    if reach_goal:
        print("The ground robot reaches its goal")
        pc[0].append(core.Action(target=goal.id, rtype=RobotType.Ground, start_pose=goal.coord))
    # print(f"The length of the policy is: {len(pc[0])}")
    plotting.plot_policy(graph, actions=pc[0], startID=s.id, \
                               goalID=goal.id, seed=seed, verbose=verbose)
    # plotting.plot_firstAction(graph, action=pc[0][0], startID=s.id, \
    #                            goalID=goal.id, seed=seed, verbose=True)



def test_baseline_policy_dg_swinging():
    seed = 2000
    np.random.seed(seed)
    random.seed(seed)
    exp_param=30.0
    num_iters = 500
    start, goal, graph = graphs.disjoint_unc()
    v1 = [poi for poi in graph.vertices+graph.pois if poi.id==5][0]

    for poi in graph.pois:
        if poi.id == 6:
            poi.block_status = int(0)
            poi.block_prob = 0.9
        if poi.id == 8:
            poi.block_status = int(0)
        if poi.id == 7:
            poi.block_status = int(1)
            poi.block_prob = 1.0
        if poi.id == 5:
            poi.block_status = int(0)
            poi.block_prob = 0.0

    # robot = Robot(position=[start.coord[0],start.coord[1]], cur_node=start.id, at_node=True)
    robot = Robot(position=[v1.coord[0],v1.coord[1]], cur_node=v1.id, at_node=True)
    drones = []
    
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    action5 = core.Action(target=5)
    action7 = core.Action(target=7)
    init_state.history.add_history(action=action5, outcome=EventOutcome.TRAV)
    init_state.history.add_history(action=action7, outcome=EventOutcome.BLOCK)


    
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=num_iters,\
                                             rollout_fn=core.sctp_rollout3)
    assert pc[0][0].target == 1
    reach_goal = False
    for p in pc[0]:
        print(p)
        if p.rtype==RobotType.Ground and p.target == goal.id:
            reach_goal = True 
    if reach_goal:
        print("The ground robot reaches its goal")
        pc[0].append(core.Action(target=goal.id, rtype=RobotType.Ground, start_pose=goal.coord))
    plotting.plot_policy(graph, actions=pc[0], startID=v1.id, \
                               goalID=goal.id, seed=seed, verbose=True)    



def test_baseline_policy_graph_stuck1():
    seed = 2005
    np.random.seed(seed)
    random.seed(seed)
    exp_param=30.0
    num_iters = 100
    start, goal, graph = graphs.graph_stuck()
    # graph = graphs.modify_graph(graph=graph, robot_edge=[6,6], poiIDs=[12])
    drones = []
    poi6 = [poi for poi in graph.pois if poi.id==6][0]
    poi9 = [poi for poi in graph.pois if poi.id==9][0]
    poi8 = [poi for poi in graph.pois if poi.id==8][0]
    poi11 = [poi for poi in graph.pois if poi.id==11][0]
    poi6.block_prob = 0.3
    poi11.block_prob = 0.77
    poi8.block_prob = 0.30
    v2 = [v for v in graph.vertices if v.id==2][0]
    s = start
    robot = Robot(position=[s.coord[0],s.coord[1]], cur_node=s.id, at_node=True)
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    
    
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=num_iters,\
                                             rollout_fn=core.sctp_rollout3)
    # assert pc[0][0].target == 2
    reach_goal = False
    for p in pc[0]:
        print(p)
        if p.rtype==RobotType.Ground and p.target == goal.id:
            reach_goal = True 
    if reach_goal:
        print("The ground robot reaches its goal")
        pc[0].append(core.Action(target=goal.id, rtype=RobotType.Ground, start_pose=goal.coord))
    plotting.plot_policy(graph, actions=pc[0], startID=s.id, \
                               goalID=goal.id, seed=seed, verbose=True)

def test_baseline_policy_graph_stuck2():
    seed = 2001
    np.random.seed(seed)
    random.seed(seed)
    exp_param=30.0
    num_iters = 2000
    start, goal, graph = graphs.graph_stuck()
    drones = []
    poi6 = [poi for poi in graph.pois if poi.id==6][0]
    poi7 = [poi for poi in graph.pois if poi.id==7][0]
    poi8 = [poi for poi in graph.pois if poi.id==8][0]
    poi9 = [poi for poi in graph.pois if poi.id==9][0]
    poi10 = [poi for poi in graph.pois if poi.id==10][0]
    # poi11 = [poi for poi in graph.pois if poi.id==10][0]
    # poi12 = [poi for poi in graph.pois if poi.id==12][0]
    poi10.block_prob = 0.10
    poi6.block_prob = 0.10
    poi8.block_prob = 0.85
    poi9.block_prob = 0.85
    poi7.block_prob = 0.1
    
    v1 = [v for v in graph.vertices if v.id==1][0]
    v2 = [v for v in graph.vertices if v.id==2][0]
    v3 = [v for v in graph.vertices if v.id==3][0]
    s = v2
    robot = Robot(position=[s.coord[0],s.coord[1]], cur_node=s.id, at_node=True)
    init_state = core.SCTPState(graph=graph, goalID=goal.id, robot=robot, drones=drones)
    
    
    ba, ec, pc  = policy.po_mcts(init_state, C=exp_param, n_iterations=num_iters,\
                                             rollout_fn=core.sctp_rollout3)
    # assert pc[0][0].target == 2
    reach_goal = False
    for p in pc[0]:
        print(p)
        if p.rtype==RobotType.Ground and p.target == goal.id:
            reach_goal = True 
    if reach_goal:
        print("The ground robot reaches its goal")
        pc[0].append(core.Action(target=goal.id, rtype=RobotType.Ground, start_pose=goal.coord))
    plotting.plot_policy(graph, actions=pc[0], startID=s.id, \
                               goalID=goal.id, seed=seed, verbose=True)


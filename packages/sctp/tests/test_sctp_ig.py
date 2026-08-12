import pytest, random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sctp import sctp_graphs as graphs
# from sctp.utils import plotting 
# from sctp.robot import Robot
from sctp.utils import plotting
from sctp import param
from sctp import core

def test_sctp_ig_djgraph():
   start, goal, graph = graphs.disjoint_unc()
   poi_check = graph.pois[0]
   poi6 = graph.pois[1]
   poi7 = graph.pois[2]
   poi8 = graph.pois[3]
   poi_check.block_prob = 0.0 
   poi6.block_prob = 0.4
   poi8.block_prob = 0.0
   poi7.block_prob = 0.5
   V2 = graph.vertices[1]
   V3 = graph.vertices[2]
   V4 = graph.vertices[3]
   # print(f"The node to check with ID: {poi_check.id}")
   # robot_edge = [start.id, start.id]
   robot_edge = [start.id, poi_check.id]
   d0 = 1.0
   d1 = 1.0
   num_samples = 15000
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   assert heuristic == pytest.approx(47.196, abs=1.0)
   # print(f"Heuristic: {heuristic}")
   action = core.Action(target = poi_check.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   assert cost == pytest.approx(-d0, abs=0.5)


def test_sctp_ig_sgraph():
   print("")
   start, goal, graph = graphs.s_graph_unc()
   poi5 = graph.pois[0]
   poi6 = graph.pois[1]
   poi7 = graph.pois[2]
   poi8 = graph.pois[3]
   poi9 = graph.pois[4]   
   V2 = graph.vertices[1]
   V3 = graph.vertices[2]
   V4 = graph.vertices[3]
   action = core.Action(target = poi7.id)
   num_samples = 15000
   # if all edges are passable, edge 7's not not positive
   poi6.block_prob = 0.0
   poi5.block_prob = 0.0
   poi7.block_prob = 0.5
   poi8.block_prob = 0.0
   poi9.block_prob = 0.0
   robot_edge = [start.id, start.id]
   d0 = 0.0
   d1 = 0.0
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   assert cost == pytest.approx(-2.24, abs=0.5)
   print(f"Cost: {cost}")
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()
   
   poi6.block_prob = 1.0
   poi5.block_prob = 0.0
   poi7.block_prob = 0.5
   poi8.block_prob = 0.0
   poi9.block_prob = 1.0
   robot_edge = [start.id, start.id]
   d0 = 0.0
   d1 = 0.0
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   assert cost == pytest.approx(-2.24, abs=0.5)
   print(f"Cost: {cost}")
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()
   
   # if all ways to goal are blocked, edge 7's cost is not positive, too
   poi8.block_prob = 1.0
   poi9.block_prob = 1.0
   poi6.block_prob = 0.2
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   # print(f"Heuristic: {heuristic}")
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)   
   print(f"Cost: {cost}")
   assert cost == pytest.approx(-2.24, abs=0.5)
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()
   # if 6,9 are blocked, edge 7's cost is not positive, too
   poi8.block_prob = 0.1
   poi9.block_prob = 1.0
   poi6.block_prob = 1.0
   poi5.block_prob = 0.4
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)   
   print(f"Cost: {cost}")
   assert cost == pytest.approx(-2.24, abs=1.0) or cost < 0.0
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()
   # if 6,8 have highly blocked likelihood, edge 7's has most value
   poi8.block_prob = 0.5
   poi9.block_prob = 0.9
   poi6.block_prob = 0.5
   poi5.block_prob = 0.9

   poi7.block_prob = 0.1
   num_samples = 20000
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   cost3 = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)   
   # print(f"Cost3: {cost3}")
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()
   poi7.block_prob = 0.5
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   # print(f"Heuristic: {heuristic}")
   cost1 = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)   
   # print(f"Cost1: {cost1}")
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()

   poi7.block_prob = 0.8
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=False, startNode=start.id,
                                     n_samples=num_samples)
   cost2 = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)   
   # print(f"Cost2: {cost2}")
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()
   
   assert cost1 > cost3
   assert cost1 > cost2


def test_sctp_middle_sgraph():
   print("")
   start, goal, graph = graphs.s_graph_unc()
   poi5 = graph.pois[0]
   poi6 = graph.pois[1]
   poi7 = graph.pois[2]
   poi8 = graph.pois[3]
   poi9 = graph.pois[4]   
   V2 = graph.vertices[1]
   V3 = graph.vertices[2]
   V4 = graph.vertices[3]
   action = core.Action(target = poi7.id)
   num_samples = 50000
   # if all edges are passable, edge 7's not not positive
   poi6.block_prob = 0.1
   poi5.block_prob = 0.1
   poi7.block_prob = 0.6
   poi8.block_prob = 0.1
   poi9.block_prob = 0.1
   robot_edge = [poi6.id, V3.id]
   d0 = 1.0
   d1 = 1.0
   atNode = False
   drone_pose = (3.0, 0.0)
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=atNode, startNode=start.id,
                                     n_samples=num_samples)
   b1 = 0.9*(0.1*200.0+ 0.9*(0.1*200+0.9*(8*1.41421356+ 3)))
   h = (1.0-poi9.block_prob)*5 + poi9.block_prob*((1.0-poi7.block_prob)*(0.9*(4*1.41421356+ 5)+0.1*200) + poi7.block_prob*(0.1*200+b1))
   assert heuristic == pytest.approx(h, abs=0.5)
   pass_a7 = 0.9*5 + 0.1*(0.9*(4*1.41421356+ 5)+0.1*200)
   block_a7 = 0.9*5 + 0.1*(b1+0.1*200)
   bh, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   cost_true  = h - pass_a7*(1.0-poi7.block_prob) - block_a7*poi7.block_prob - 2.236/param.VEL_RATIO
   assert cost == pytest.approx(cost_true, abs=0.5)
   

   
def test_sctp_ig_island_sgraph():
   print("")
   start, goal, graph = graphs.island_sgraph()
   num_samples = 15000
   poi9 = graph.pois[0]
   poi10 = graph.pois[1]
   poi11 = graph.pois[2]
   poi12 = graph.pois[3]
   poi13 = graph.pois[4]
   poi14 = graph.pois[5]
   poi15 = graph.pois[6]
   poi18 = graph.pois[9]
   poi16 = graph.pois[7]
   poi17 = graph.pois[8]
   poi19 = graph.pois[10]   
   # if all edges are passable, edge 7's not not positive
   poi9.block_prob = 0.0
   # poi5.block_prob = 0.0
   # poi7.block_prob = 0.5
   # poi8.block_prob = 0.0
   # poi9.block_prob = 0.0
   robot_edge = [start.id, start.id]
   d0 = 0.0
   d1 = 0.0
   
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=True, startNode=start.id,
                                     n_samples=num_samples)
   print(f"Heuristic: {heuristic}")
   action = core.Action(target = poi9.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   # assert cost == pytest.approx(-2.24, abs=0.5)
   print(f"Action {action.target} has a cost of : {cost}")
   
   action = core.Action(target = poi14.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")
   
   action = core.Action(target = poi15.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")
   
   action = core.Action(target = poi10.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")

   action = core.Action(target = poi11.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")

   action = core.Action(target = poi12.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")

   action = core.Action(target = poi13.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")

   action = core.Action(target = poi18.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")

   action = core.Action(target = poi16.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")

   action = core.Action(target = poi19.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")

   action = core.Action(target = poi17.id)
   cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=True, goalID=goal.id, drone_pose=start.coord,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has a cost of : {cost}")
   
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(start.coord[0], start.coord[1], marker='o', color='r')
   plt.text(start.coord[0], start.coord[1], 'start', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0], goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()

def test_sctp_ig_island_mgraph():
   print("")
   start, goal, graph = graphs.island_mgraph()
   num_samples = 100000
   poi9 = graph.pois[0]
   poi10 = graph.pois[1]
   poi11 = graph.pois[2]
   poi12 = graph.pois[3]
   poi13 = graph.pois[4]
   poi14 = graph.pois[5]
   poi15 = graph.pois[6]
   poi16 = graph.pois[7]   
   poi17 = graph.pois[8]
   poi18 = graph.pois[9]
   poi19 = graph.pois[10]
   poi20 = graph.pois[11]   
   # if all edges are passable, edge 7's not not positive   
   # robot_edge = [start.id, start.id]
   # d0 = 0.0
   # d1 = 0.0
   # atNode = True
   # if drone is moved
   robot_edge = [start.id, poi9.id]
   poi9.block_prob = 0.0
   d0 = 1.0
   d1 = 1.0
   atNode = False
   cur_robot_node = start.id
   cur_drone_pose = (2.0,3.0)
   cur_robot_pose = (1.0,3.0)
   
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=atNode, startNode=cur_robot_node,
                                     n_samples=num_samples)
   print(f"Heuristic: {heuristic}")
   # action = core.Action(target = poi9.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=start.coord,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # assert cost == pytest.approx(-2.24, abs=0.5)
   # print(f"Action {action.target} has behave changes : {behave_change:.2f} and cost: {cost:.2f}")
   action = core.Action(target = poi14.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has behave changes : {behave_change:.2f} and cost: {cost:.2f}")
      
   action = core.Action(target = poi15.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has behave changes : {behave_change:.2f} and cost: {cost:.2f}")
   
   action = core.Action(target = poi10.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has behave changes : {behave_change:.2f} and cost: {cost:.2f}")
   
   action = core.Action(target = poi11.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has behave changes : {behave_change:.2f} and cost: {cost:.2f}")
   
   action = core.Action(target = poi12.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has behave changes : {behave_change:.2f} and cost: {cost:.2f}")
   
   action = core.Action(target = poi13.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has behave changes : {behave_change:.2f} and cost: {cost:.2f}")
   
   action = core.Action(target = poi18.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has behave changes : {behave_change:.2f} and cost: {cost:.2f}")
   
   action = core.Action(target = poi16.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has behave changes : {behave_change:.2f} and cost: {cost:.2f}")
   
   action = core.Action(target = poi19.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has behave changes : {behave_change:.2f} and cost: {cost:.2f}")

   action = core.Action(target = poi17.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has behave changes : {behave_change:.2f} and cost: {cost:.2f}")
         
   action = core.Action(target = poi20.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"Action {action.target} has behave changes : {behave_change:.2f} and cost: {cost:.2f}")
   
   fig = plt.figure(figsize=(10, 10), dpi=300)
   plt.scatter(cur_robot_pose[0], cur_robot_pose[1], marker='o', color='r')
   plt.text(cur_robot_pose[0]-0.3, cur_robot_pose[1], 'R_s', fontsize=8)
   plt.scatter(cur_drone_pose[0], cur_drone_pose[1], marker='o', color='r')
   plt.text(cur_drone_pose[0]-0.3, cur_drone_pose[1], 'D_s', fontsize=8)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0]+0.2, goal.coord[1], 'goal', fontsize=8)
   plotting.plot_sctpgraph(graph, plt, verbose=True)
   plt.show()


def test_sctp_ig_island_random_small():
   print("")
   # seed = 3000
   # random.seed(seed)
   seed = 3000
   np.random.seed(seed)
   random.seed(seed)
   start, goal, graph = graphs.random_island_graph(n_island=3,SG_dist_min=8)
   num_samples = 50000
   poi51 = graph.pois[32]
   poi50 = graph.pois[31]
   poi50.block_prob = 0.65
   poi49 = graph.pois[30]
   poi26 = graph.pois[13]
   poi47 = graph.pois[28]
   poi46 = graph.pois[27]
   poi39 = graph.pois[20]
   poi39.block_prob = 0.1
   poi11 = graph.pois[4]
   poi15 = graph.pois[8]   
   poi15.block_prob = 0.1
   poi14 = graph.pois[7]
   poi14.block_prob = 0.1
   
   poi7 = graph.pois[0]
   poi8 = graph.pois[1]  
   poi9 = graph.pois[2]  
   poi13 = graph.pois[8] # it should be 6   
   assert poi15 == poi13
   poi40 = graph.pois[21]
   poi40.block_prob = 0.15
   # if all edges are passable, edge 7's not not positive   
   start = graph.get_vertex_by_id(5)
   goal = graph.get_vertex_by_id(34)
   robot_edge = [start.id, start.id]
   # poi14.block_prob = 0.75
   d0 = 0.0
   d1 = 0.0
   atNode = True
   cur_robot_node = start.id
   cur_drone_pose = (start.coord[0],start.coord[1])
   cur_robot_pose = (start.coord[0],start.coord[1])
   
   # if drone is moved and ground robot is at the middle of 
   # robot_edge = [start.id, poi9.id]
   # poi9.block_prob = 0.0
   # d0 = 1.0
   # d1 = 1.0
   # atNode = False
   # cur_robot_node = start.id
   # cur_drone_pose = (2.0,3.0)
   # cur_robot_pose = (1.0,3.0)
   
   heuristic = core.sampling_rollout(graph=graph, robot_edge=robot_edge, d0=d0, d1=d1,
                                     goalID=goal.id, atNode=atNode, startNode=cur_robot_node,
                                     n_samples=num_samples)
   print(f"Heuristic: {heuristic:.2f}")
   # action = core.Action(target = poi51.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   # action = core.Action(target = poi50.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   # action = core.Action(target = poi49.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   # action = core.Action(target = poi26.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=start.coord,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   # action = core.Action(target = poi47.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   # action = core.Action(target = poi46.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   # action = core.Action(target = poi39.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   # action = core.Action(target = poi11.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   action = core.Action(target = poi13.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   # action = core.Action(target = poi14.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   # action = core.Action(target = poi7.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   # action = core.Action(target = poi8.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   # action = core.Action(target = poi9.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   action = core.Action(target = poi15.id)
   behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
                                atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
                                cur_heuristic=heuristic, n_samples=num_samples)
   print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   # action = core.Action(target = poi40.id)
   # behave_change, cost = core.get_action_value(graph=graph, action=action, d0=d0, d1=d1, robot_edge=robot_edge,
   #                              atNode=atNode, goalID=goal.id, drone_pose=cur_drone_pose,
   #                              cur_heuristic=heuristic, n_samples=num_samples)
   # print(f"A{action.target}({behave_change:.2f}/{cost:.2f})->")
   
   fig = plt.figure(figsize=(10, 10), dpi=300)
   textsize = 4
   _, ax = plt.subplots()
   plt.scatter(cur_robot_pose[0], cur_robot_pose[1], marker='o', color='r')
   plt.text(cur_robot_pose[0]-0.3, cur_robot_pose[1]+0.5, 'R_s', fontsize=textsize)
   plt.scatter(cur_drone_pose[0], cur_drone_pose[1], marker='o', color='r')
   plt.text(cur_drone_pose[0]-0.3, cur_drone_pose[1]-1.5, 'D_s', fontsize=textsize)
   plt.scatter(goal.coord[0], goal.coord[1], marker='x', color='r')
   plt.text(goal.coord[0]+0.2, goal.coord[1], 'goal', fontsize=textsize)
   plotting.plot_sctpgraph(graph, ax, textsize=textsize, verbose=True)
   plt.show()


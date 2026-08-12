import numpy as np
import sctp.sctp_graphs as graphs
import sctp.param as param
import sctp.utils.paths as paths

# helper functions
def shortestPath_heuristic(redge, graph, goalID, vertices_map, robot, history):
    block_pois = [key.target for key, value in history.get_data().items() if value == param.EventOutcome.BLOCK]        
    new_graph = graphs.modify_graph(graph=graph, robot_edge=redge, poiIDs=block_pois)
    min_dist1 = paths.get_shortestPath_cost(graph=new_graph, start=redge[0], goal=goalID)
    min_dist2 = paths.get_shortestPath_cost(graph=new_graph, start=redge[1], goal=goalID)
    heuristic = 0.0
    if min_dist1 < 0.0 and min_dist2 < 0.0:
        heuristic = param.STUCK_COST
        return
    
    if min_dist1 < 0.0 or min_dist2 < 0.0:
        if min_dist1 > 0:
            v = vertices_map[redge[0]]
            heuristic = min_dist1 + np.linalg.norm(np.array(robot.cur_pose)-np.array(v.coord))
        else:
            v = vertices_map[redge[1]]
            heuristic = min_dist2 + np.linalg.norm(np.array(robot.cur_pose)-np.array(v.coord))
        return
    heuristic = min(min_dist2 + np.linalg.norm(np.array(robot.cur_pose)-np.array(vertices_map[redge[1]].coord)),
                            min_dist1 + np.linalg.norm(np.array(robot.cur_pose)-np.array(vertices_map[redge[0]].coord)))
    return heuristic    
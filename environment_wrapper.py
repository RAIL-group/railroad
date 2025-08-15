import random
import numpy as np
from environment import Map, Location
from mrppddl.helper import construct_search_operator, construct_move_visited_operator
import mrppddl
from environment import SymbolicToRealSimulator, Robot
from mrppddl.core import State, Fluent, transition, get_next_actions, get_action_by_name
from mrppddl.planner import MCTSPlanner
from time import time

PICK_TIME = 3







if __name__ == "__main__":
    # Initialize environment
    map = Map(n_robots=2, max_locations=3, seed=1015)
    print("Map locations:", map.locations)

    r1 = Robot(name='r1', start=map.robot_poses[0])
    r2 = Robot(name='r2', start=map.robot_poses[1])
    robots = [r1, r2]
    goal_fluents = {Fluent("found Knife"), Fluent("found Notebook"), Fluent("found Clock")}

    # Simulator: This handles simulation
    simulator = SymbolicToRealSimulator(map, robots, goal_fluents)


    # While goal is not reached, we will keep executing actions
    i = 0
    while not simulator.is_goal():
        objects_by_type = {
            "robot": [r.name for r in simulator.robots],
            "location": [location.name for location in simulator.locations if location.name not in simulator.visited_locations],
            "object": simulator.object_of_interest,
        }

        search_actions = construct_search_operator(simulator.get_likelihood_of_object,
                                                simulator.get_move_cost, PICK_TIME).instantiate(objects_by_type)

        all_actions = search_actions
        stime = time()

        print('-----------------------------------------')

        mcts = MCTSPlanner(all_actions)
        action_name = mcts(simulator.state, goal_fluents, max_iterations=10000, c=10)
        action = get_action_by_name(all_actions, action_name)
        print(f"Action selected: {action.name} in {time() - stime:.2f} seconds")

        simulator.execute_action(action)
        i += 1
        print(f"Iteration {i}, Current state: {simulator.state.fluents}")

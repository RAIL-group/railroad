import random
import numpy as np
from environment import Map, get_location_object_likelihood
from mrppddl.helper import construct_search_operator
import mrppddl
from environment import SymbolicToRealSimulator
from mrppddl.core import State, Fluent, transition, get_next_actions, get_action_by_name
from mrppddl.planner import MCTSPlanner
from time import time

PICK_TIME = 3

class Robot:
    def __init__(self, name=None, pose=None):
        self.pose = pose
        self.name = name

class Location:
    def __init__(self, name=None, location=None):
        self.name = name
        self.location = location



if __name__ == "__main__":
    # Initialize environment
    map = Map()
    start = Location(name='start', location=(0, 0))
    locations = [start] + [Location(name=name, location=location) for location, name in map.coords_locations.items()]
    # Initialize robots
    r1 = Robot(name='r1', pose=start.location)
    r2 = Robot(name='r2', pose=start.location)
    robots = [r1, r2]

    # Make initial state
    initial_state = State(
        time=0,
        fluents={
            Fluent("at r1 start"),
            Fluent("at r2 start"),
            Fluent("free r1"),
            Fluent("free r2"),
        }
    )
    goal_fluents = {Fluent("found Knife"), Fluent("found Notebook")}

    # Simulator: This handles everything
    simulator = SymbolicToRealSimulator(locations, robots, state=initial_state)

    objects_by_type = {
        "robot": [r.name for r in robots],
        "location": [location.name for location in locations],
        "object": map.objects_in_environment,
    }
    search_actions = construct_search_operator(simulator.get_likelihood_of_object,
                                               simulator.get_move_cost, PICK_TIME).instantiate(objects_by_type)

    # Get action
    print(objects_by_type['object'])
    print(search_actions[50:60])

    available_actions = get_next_actions(initial_state, search_actions)
    print(len(search_actions))
    print(len(available_actions))
    exit()

    for _ in range(100):
        stime = time()
        available_actions = get_next_actions(initial_state, search_actions)
        mcts = MCTSPlanner(available_actions)
        action_name = mcts(initial_state, goal_fluents, max_iterations=10000, c=10)

        action = get_action_by_name(available_actions, action_name)
        print(f"MCTS: {{Initial State: Best Action}}, time={time()-stime:.3f}")
        print(action)
        outcomes = transition(initial_state, action)
        # simulator.execute_action(action)
        exit()

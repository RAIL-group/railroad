import random
import numpy as np
from environment import Map, get_location_object_likelihood
from mrppddl.helper import construct_search_operator, construct_move_visited_operator
import mrppddl
from environment import SymbolicToRealSimulator
from mrppddl.core import State, Fluent, transition, get_next_actions, get_action_by_name
from mrppddl.planner import MCTSPlanner
from time import time

PICK_TIME = 3

class Robot:
    def __init__(self, name=None, pose=None):
        self.name = name
        self.prev_pose = None
        self.pose = pose
        self.net_motion = 0

    def move(self, target_pose):
        self.prev_pose = self.pose
        self.pose = target_pose
        self.net_motion += np.linalg.norm(np.array(self.prev_pose) - np.array(self.pose))

    # def move(self, distance):
    #     direction = np.array(self.target_pose) - np.array(self.pose)
    #     print(f"move {self.name} from {self.pose} towards {self.target_pose}")
    #     print(f'{direction=}, {distance=}')
    #     if not np.all(direction) == 0:
    #         self.pose = self.pose + distance * direction / np.linalg.norm(direction)
    #         self.net_motion += distance
    #     print(f'updated_pose = {self.pose}')

class Location:
    def __init__(self, name=None, location=None):
        self.name = name
        self.location = location


if __name__ == "__main__":
    # Initialize environment
    map = Map()
    start = Location(name='start', location=[0, 0])
    locations = [start] + [Location(name=name, location=location) for name, location in map.coords_locations.items()]
    for location in locations:
        print(location.name, location.location)
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
            Fluent("visited start"),
        }
    )

    goal_fluents = {Fluent("visited storage"), Fluent("visited desk"), Fluent("visited desk"), Fluent("visited kitchen")}

    # Simulator: This handles everything
    simulator = SymbolicToRealSimulator(locations, robots, state=initial_state, goal_fluents=goal_fluents)

    objects_by_type = {
        "robot": [r.name for r in robots],
        "location": [location.name for location in locations],
    }
    move_actions = construct_move_visited_operator(simulator.get_move_cost).instantiate(objects_by_type)

    for i in range(5):
        # Filter actions by current simulator state
        available_actions = get_next_actions(simulator.state, move_actions)
        print(f"---------TIMESTAMP {i}---------------------")
        for action in available_actions:
            print(action.name)
        # Use MCTS
        mcts = MCTSPlanner(available_actions)
        action_name = mcts(simulator.state, goal_fluents, max_iterations=10000, c=10)
        action = get_action_by_name(available_actions, action_name)

        # Execute action: This moves robot's position, changes state (updates fluents).
        print(f"MCTS: Best Action: {action.name}")
        simulator.execute_action(action)





# objects_by_type = {
#     "robot": [r.name for r in robots],
#     "location": [location.name for location in locations],
#     "object": map.objects_in_environment,
# }
# goal_fluents = {Fluent("found Knife"), Fluent("found Notebook")}
# search_actions = construct_search_operator(simulator.get_likelihood_of_object,
#                                            simulator.get_move_cost, PICK_TIME).instantiate(objects_by_type)

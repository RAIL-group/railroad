import pytest
from common import Pose
import environments
import environments.procthor
from mrppddl.planner import MCTSPlanner
import random
from mrppddl.core import Fluent as F, State, get_action_by_name
import procthor
import matplotlib.pyplot as plt
from pathlib import Path
from environments import plotting, utils
from environments.core import EnvironmentInterface as Simulator
from mrppddl._bindings import ff_heuristic
from mrppddl.dashboard import PlannerDashboard


def get_args():
    args = lambda: None
    # args.num_robots = 1
    args.num_robots = 2
    args.current_seed = 4001
    args.resolution = 0.05
    args.save_dir = './data/test_logs'
    return args


def main():

    args = get_args()
    args.current_seed = 4001
    env = environments.procthor.ProcTHOREnvironment(args)
    objects = ['teddybear_6', 'pencil_17']
    to_loc = 'garbagecan_5'

    objects_by_type = {
        "robot": [f'r{i+1}' for i in range(args.num_robots)] ,
        "location": env.locations.keys(),
        "object": objects,
    }

    initial_state = State(
            time=0,
            fluents={
                F("revealed start"),
                F("at r1 start"), F("free r1"),
                F("at r2 start"), F("free r2"),
            },
    )
    # Task: Place all objects at random_location
    goal_fluents = {F(f"at {obj} {to_loc}") for obj in objects}
    # goal_fluents = {F(f'found {objects[0]}')}
    # goal_fluents = {F(f'holding r1 {objects[0]}')}
    # goal_fluents = {F(f"at {objects[0]} {to_loc}")}
    # goal_fluents = {F(f"at {obj} {to_loc}") for obj in objects}

    # Create operators
    move_time_fn = env.get_move_cost_fn()
    search_time = lambda r, l: 5
    pick_time = lambda r, l, o: 5
    place_time = lambda r, l, o: 5
    object_find_prob = lambda r, l, o: 1.0
    move_op = environments.operators.construct_move_operator(move_time_fn)
    search_op = environments.operators.construct_search_operator(object_find_prob, search_time)
    pick_op = environments.operators.construct_pick_operator(pick_time)
    place_op = environments.operators.construct_place_operator(place_time)


    # Create simulator
    sim = Simulator(
        initial_state,
        objects_by_type,
        [pick_op, place_op, move_op, search_op],
        env
    )

    # Planning loop
    actions_taken = []
    max_iterations = 60  # Limit iterations to avoid infinite loops

    actions_taken = []
    for _ in range(max_iterations):
        if sim.goal_reached(goal_fluents):
            print("Goal reached!")
            break
        all_actions = sim.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(sim.state, goal_fluents, max_iterations=10000, c=1.41)
        print("------------------------")
        print(f'{action_name=}')
        print("------------------------")
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action, do_interrupt=False)
            print(sim.state.fluents)
            actions_taken.append(action_name)
        else:
            for action in all_actions:
                name, r, start, end = action.name.split()
                if name == 'move' and start == 'garbagecan_5':
                    print(action.name)
                    print(action.preconditions)
                    print(sim.state.satisfies_precondition(action))
                    print('---')
            break


    print(f"Actions taken: {actions_taken}")

    robot_all_poses = [Pose(*env.locations['start'])]
    for action in actions_taken:
        if not action.startswith('move'):
            continue
        _, _, _, to = action.split()
        robot_all_poses.append(Pose(*env.locations[to]))
    print(robot_all_poses)


if __name__ == "__main__":
    main()
"""
ProcTHOR environment planning demonstration.

Uses the new Goal API for defining planning objectives.
"""

from functools import reduce
from operator import and_

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
from environments import plotting, utils, SimulatedRobot
from environments.core import EnvironmentInterface
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
    robot_locations = {
        'robot1': 'start',
        'robot2': 'start',
    }
    env = environments.procthor.ProcTHOREnvironment(args, robot_locations=robot_locations)
    objects = ['teddybear_6', 'pencil_17']
    to_loc = 'garbagecan_5'

    objects_by_type = {
        "robot": robot_locations.keys(),
        "location": env.locations.keys(),
        "object": objects,
    }

    initial_state = State(
            time=0,
            fluents={
                F("revealed start"),
                F("at robot1 start"), F("free robot1"),
                F("at robot2 start"), F("free robot2"),
            },
    )
    # Task: Place all objects at random_location
    # Using Goal API: reduce(and_, [...]) creates an AndGoal
    goal = reduce(and_, [F(f"at {obj} {to_loc}") for obj in objects])

    # Create operators
    move_time_fn = env.get_skills_cost_fn(skill_name='move')
    search_time = env.get_skills_cost_fn(skill_name='search')
    pick_time = env.get_skills_cost_fn(skill_name='pick')
    place_time = env.get_skills_cost_fn(skill_name='place')
    object_find_prob = lambda r, l, o: 1.0
    move_op = environments.operators.construct_move_operator(move_time_fn)
    search_op = environments.operators.construct_search_operator(object_find_prob, search_time)
    pick_op = environments.operators.construct_pick_operator(pick_time)
    place_op = environments.operators.construct_place_operator(place_time)
    no_op = environments.operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Create simulator
    sim = EnvironmentInterface(
        initial_state,
        objects_by_type,
        [no_op, pick_op, place_op, move_op, search_op],
        env
    )

    # Planning loop
    actions_taken = []
    max_iterations = 60  # Limit iterations to avoid infinite loops

    # Dashboard
    h_value = ff_heuristic(initial_state, goal, sim.get_actions())
    with PlannerDashboard(goal, initial_heuristic=h_value) as dashboard:
        # (Optional) initial dashboard update
        dashboard.update(sim_state=sim.state)

        for iteration in range(max_iterations):
            # Check if goal is reached
            if goal.evaluate(sim.state.fluents):
                break

            # Get available actions
            all_actions = sim.get_actions()

            # Plan next action
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(sim.state, goal, max_iterations=10000, c=300, max_depth=20)

            if action_name == 'NONE':
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            # Execute action
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action, do_interrupt=False)
            actions_taken.append(action_name)

            tree_trace = mcts.get_trace_from_last_mcts_tree()
            h_value = ff_heuristic(sim.state, goal, sim.get_actions())
            relevant_fluents = {
                f for f in sim.state.fluents
                if any(keyword in f.name for keyword in ["at", "holding", "found", "searched"])
            }
            dashboard.update(
                sim_state=sim.state,
                relevant_fluents=relevant_fluents,
                tree_trace=tree_trace,
                step_index=iteration,
                last_action_name=action_name,
                heuristic_value=h_value,
            )

    # Print the full dashboard history to the console (optional)
    dashboard.print_history(sim.state, actions_taken)


if __name__ == "__main__":
    main()

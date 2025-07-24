from mrppddl.core import Operator, Fluent, Effect, transition
from mrppddl.core import State
from mrppddl.planner import astar, mcts
from mrppddl.heuristic import ff_heuristic
from mrppddl.helper import construct_move_operator, construct_search_operator
import time


def get_problem():
    # Define objects
    objects_by_type = {
        "robot": ["r1"],
        "location": ["start", "roomA", "roomB"],
        "object": ["cup", "bowl"]
    }

    def object_search_prob(robot, search_loc, obj):
        if search_loc == 'roomA':
            return 0.6
        else:
            return 0.4

    # Ground actions
    search_actions = construct_search_operator(object_search_prob, 5.0, 3).instantiate(objects_by_type)
    # Initial state
    initial_state = State(
        time=0,
        fluents={
            Fluent("at r1 start"),
            Fluent("free r1"),
        }
    )

    def is_goal_cup_found(fluents: frozenset[Fluent]) -> bool:
        return (
            Fluent("found cup") in fluents
        )

    ff_memory = dict()
    all_actions = search_actions
    out, node = mcts(initial_state, all_actions, is_goal_cup_found,            lambda state: ff_heuristic(state, is_goal_cup_found, all_actions, ff_memory=ff_memory), max_iterations=10000)

    print("MCTS: {Initial State: Best Action}")
    print(out)


get_problem()

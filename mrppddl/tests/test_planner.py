import pytest
from mrppddl.core import Fluent, State, transition, get_action_by_name
from mrppddl.helper import construct_move_visited_operator
from mrppddl.planner import MCTSPlanner
import random

F = Fluent


@pytest.mark.parametrize(
    "initial_fluents",
    [
        {F("at r1 start"), F("free r1"), F("visited start")},
        {
            F("at r1 start"),
            F("free r1"),
            F("at r2 start"),
            F("free r2"),
            F("visited start"),
        },
        {
            F("at r1 start"),
            F("free r1"),
            F("at r2 start"),
            F("free r2"),
            F("at r3 start"),
            F("free r3"),
            F("visited start"),
        },
    ],
    ids=["one robot", "two robots", "three robots"],
)
def test_planner_mcts_move_visit_multirobot(initial_fluents):
    # Get all actions
    objects_by_type = {
        "robot": ["r1", "r2", "r3"],
        "location": ["start", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
    }
    random.seed(8616)
    move_op = construct_move_visited_operator(lambda *args: 5.0 + random.random())
    all_actions = move_op.instantiate(objects_by_type)

    # Initial state
    initial_state = State(time=0, fluents=initial_fluents)
    goal_fluents = {
        F("visited a"),
        F("visited b"),
        F("visited c"),
        F("visited d"),
        F("visited e"),
    }

    def is_goal(state):
        return all(gf in state.fluents for gf in goal_fluents)

    state = initial_state
    mcts = MCTSPlanner(all_actions)
    for _ in range(15):
        if is_goal(state):
            print("Goal found!")
            break
        action_name = mcts(state, goal_fluents, 10000, c=10)
        if action_name == "NONE":
            break
        action = get_action_by_name(all_actions, action_name)

        state = transition(state, action)[0][0]
        print(action_name, state, is_goal(state))
    assert is_goal(state)

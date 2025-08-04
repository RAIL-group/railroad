import pytest
from mrppddl.core import (
    Fluent,
    State,
    transition,
    get_action_by_name,
    GroundedEffect,
    Action,
)
from mrppddl.helper import construct_move_visited_operator, construct_wait_operator
from mrppddl.planner import MCTSPlanner
import random

F = Fluent


def test_wait_for_transition():
    state = State(0, {F("free r1"), F("free r2")})
    # Action 1 is for robot 1: work quickly
    action_1 = Action(name="work r1",
                      preconditions={F("free r1")},
                      effects=[
                          GroundedEffect(0, {F("not free r1")}),
                          GroundedEffect(1.0, {F("free r1")})
                      ])
    # Action 2 is for robot 2: work slowly
    action_2 = Action(name="work r2",
                      preconditions={F("free r2")},
                      effects=[
                          GroundedEffect(0, {F("not free r2")}),
                          GroundedEffect(2.0, {F("free r2")})
                      ])
    # Action 3 is for robot 1: wait for r2
    action_3 = Action(name="wait r1 r2",
                      preconditions={F("free r1"), F("not free r2")},
                      effects=[
                          GroundedEffect(0, {F("not free r1"), F("waiting r1 r2")}),
                      ])

    state = transition(state, action_1)[0][0]
    assert state.time == 0

    state = transition(state, action_2)[0][0]
    assert state.time == 1

    state = transition(state, action_3)[0][0]
    assert state.time == 2
    assert F("free r1") in state.fluents
    assert F("waiting r1 r2") not in state.fluents
    assert F("free r2") in state.fluents
    assert F("free r3") not in state.fluents

@pytest.mark.parametrize(
    "initial_fluents",
    [
        {
            F("at r1 start"),
            F("free r1"),
            F("at r2 start"),
            F("free r2"),
            F("at r3 start"),
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
    ids=["two robots", "three robots"],
)
def test_planner_mcts_move_visit_wait_multirobot(initial_fluents):
    # Get all actions
    objects_by_type = {
        "robot": ["r1", "r2", "r3"],
        "location": ["start", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
    }
    random.seed(8616)
    move_op = construct_move_visited_operator(lambda *args: 5.0 + random.random())
    wait_op = construct_wait_operator()
    all_actions = move_op.instantiate(objects_by_type) + wait_op.instantiate(objects_by_type)
    print(all_actions)

    # Initial state
    initial_state = State(time=0, fluents=initial_fluents)
    goal_fluents = {
        F("at r1 start"),
        F("at r2 start"),
        F("at r3 start"),
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

import pytest
from mrppddl.core import (
    Fluent,
    Effect,
    State,
    get_next_actions,
    transition,
    get_action_by_name,
    GroundedEffect,
    Action,
)
from mrppddl.helper import construct_move_operator, construct_search_operator
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
    state = transition(state, action_2)[0][0]
    state = transition(state, action_3)[0][0]

    assert F("free r1") in state.fluents
    assert F("waiting r1 r2") not in state.fluents
    assert F("free r2") in state.fluents
    assert F("free r3") not in state.fluents

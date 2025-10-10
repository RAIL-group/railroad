import pytest
from mrppddl.core import (
    Fluent as F,
    Effect,
    State,
    get_next_actions,
    transition,
    get_action_by_name,
    GroundedEffect,
    Action,
)
import environments

def test_move_operator():
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": {"start", "roomA", "roomB", "roomC"},
    }
    move_time = lambda r, f, t: 10 if r == "r1" else 15
    move_op = environments.actions.construct_move_operator(move_time=move_time)

    initial_state = State(
            time=0,
            fluents={
                F("at", "r1", "start"),
                F("at", "r2", "start"),
                F("free", "r1"),
                F("free", "r2"),
            },
    )
    move_actions = move_op.instantiate(objects_by_type)
    a1 = get_action_by_name(move_actions, "move r1 start roomA")
    outcomes = transition(initial_state, a1)
    assert len(outcomes) == 1
    state1, prob1 = outcomes[0]
    assert prob1 == 1.0
    assert F("free r1") not in state1.fluents
    assert F("free r2") in state1.fluents
    assert F("at r1 start") not in state1.fluents
    assert F("at r2 start") in state1.fluents

    a2 = get_action_by_name(move_actions, "move r2 start roomB")
    outcomes = transition(state1, a2)
    assert len(outcomes) == 1
    state2, prob2 = outcomes[0]
    assert prob2 == 1.0
    assert F("free r1") in state2.fluents
    assert F("free r2") not in state2.fluents
    assert F("at r1 roomA") in state2.fluents
    assert F("at r2 start") not in state2.fluents
    assert F("at r2 roomB") not in state2.fluents
    assert state2.time == 10

    a3 = get_action_by_name(move_actions, "move r1 roomA roomC")
    outcomes = transition(state2, a3)
    assert len(outcomes) == 1
    state3, prob3 = outcomes[0]
    assert prob3 == 1.0
    assert F("free r1") not in state3.fluents
    assert F("free r2") in state3.fluents
    assert F("at r1 roomA") not in state3.fluents
    assert F("at r2 start") not in state3.fluents
    assert F("at r2 roomB") in state3.fluents
    assert F("at r1 roomC") not in state3.fluents
    assert state3.time == 15

def test_search_operator():
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": {"roomA", "roomB"},
        "object": {"objA", "objB"}
    }
    search_time = lambda r, l: 10 if r == "r1" else 15
    object_find_prob = lambda r, l, o: 0.8 if l == "roomA" else 0.2
    search_op = environments.actions.construct_search_operator(object_find_prob=object_find_prob, search_time=search_time)

    initial_state = State(
            time=0,
            fluents={
                F("at", "r1", "roomA"),
                F("at", "r2", "roomB"),
                F("free", "r1"),
                F("free", "r2"),
            },
    )
    search_actions = search_op.instantiate(objects_by_type)
    a1 = get_action_by_name(search_actions, "search r1 roomA objA")
    outcomes = transition(initial_state, a1)
    assert len(outcomes) == 1
    state, prob = outcomes[0]
    assert F("free r1") not in state.fluents
    assert F("free r2") in state.fluents
    assert F("at r1 roomA") in state.fluents
    assert F("lock-search roomA") in state.fluents
    assert F("searched roomA objA") not in state.fluents
    assert F("found objA") not in state.fluents

    a2 = get_action_by_name(search_actions, "search r2 roomB objA")
    outcomes = transition(state, a2)
    # r1 searches roomA first
    high_prob_state = next(s for s, p in outcomes if round(p, 2) == 0.8)
    assert F("free r1") in high_prob_state.fluents
    assert F("free r2") not in high_prob_state.fluents
    assert F("found objA") in high_prob_state.fluents
    assert F("searched roomA objA") in high_prob_state.fluents
    assert F("lock-search roomA") not in high_prob_state.fluents
    assert F("lock-search roomB") in high_prob_state.fluents

    low_prob_state = next(s for s, p in outcomes if round(p, 2) == 0.2)
    assert F("free r1") in low_prob_state.fluents
    assert F("free r2") not in low_prob_state.fluents
    assert F("found objA") not in low_prob_state.fluents
    assert F("searched roomA objA") in low_prob_state.fluents
    assert F("lock-search roomA") not in low_prob_state.fluents
    assert F("lock-search roomB") in low_prob_state.fluents

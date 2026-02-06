from railroad.core import (
    Fluent as F,
    State,
    transition,
    get_action_by_name,
)
from railroad import operators


def test_move_operator():
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": {"start", "roomA", "roomB", "roomC"},
    }
    def move_time(r, f, t):
        return 10 if r == "r1" else 15
    move_op = operators.construct_move_operator(move_time=move_time)

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
    def search_time(r, loc, o):
        return 10 if r == "r1" else 15

    def object_find_prob(r, loc, o):
        return 0.8 if loc == "roomA" else 0.2
    search_op = operators.construct_search_operator(
        object_find_prob=object_find_prob, search_time=search_time)

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


def test_pick_and_place_operator():
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": {"start", "roomA", "roomB", "roomC"},
        "object": {"objA", "objB"}
    }

    def pick_time(r, loc, o):
        return 10 if r == "r1" else 15

    def place_time(r, loc, o):
        return 10 if r == "r1" else 15

    initial_state = State(
        time=0,
        fluents={
            F("at", "r1", "roomA"), F("at", "objA", "roomA"), F("at", "objC", "roomA"),
            F("at", "r2", "roomB"), F("at", "objB", "roomB"),
            F("free", "r1"),
            F("free", "r2"),
        },
    )
    pick_op = operators.construct_pick_operator(pick_time=pick_time)
    place_op = operators.construct_place_operator(place_time=place_time)

    pick_actions = pick_op.instantiate(objects_by_type)
    place_actions = place_op.instantiate(objects_by_type)

    actions = pick_actions + place_actions

    # assign r1
    a1 = get_action_by_name(actions, "pick r1 roomA objA")
    outcomes = transition(initial_state, a1)
    assert len(outcomes) == 1
    state, _ = outcomes[0]
    assert state.time == 0
    # r1 is just assigned to pick objA from roomA
    assert F("free r1") not in state.fluents
    assert F("at r1 roomA") in state.fluents
    assert F("at objA roomA") not in state.fluents
    assert F("holding r1 objA") not in state.fluents
    # r2 is unaffected
    assert F("free r2") in state.fluents
    assert F("at r2 roomB") in state.fluents

    # assign r2
    a2 = get_action_by_name(actions, "pick r2 roomB objB")
    outcomes = transition(state, a2)
    assert len(outcomes) == 1
    state, _ = outcomes[0]
    # r1 finishes picking objA from roomA first
    assert state.time == 10
    assert F("free r1") in state.fluents
    assert F("at r1 roomA") in state.fluents
    assert F("at objA roomA") not in state.fluents
    assert F("holding r1 objA") in state.fluents
    # r2 is still picking objB from roomB
    assert F("free r2") not in state.fluents
    assert F("at r2 roomB") in state.fluents
    assert F("at objB roomB") not in state.fluents
    assert F("holding r2 objB") not in state.fluents

    # assign r1 to place objA at roomA
    a3 = get_action_by_name(actions, "place r1 roomA objA")
    outcomes = transition(state, a3)
    assert len(outcomes) == 1
    state, _ = outcomes[0]
    # r2 finishes picking objB
    assert state.time == 15
    assert F("free r2") in state.fluents
    assert F("at r2 roomB") in state.fluents
    assert F("holding r2 objB") in state.fluents
    assert F("at objB roomB") not in state.fluents
    # r1 is still placing
    assert F("free r1") not in state.fluents
    assert F("at r1 roomA") in state.fluents
    assert F("holding r1 objA") not in state.fluents
    assert F("at objA roomA") not in state.fluents

    # assign r2 to place objB at roomB
    a4 = get_action_by_name(actions, "place r2 roomB objB")
    outcomes = transition(state, a4)
    assert len(outcomes) == 1
    state, _ = outcomes[0]
    # r1 finishes placing objA
    assert state.time == 20
    assert F("free r1") in state.fluents
    assert F("at r1 roomA") in state.fluents
    assert F("holding r1 objA") not in state.fluents
    assert F("at objA roomA") in state.fluents
    # r2 is still placing objB
    assert F("free r2") not in state.fluents
    assert F("at r2 roomB") in state.fluents
    assert F("holding r2 objB") not in state.fluents
    assert F("at objB roomB") not in state.fluents

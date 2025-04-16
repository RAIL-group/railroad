import pytest
from mrppddl.core import Fluent, Effect, State, get_next_actions, transition, get_action_by_name
from mrppddl.helper import construct_move_operator, construct_search_operator


def test_fluent_equality():
    assert Fluent("at", "r1", "roomA") == Fluent("at", "r1", "roomA")
    assert Fluent("at", "r1", "roomA") == Fluent("at r1 roomA")
    assert not Fluent("at", "r1", "roomA") == Fluent("at", "r1", "roomB")
    assert not Fluent("at r1 roomA") == Fluent("at r1 roomB")
    assert not Fluent("at r1 roomA") == Fluent("at r1 rooma")
    
    # Test Negation
    assert Fluent("not at r1 roomA") == ~Fluent("at r1 roomA")
    assert Fluent("at r1 roomA") == ~Fluent("not at r1 roomA")
    assert not Fluent("at", "r1", "roomA") == ~Fluent("at", "r1", "roomA")
    assert not Fluent("at", "r1", "roomA") == ~Fluent("at r1 roomA")
    assert not Fluent("at", "r1", "roomA") == ~Fluent("at r1 roomA")


def test_fluents_update_1():
    f1 = Fluent("at", "r1", "roomA")
    f2 = Fluent("free", "r1")
    f3 = Fluent("holding", "r1", "medkit")

    s = State(fluents={f1, f2})

    # Apply update with positive fluent and no conflict
    s.update_fluents({f3})
    assert f3 in s.fluents

    # # Apply update with negation of f2 (should remove f2)
    s.update_fluents({~f2})
    assert f2 not in s.fluents
    assert f3 in s.fluents

    # Apply update with both positive and negated fluent
    s.update_fluents({~f3, f2})
    assert f3 not in s.fluents
    assert f2 in s.fluents


def test_fluents_update_2():
    state = State(fluents={
        Fluent('at robot1 bedroom'),
        Fluent('free robot1'),
    })
    state.update_fluents({
        ~Fluent('free robot1'),
        ~Fluent('at robot1 bedroom'),
        Fluent('at robot1 kitchen'),
        ~Fluent('found fork'),
    })
    expected = State(fluents={
        Fluent('at robot1 kitchen'),
    })

    assert state == expected, f"Unexpected result: {state}"

    # Now re-add a positive fluent
    state.update_fluents({Fluent('free robot1')})
    expected = State(fluents={
        Fluent('free robot1'),
        Fluent('at robot1 kitchen'),
    })
    assert state == expected, f"Unexpected result after re-adding: {state}"


def test_ground_effect_time_float():
    """Show that we can properly ground effects with both floats and lambdas for times."""
    eff_time = 5
    lifted = Effect(time=eff_time, resulting_fluents={Fluent("free ?robot")})
    grounded = lifted._ground({"?robot": "wall-e"})

    assert grounded.time == eff_time
    assert grounded.resulting_fluents == {Fluent("free wall-e")}


def test_ground_effect_time_parametrized():
    # Define a mock travel-time function
    def mock_travel_time(robot: str, loc_from: str, loc_to: str) -> float:
        assert robot == "robot1"
        assert loc_from == "roomA"
        assert loc_to == "roomB"
        return 7.5

    # Create a lifted effect using the new syntax
    lifted = Effect(
        time=(mock_travel_time, ["?robot", "?loc_from", "?loc_to"]),
        resulting_fluents={
            Fluent("free", "?robot"),
            Fluent("at", "?robot", "?loc_to"),
            ~Fluent("at", "?robot", "?loc_from")
        }
    )

    # Ground it with a specific binding
    binding = {
        "?robot": "robot1",
        "?loc_from": "roomA",
        "?loc_to": "roomB"
    }
    grounded = lifted._ground(binding)

    # Check time is evaluated correctly
    assert grounded.time == 7.5

    # Check fluents are properly substituted
    expected_fluents = {
        Fluent("free", "robot1"),
        Fluent("at", "robot1", "roomB"),
        ~Fluent("at", "robot1", "roomA")
    }
    assert grounded.resulting_fluents == expected_fluents




@pytest.mark.parametrize("move_time", [5, lambda r, f, t: 5])
def test_move_sequence(move_time):
    # Define move operator
    move_op = construct_move_operator(move_time)

    # Ground actions
    objects_by_type = {
        "robot": ["r1", "r2"],
        "location": ["roomA", "roomB"],
    }
    move_actions = move_op.instantiate(objects_by_type)

    # Initial state
    initial_state = State(
        time=0,
        fluents={
            Fluent("at", "r1", "roomA"),
            Fluent("at", "r2", "roomA"),
            Fluent("free", "r1"),
            Fluent("free", "r2")
        }
    )
    for a in move_actions:
        print(a)
    print(initial_state)

    # First transition: move r1 from roomA to roomB
    available = get_next_actions(initial_state, move_actions)
    print(available)
    assert any(a.name == "move r1 roomA roomB" for a in available)
    a1 = get_action_by_name(available, "move r1 roomA roomB")
    outcomes = transition(initial_state, a1)
    assert len(outcomes) == 1
    state1, prob1 = outcomes[0]
    assert prob1 == 1.0
    assert Fluent("free", "r1") not in state1.fluents
    assert Fluent("free", "r2") in state1.fluents
    assert len(state1.upcoming_effects) == 1

    # Second transition: move r2 from roomA to roomB
    available = get_next_actions(state1, move_actions)
    assert any(a.name == "move r2 roomA roomB" for a in available)
    a2 = get_action_by_name(available, "move r2 roomA roomB")
    outcomes = transition(state1, a2)
    assert len(outcomes) == 1
    state2, prob2 = outcomes[0]
    assert prob2 == 1.0
    assert Fluent("at", "r1", "roomB") in state2.fluents
    assert Fluent("at", "r2", "roomB") in state2.fluents
    assert Fluent("free", "r1") in state2.fluents
    assert Fluent("free", "r2") in state2.fluents
    assert state2.time == 5
    assert len(state2.upcoming_effects) == 0


def test_search_sequence():
    # Define objects
    objects_by_type = {
        "robot": ["r1"],
        "location": ["roomA", "roomB"],
        "object": ["cup", "bowl"]
    }

    def object_search_prob(robot, search_loc, obj):
        if obj == 'cup':
            return 0.8
        else:
            return 0.6

    # Ground actions
    search_actions = construct_search_operator(object_search_prob, 5.0, 3).instantiate(objects_by_type)
    # Initial state
    initial_state = State(
        time=0,
        fluents={
            Fluent("at r1 roomA"),
            Fluent("free r1"),
        }
    )

    # Select action: search r1 roomA roomB cup
    action_1 = get_action_by_name(search_actions, 'search r1 roomA roomB cup')
    outcomes = transition(initial_state, action_1)

    # Assert both probabilistic outcomes exist
    assert len(outcomes) == 2
    probs = {round(p, 2) for _, p in outcomes}
    assert probs == {0.8, 0.2}

    # Verify high-probability (success) branch
    high_prob_state = next(s for s, p in outcomes if round(p, 2) == 0.8)
    assert Fluent("at r1 roomB") in high_prob_state.fluents
    assert Fluent("holding r1 cup") in high_prob_state.fluents
    assert Fluent("free r1") in high_prob_state.fluents
    assert Fluent("found cup") in high_prob_state.fluents
    assert Fluent("searched roomB cup") in high_prob_state.fluents
    assert high_prob_state.time == 8

    # Verify low-probability (failure to find object) branch
    low_prob_state = next(s for s, p in outcomes if round(p, 2) == 0.2)
    assert Fluent("at r1 roomB") in low_prob_state.fluents
    assert Fluent("free r1") in low_prob_state.fluents
    assert Fluent("found cup") not in low_prob_state.fluents
    assert Fluent("holding r1 cup") not in low_prob_state.fluents
    assert Fluent("searched roomB cup") in low_prob_state.fluents
    assert low_prob_state.time == 5

    # Continue from high-probability outcome
    action_2 = get_action_by_name(search_actions, 'search r1 roomB roomA bowl')
    next_outcomes = transition(high_prob_state, action_2)

    assert len(next_outcomes) == 2
    for state, prob in next_outcomes:
        assert Fluent("at", "r1", "roomA") in state.fluents
        assert Fluent("searched", "roomA", "bowl") in state.fluents
        assert Fluent("found", "cup") in state.fluents
        assert Fluent("holding", "r1", "cup") in state.fluents
        if round(prob, 2) == 0.6:
            assert Fluent("found", "bowl") in state.fluents
            assert Fluent("holding", "r1", "bowl") in state.fluents
            assert Fluent("free", "r1") in state.fluents
            assert state.time == 16
        elif round(prob, 2) == 0.4:
            assert Fluent("found", "bowl") not in state.fluents
            assert Fluent("holding", "r1", "bowl") not in state.fluents
            assert Fluent("free", "r1") in state.fluents
            assert state.time == 13
        else:
            assert round(prob, 2) in {0.6, 0.4}

import pytest
from railroad.core import (
    Fluent,
    Effect,
    State,
    get_next_actions,
    transition,
    get_action_by_name,
    GroundedEffect,
    Action,
    Operator,
)
from railroad.helper import construct_move_operator, construct_search_operator
import random

F = Fluent


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


def test_fluent_equality_hash():

    assert hash(F("at", "r1", "roomA")) == hash(F("at", "r1", "roomA"))
    assert hash(F("at", "r1", "roomA")) == hash(F("at r1 roomA"))
    assert not hash(F("at", "r1", "roomA")) == hash(F("at", "r1", "roomB"))
    assert not hash(F("at r1 roomA")) == hash(F("at r1 roomB"))
    assert not hash(F("at r1 roomA")) == hash(F("at r1 rooma"))

    # Test Negation
    assert hash(F("not at r1 roomA")) == hash(~F("at r1 roomA"))
    assert hash(F("at r1 roomA")) == hash(~F("not at r1 roomA"))
    assert not hash(F("at", "r1", "roomA")) == hash(~F("at", "r1", "roomA"))
    assert not hash(F("at", "r1", "roomA")) == hash(~F("at r1 roomA"))
    assert not hash(F("at", "r1", "roomA")) == hash(~F("at r1 roomA"))


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
    state = State(
        fluents={
            Fluent("at robot1 bedroom"),
            Fluent("free robot1"),
        }
    )
    state.update_fluents(
        {
            ~Fluent("free robot1"),
            ~Fluent("at robot1 bedroom"),
            Fluent("at robot1 kitchen"),
            ~Fluent("found fork"),
        }
    )
    expected = State(
        fluents={
            Fluent("at robot1 kitchen"),
        }
    )

    assert state == expected, f"Unexpected result: {state}"

    # Now re-add a positive fluent
    state.update_fluents({Fluent("free robot1")})
    expected = State(
        fluents={
            Fluent("free robot1"),
            Fluent("at robot1 kitchen"),
        }
    )
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
            ~Fluent("at", "?robot", "?loc_from"),
        },
    )

    # Ground it with a specific binding
    binding = {"?robot": "robot1", "?loc_from": "roomA", "?loc_to": "roomB"}
    grounded = lifted._ground(binding)

    # Check time is evaluated correctly
    assert grounded.time == 7.5

    # Check fluents are properly substituted
    expected_fluents = {
        Fluent("free", "robot1"),
        Fluent("at", "robot1", "roomB"),
        ~Fluent("at", "robot1", "roomA"),
    }
    assert grounded.resulting_fluents == expected_fluents


def test_effect_instantiation():
    f = Fluent("at r1 roomA")
    e = GroundedEffect(2.5, {f})
    pe = GroundedEffect(3.0, prob_effects=[(0.5, [e]), (0.5, [e])])
    assert len(pe.prob_effects) == 2
    for prob, e in pe.prob_effects:
        print(e)
        assert prob == 0.5
        assert len(e) == 1


def test_effect_hashing():
    f1 = F("at r1 roomA")
    f2 = F("at r1 roomB")
    e1 = GroundedEffect(2.5, {f1, f2})
    e1_alt = GroundedEffect(2.5, {f1, f2})
    e1_reordered = GroundedEffect(2.5, {f2, f1})
    e1_time_change = GroundedEffect(2.0, {f1, f2})
    e2 = GroundedEffect(2.0, {f2, f1})
    assert hash(e1) == hash(e1_alt)
    assert hash(e1) == hash(e1_reordered)
    assert not hash(e1) == hash(e1_time_change)
    assert not hash(e1) == hash(e2)
    # Doubled for caching
    assert hash(e1) == hash(e1_alt)
    assert hash(e1) == hash(e1_reordered)
    assert not hash(e1) == hash(e1_time_change)
    assert not hash(e1) == hash(e2)

    e1 = GroundedEffect(2.5, {f1})
    e2 = GroundedEffect(2.5, {f2})
    pea = GroundedEffect(3.0, prob_effects=[(0.5, [e1]), (0.5, [e2])])
    peb = GroundedEffect(3.0, prob_effects=[(0.5, [e2]), (0.5, [e1])])
    pe2 = GroundedEffect(3.0, prob_effects=[(0.5, [e2]), (0.5, [e2])])
    assert hash(pea) == hash(
        peb
    ), "Failed to generate order-independent hash for prob effects."
    assert not hash(pea) == hash(
        pe2
    ), "Failed to gen different hash for different prob effects."

    e1 = GroundedEffect(2.5, {f1})
    e2 = GroundedEffect(2.5, {f2})
    pea = GroundedEffect(3.0, prob_effects=[(0.5, [e1]), (0.5, [e2])])
    peb = GroundedEffect(3.5, prob_effects=[(0.5, [e1]), (0.5, [e2])])
    assert not hash(pea) == hash(peb), "Failed: has does not consider time of effect."

    e1 = GroundedEffect(2.5, {f1})
    e2 = GroundedEffect(2.5, {f2})
    pea = GroundedEffect(3.0, prob_effects=[(0.5, [e1]), (0.5, [e2])])
    peb = GroundedEffect(3.0, prob_effects=[(0.4, [e1]), (0.6, [e2])])
    assert not hash(pea) == hash(
        peb
    ), "Failed: has does not consider probability of effects."

    e1 = GroundedEffect(2.0, {f1})
    e2 = GroundedEffect(5.0, {f1})
    assert not hash(e1) == hash(e2)
    pea = GroundedEffect(3.0, prob_effects=[(1.0, [e1])])
    peb = GroundedEffect(3.0, prob_effects=[(1.0, [e2])])
    assert not hash(pea) == hash(
        peb
    ), "Failed: has does not consider time of probabilitstic effects."

    e1 = GroundedEffect(2.0, {f1})
    e2 = GroundedEffect(5.0, {f2})
    assert not hash(e1) == hash(e2)
    pea = GroundedEffect(3.0, prob_effects=[(0.4, [e1]), (0.6, [e2])])
    peb = GroundedEffect(3.0, prob_effects=[(0.4, [e2]), (0.6, [e1])])
    assert not hash(pea) == hash(
        peb
    ), "Failed: has does not consider associativity of probabilitstic effects."


def test_action_instantiation():
    a1 = Action(
        preconditions={Fluent("at roomA r1")},
        effects=[
            GroundedEffect(2.5, {Fluent("at roomB r1"), Fluent("not at roomA r1")})
        ],
        name="move r1 roomA roomB",
    )
    a2 = Action(
        preconditions={Fluent("at roomB r1")},
        effects=[
            GroundedEffect(2.5, {Fluent("at roomA r1"), Fluent("not at roomB r1")})
        ],
        name="move r1 roomB roomA",
    )
    s = State(fluents={Fluent("at roomA r1")})
    assert s.satisfies_precondition(a1)
    assert not s.satisfies_precondition(a2)

    transition(s, a1)  # Confirm this runs


def test_state_effects_pop_order_correct():
    state = State(fluents={Fluent("free robot")})
    action = Action(
        preconditions=set(),
        effects=[
            GroundedEffect(0, {Fluent("not free robot")}),
            GroundedEffect(2.5, {Fluent("free robot")}),
            GroundedEffect(3.0, {Fluent("next effect")}),
        ],
        name="queue effects",
    )
    state_out = transition(state, action)[0][0]
    assert len(state_out.upcoming_effects) == 1
    assert state_out.upcoming_effects[0][0] == 3.0
    assert Fluent("free robot") in state_out.fluents
    assert Fluent("next effect") not in state_out.fluents


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
            Fluent("free", "r2"),
        },
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


def test_deepcopy():
    import copy

    f = Fluent("at robot")
    f2 = copy.deepcopy(f)
    assert f.name == f2.name
    assert f.args == f2.args
    assert f.negated == f2.negated
    assert f == f2

    ge = GroundedEffect(2.0, {Fluent("at robot counter")})
    ge_copy = copy.deepcopy(ge)
    assert hash(ge) == hash(ge_copy)

    ge_alt = GroundedEffect(3.0, {Fluent("at robot cabinet")})
    ge_prob = GroundedEffect(3.0, prob_effects=[(0.4, [ge]), (0.6, [ge_alt])])
    ge_prob_copy = copy.deepcopy(ge_prob)
    assert hash(ge_prob) == hash(ge_prob_copy)

    # Get all actions
    objects_by_type = {
        "robot": ["r1", "r2"],
        "location": ["start", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
    }
    random.seed(8616)
    move_op = construct_move_operator(lambda *args: 5.0 + random.random())
    all_actions = move_op.instantiate(objects_by_type)

    # Initial state
    initial_state = State(
        time=0,
        fluents={
            F("at r1 start"),
            F("free r1"),
            F("at r2 start"),
            F("free r2"),
            F("visited start"),
        },
    )

    # Test deepcopy
    all_actions = [copy.deepcopy(a) for a in all_actions]
    initial_state = copy.deepcopy(initial_state)


def test_search_sequence():
    # Define objects
    objects_by_type = {
        "robot": ["r1"],
        "location": ["roomA", "roomB"],
        "object": ["cup", "bowl"],
    }

    def object_search_prob(robot, search_loc, obj):
        if obj == "cup":
            return 0.8
        else:
            return 0.6

    # Ground actions
    search_actions = construct_search_operator(object_search_prob, 5.0, 3).instantiate(
        objects_by_type
    )
    # Initial state
    initial_state = State(
        time=0,
        fluents={
            Fluent("at r1 roomA"),
            Fluent("free r1"),
        },
    )

    # Select action: search r1 roomA roomB cup
    action_1 = get_action_by_name(search_actions, "search r1 roomA roomB cup")
    for eff in action_1.effects:
        if eff.is_probabilistic:
            assert len(eff.prob_effects) > 1
            if eff.prob_effects[0].prob == eff.prob_effects[1].prob:
                continue
            assert not hash(eff.prob_effects[0]) == hash(
                eff.prob_effects[1]
            ), "Hashes for the probabilistic effects must be different!"
            for peff in eff.prob_effects:
                print(peff.prob)
                print(peff.effects)
            break
    else:
        raise ValueError("At least one effect must be probabilistic.")

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
    action_2 = get_action_by_name(search_actions, "search r1 roomB roomA bowl")
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


def test_action_extra_cost():
    """Test that extra_cost is properly passed from Operator to Action."""
    # Define a simple operator with extra_cost
    test_operator = Operator(
        name="test_action",
        parameters=[("?robot", "robot"), ("?loc", "location")],
        preconditions=[Fluent("at", "?robot", "?loc")],
        effects=[
            Effect(
                time=1.0,
                resulting_fluents={Fluent("tested", "?robot")}
            )
        ],
        extra_cost=5.0
    )

    # Instantiate the operator
    objects_by_type = {
        "robot": ["r1"],
        "location": ["roomA"],
    }
    actions = test_operator.instantiate(objects_by_type)

    # Verify that the action has the extra_cost
    assert len(actions) == 1
    action = actions[0]
    assert action.extra_cost == 5.0
    assert action.name == "test_action r1 roomA"

    # Test with default extra_cost (should be 0.0)
    default_operator = Operator(
        name="default_action",
        parameters=[("?robot", "robot")],
        preconditions=[Fluent("free", "?robot")],
        effects=[Effect(time=1.0, resulting_fluents={Fluent("done", "?robot")})]
    )

    default_actions = default_operator.instantiate({"robot": ["r1"]})
    assert len(default_actions) == 1
    assert default_actions[0].extra_cost == 0.0

    # Test creating Action directly with extra_cost
    direct_action = Action(
        preconditions={Fluent("at r1 roomA")},
        effects=[GroundedEffect(1.0, {Fluent("tested r1")})],
        name="direct_action",
        extra_cost=10.0
    )
    assert direct_action.extra_cost == 10.0

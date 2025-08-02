import pytest
from mrppddl.core import GroundedEffect, Fluent, Action, State, transition, get_action_by_name
from mrppddl.helper import construct_move_operator, construct_move_visited_operator
import random

F = Fluent

def test_cpp_fluent_equality():
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


def test_cpp_fluent_equality_hash():

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

def test_cpp_effect_instantiation():
    f = Fluent("at r1 roomA")
    e = GroundedEffect(2.5, {f})
    pe = GroundedEffect(3.0, prob_effects=[(0.5, [e]), (0.5, [e])])
    assert len(pe.prob_effects) == 2
    for prob, e in pe.prob_effects:
        print(e)
        assert prob == 0.5
        assert len(e) == 1

def test_cpp_effect_hashing():
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
    assert hash(pea) == hash(peb), "Failed to generate order-independent hash for prob effects."
    assert not hash(pea) == hash(pe2), "Failed to gen different hash for different prob effects."

    e1 = GroundedEffect(2.5, {f1})
    e2 = GroundedEffect(2.5, {f2})
    pea = GroundedEffect(3.0, prob_effects=[(0.5, [e1]), (0.5, [e2])])
    peb = GroundedEffect(3.5, prob_effects=[(0.5, [e1]), (0.5, [e2])])
    assert not hash(pea) == hash(peb), "Failed: has does not consider time of effect."

    e1 = GroundedEffect(2.5, {f1})
    e2 = GroundedEffect(2.5, {f2})
    pea = GroundedEffect(3.0, prob_effects=[(0.5, [e1]), (0.5, [e2])])
    peb = GroundedEffect(3.0, prob_effects=[(0.4, [e1]), (0.6, [e2])])
    assert not hash(pea) == hash(peb), "Failed: has does not consider probability of effects."

    e1 = GroundedEffect(2.0, {f1})
    e2 = GroundedEffect(5.0, {f1})
    assert not hash(e1) == hash(e2)
    pea = GroundedEffect(3.0, prob_effects=[(1.0, [e1])])
    peb = GroundedEffect(3.0, prob_effects=[(1.0, [e2])])
    assert not hash(pea) == hash(peb), "Failed: has does not consider time of probabilitstic effects."

    e1 = GroundedEffect(2.0, {f1})
    e2 = GroundedEffect(5.0, {f2})
    assert not hash(e1) == hash(e2)
    pea = GroundedEffect(3.0, prob_effects=[(0.4, [e1]), (0.6, [e2])])
    peb = GroundedEffect(3.0, prob_effects=[(0.4, [e2]), (0.6, [e1])])
    assert not hash(pea) == hash(peb), "Failed: has does not consider associativity of probabilitstic effects."

def test_cpp_action_instantiation():
    a1 = Action(preconditions={Fluent("at roomA r1")},
               effects=[GroundedEffect(2.5, {Fluent("at roomB r1"), Fluent("not at roomA r1")})],
               name="move r1 roomA roomB")
    a2 = Action(preconditions={Fluent("at roomB r1")},
               effects=[GroundedEffect(2.5, {Fluent("at roomA r1"), Fluent("not at roomB r1")})],
               name="move r1 roomB roomA")
    s = State(fluents={Fluent("at roomA r1")})
    assert s.satisfies_precondition(a1)
    assert not s.satisfies_precondition(a2)

    transition(s, a1)  # Confirm this runs

def test_state_effects_pop_order_correct():
    state = State(fluents={Fluent("free robot")})
    action = Action(preconditions=set(),
                    effects=[
                        GroundedEffect(0, {Fluent("not free robot")}),
                        GroundedEffect(2.5, {Fluent("free robot")}),
                        GroundedEffect(3.0, {Fluent("next effect")}),
                    ], name="queue effects")
    state_out = transition(state, action)[0][0]
    assert len(state_out.upcoming_effects) == 1
    assert state_out.upcoming_effects[0][0] == 3.0
    assert Fluent("free robot") in state_out.fluents
    assert Fluent("next effect") not in state_out.fluents



# DEBUG

from typing import List, Tuple, Sequence, Dict, Set
from mrppddl._bindings import astar, MCTSPlanner, make_goal_test

def test_cpp_deepcopy():
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
        "location": ["start", "a", "b", "c",
                     "d", "e", "f", "g", "h",
                     "i", "j", "k"],
    }
    random.seed(8616)
    move_op = construct_move_operator(lambda *args: 5.0 + random.random())
    all_actions = move_op.instantiate(objects_by_type)

    # Initial state
    initial_state = State(
        time=0,
        fluents={
            F("at r1 start"), F("free r1"),
            F("at r2 start"), F("free r2"),
            F("visited start"),
        })

    # Test deepcopy
    all_actions = [copy.deepcopy(a) for a in all_actions]
    initial_state = copy.deepcopy(initial_state)


@pytest.mark.parametrize(
    "initial_fluents", [
        {F("at r1 start"), F("free r1"),
         F("visited start")},
        {F("at r1 start"), F("free r1"),
         F("at r2 start"), F("free r2"),
         F("visited start")},
        {F("at r1 start"), F("free r1"),
         F("at r2 start"), F("free r2"),
         F("at r3 start"), F("free r3"),
         F("visited start")},
    ],
    ids=["one robot", "two robots", "three robots"])
def test_planner_mcts_move_visit_multirobot(initial_fluents):
    # Get all actions
    objects_by_type = {
        "robot": ["r1", "r2", "r3"],
        "location": ["start", "a", "b", "c",
                     "d", "e", "f", "g", "h",
                     "i", "j", "k"],
    }
    random.seed(8616)
    move_op = construct_move_visited_operator(lambda *args: 5.0 + random.random())
    all_actions = move_op.instantiate(objects_by_type)

    # Initial state
    initial_state = State(
        time=0,
        fluents=initial_fluents)
    goal_fluents = {
        F("visited a"),
        F("visited b"),
        F("visited c"),
        F("visited d"),
        F("visited e"),
    }
    def is_goal(state):
        return all(gf in state.fluents
                   for gf in goal_fluents)
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

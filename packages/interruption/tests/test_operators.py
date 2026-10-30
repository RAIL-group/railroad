from railroad.core import Fluent as F, State, get_action_by_name, transition

from interruption.operators import (
    BLOCKING_DURATION, construct_brew_coffee_operator,
    construct_pick_with_left_hand_operator_blocking, construct_wash_operator,
    construct_wash_operator_blocking, only_clears_blocking_facts,
)

OBJECTS = {"robot": {"robot1"}, "location": {"coffee_maker", "sink"}}


def _brew_actions(brew_time=60, start_time=2):
    return construct_brew_coffee_operator(brew_time, start_time).instantiate(OBJECTS)


def _start_state(*extra):
    return State(time=0, fluents={
        F("at robot1 coffee_maker"), F("free robot1"), F("coffee-maker coffee_maker"), *extra
    })


def test_brew_coffee_frees_the_robot_before_the_coffee_is_brewed():
    action = get_action_by_name(_brew_actions(), "brew-coffee robot1 coffee_maker")
    (state, prob), = transition(_start_state(), action)

    assert prob == 1
    # the robot is free once it has started the coffee maker, not after the brew
    assert state.time == 2
    assert F("free robot1") in state.fluents
    assert F("brewing coffee_maker") in state.fluents
    assert F("brewed coffee_maker") not in state.fluents
    # the brew finishes on its own, 60 after the action started
    (time, effect), = state.upcoming_effects
    assert time == 60
    assert effect.resulting_fluents == {~F("brewing coffee_maker"), F("brewed coffee_maker")}


def test_brew_coffee_only_applies_at_a_free_robots_coffee_maker():
    actions = _brew_actions()
    state = _start_state()

    def applicable(name):
        return state.satisfies_precondition(get_action_by_name(actions, name))

    assert applicable("brew-coffee robot1 coffee_maker")
    assert not applicable("brew-coffee robot1 sink")  # not a coffee maker
    state = _start_state(F("brewed coffee_maker"))
    assert not applicable("brew-coffee robot1 coffee_maker")  # already brewed
    state = _start_state(F("brewing coffee_maker"))
    assert not applicable("brew-coffee robot1 coffee_maker")  # still brewing
    state = State(time=0, fluents={F("at robot1 coffee_maker"), F("coffee-maker coffee_maker")})
    assert not applicable("brew-coffee robot1 coffee_maker")  # robot busy



WASH_OBJECTS = {"robot": {"robot1"}, "location": {"sink", "countertop"}, "object": {"mug"}}


def _wash_state(*extra):
    return State(time=0, fluents={
        F("at robot1 sink"), F("free robot1"), F("sink sink"), F("at mug sink"), F("dirty mug"),
        *extra,
    })


def test_wash_keeps_the_robot_busy_until_the_object_is_clean():
    actions = construct_wash_operator(15).instantiate(WASH_OBJECTS)
    action = get_action_by_name(actions, "wash robot1 sink mug")
    (state, prob), = transition(_wash_state(), action)

    assert prob == 1
    assert state.time == 15
    assert F("free robot1") in state.fluents
    assert F("dirty mug") not in state.fluents
    assert F("at mug sink") in state.fluents
    assert not state.upcoming_effects


def test_wash_only_applies_to_a_dirty_object_in_a_sink():
    actions = construct_wash_operator(15).instantiate(WASH_OBJECTS)
    action = get_action_by_name(actions, "wash robot1 sink mug")
    at_countertop = get_action_by_name(actions, "wash robot1 countertop mug")

    assert _wash_state().satisfies_precondition(action)
    at_countertop_state = State(time=0, fluents={
        F("at robot1 countertop"), F("free robot1"), F("sink sink"), F("at mug countertop"),
        F("dirty mug"),
    })
    assert not at_countertop_state.satisfies_precondition(at_countertop)  # not a sink
    clean = State(time=0, fluents={
        F("at robot1 sink"), F("free robot1"), F("sink sink"), F("at mug sink"),
    })
    assert not clean.satisfies_precondition(action)
    mug_elsewhere = State(time=0, fluents={
        F("at robot1 sink"), F("free robot1"), F("sink sink"), F("at mug countertop"),
        F("dirty mug"),
    })
    assert not mug_elsewhere.satisfies_precondition(action)


def test_blocking_wash_sets_just_washed_until_its_clean_up():
    actions = construct_wash_operator_blocking(15).instantiate(WASH_OBJECTS)
    action = get_action_by_name(actions, "wash robot1 sink mug")
    (state, _), = transition(_wash_state(), action)

    assert state.time == 15
    assert F("free robot1") in state.fluents
    assert F("dirty mug") not in state.fluents
    assert F("just-washed robot1 mug") in state.fluents
    (clean_up_time, clean_up), = state.upcoming_effects
    assert clean_up_time == 15 + BLOCKING_DURATION
    assert clean_up.resulting_fluents == {~F("just-washed robot1 mug")}
    assert only_clears_blocking_facts(clean_up)


def test_blocking_pick_cant_pick_an_object_straight_after_washing_it():
    objects = {**WASH_OBJECTS, "category": {"basin"}}
    actions = construct_pick_with_left_hand_operator_blocking(2).instantiate(objects)
    pick = get_action_by_name(actions, "pick-left robot1 basin sink mug")
    state = State(time=0, fluents={
        F("at robot1 sink"), F("free robot1"), F("at mug sink"), F("category sink basin"),
    })

    assert state.satisfies_precondition(pick)
    blocked = State(time=0, fluents={*state.fluents, F("just-washed robot1 mug")})
    assert not blocked.satisfies_precondition(pick)

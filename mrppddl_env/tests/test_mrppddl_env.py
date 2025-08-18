import pytest
from copy import copy
from mrppddl.core import Fluent, State, get_action_by_name
from mrppddl.core import transition
from mrppddl.helper import construct_move_operator

from mrppddl.core import OptCallable, Operator, Effect
from mrppddl.helper import _make_callable, _invert_prob

F = Fluent

def construct_search_operator(
    object_find_prob: OptCallable, move_time: OptCallable) -> Operator:
    object_find_prob = _make_callable(object_find_prob)
    inv_object_find_prob = _invert_prob(object_find_prob)
    move_time = _make_callable(move_time)
    return Operator(
        name="search",
        parameters=[("?robot", "robot"),
                    ("?loc_from", "location"),
                    ("?loc_to", "location"),
                    ("?object", "object")],
        preconditions=[F("free ?robot"),
                       F("at ?robot ?loc_from"),
                       F("not searched ?loc_to ?object"),
                       F("not revealed ?loc_to"),
                       F("not found ?object")],
        effects=[
            Effect(time=0, resulting_fluents={
                F("not free ?robot"),
                F("not at ?robot ?loc_from")}),
            Effect(time=(move_time, ["?robot", "?loc_from", "?loc_to"]),
                   resulting_fluents={F("at ?robot ?loc_to"),
                                      F("searched ?loc_to ?object"),
                                      F("free ?robot"),
},
                   prob_effects=[(
                       (object_find_prob, ["?robot", "?loc_to", "?object"]),
                       [Effect(time=0, resulting_fluents={F("found ?object")})]
                   ), (
                       (inv_object_find_prob, ["?robot", "?loc_to", "?object"]), [],
                   )])
        ],
    )



def test_search_reveal_simple():
    """This is a simple scenario showing that we can reveal locations in the
    simulator by searching them with a simple 'search' action from the robot. We
    show that, with a single robot, assigning the search action and then
    advancing time until something is revealed.

    We will need to handle that the robot thinks it could be in multiple states
    right now, but in fact that is not possible, fate determined by the
    environment.
    """

    def move_time(robot: str, loc_from: str, loc_to: str) -> float:
        """Robot 1 is faster than robot 2."""
        return 3.0 if robot == "r1" else 5.0

    def object_find_prob(robot: str, loc: str, obj: str):
        return 0.5

    objects_at_locations = {
        "start": dict(),
        "roomA": {"object": {"objA", "objC"}},
        "roomB": {"object": {"objB"}},
    }

    objects_by_type = {
        "robot": set(["r1", "r2"]),
        "location": set(["start", "roomA", "roomB"]),
        "object": set(["objA", "objB"]),
    }
    def _get_actions_from_objects(objects_by_type):
        search_op = construct_search_operator(object_find_prob, move_time)
        return search_op.instantiate(objects_by_type)

    def reveal(states_and_probs, objects_by_type):
        # The 'pessimistic' state is the one with the union of predicates.
        states = [s for s, _ in states_and_probs]
        new_fluents = set([fluent for fluent in states[0].fluents
                           if all(fluent in s.fluents for s in states[1:])])
        new_upcoming = [up_eff for up_eff in states[0].upcoming_effects
                        if all(up_eff in s.upcoming_effects for s in states[1:])]
        # for locations 'searched' but not 'revealed'. The set of objects should
        # contain all objects in the revealed locations

        # Update the state
        ## What are the searched locations that are not yet revealed?
        def _is_location_searched(location, new_fluents):
            return any(f.name == "searched" and f.args[0] == location
                       for f in new_fluents)

        newly_revealed_locations = [
            location for location in objects_by_type["location"]
            if _is_location_searched(location, new_fluents) 
            and F(f"revealed {location}") not in new_fluents
        ]

        new_objects_by_type = copy(objects_by_type)
        for location in newly_revealed_locations:
            new_fluents.add(F(f"revealed {location}"))
            # Add the objects in the newly_revealed_locations
            for obj_type, objects in objects_at_locations[location].items():
                new_objects_by_type[obj_type] |= set(objects)
                for obj in objects:
                    new_fluents.add(F(f"found {obj}"))
                    new_fluents.add(F(f"at {location} {obj}"))

        new_state = State(states[0].time, new_fluents, new_upcoming)
        return new_state, new_objects_by_type

    state = State(time=0, fluents={
        F("revealed start"),
        F("at r1 start"), F("free r1"),
        F("at r2 start"), F("free r2"),
    })
    all_actions = _get_actions_from_objects(objects_by_type)

    # First action gives r1 an assignment
    a1 = get_action_by_name(all_actions, "search r1 start roomA objA")
    states = transition(state, a1)
    state, objects_by_type = reveal(states, objects_by_type)
    assert F("free r1") not in state.fluents
    assert F("free r2") in state.fluents

    # Second action gives r2 an assignment & moves until r1 free again
    a2 = get_action_by_name(all_actions, "search r2 start roomB objB")
    states = transition(state, a2)
    state, objects_by_type = reveal(states, objects_by_type)
    assert F("free r1") in state.fluents
    assert F("free r2") not in state.fluents
    assert F("revealed roomA") in state.fluents
    assert F("found objA") in state.fluents
    assert F("found objC") in state.fluents

    a3 = get_action_by_name(all_actions, "search r1 roomA roomB objB")
    states = transition(state, a3)
    state, objects_by_type = reveal(states, objects_by_type)
    assert F("free r1") not in state.fluents
    assert F("free r2") in state.fluents
    assert F("revealed roomA") in state.fluents
    assert F("revealed roomB") in state.fluents
    assert F("found objA") in state.fluents
    assert F("found objC") in state.fluents

import pytest
import numpy as np
from dataclasses import dataclass
from copy import copy

from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl.core import transition
from mrppddl.helper import construct_move_operator

from mrppddl.core import OptCallable, Operator, Effect
from mrppddl.helper import _make_callable, _invert_prob

from typing import Dict, Set, List, Tuple, Callable
import environments
from environments.simulator import Simulator


class TestEnvironment:
    def __init__(self):
        self.locations = {
            "start": np.array([0, 0]),
            "roomA": np.array([10, 0]),
            "roomB": np.array([0, 15]),
            "roomC": np.array([15, 15]),
        }

        self.objects_at_locations = {
            "start": dict(),
            "roomA": {"object": {"objA", "objC"}},
            "roomB": {"object": {"objB"}},
            "roomC": {"object": {"objD"}},
        }


    def get_move_time_fn(self):
        def get_move_time(robot, loc_from, loc_to):
            distance = np.linalg.norm(self.locations[loc_from] - self.locations[loc_to])
            return distance
        return get_move_time

    def get_intermediate_coordinates(self, time, loc_from, loc_to):
        coord_from = self.locations[loc_from]
        coord_to = self.locations[loc_to]
        direction = (coord_to - coord_from) / np.linalg.norm(coord_to - coord_from)
        new_coord = coord_from + direction * time
        return new_coord


def test_move_action():
    '''Test that move action is interrupted correctly.'''
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": {"start", "roomA", "roomB", "roomC"},
    }

    initial_state = State(
            time=0,
            fluents={
                F("at", "r1", "start"),
                F("at", "r2", "start"),
                F("free", "r1"),
                F("free", "r2"),
            },
    )
    env = TestEnvironment()
    move_op = environments.actions.construct_move_operator(move_time=env.get_move_time_fn())

    sim = Simulator(initial_state, objects_by_type, [move_op], env)
    actions = sim.get_actions()
    a1 = get_action_by_name(actions, "move r1 start roomA")
    sim.advance(a1)
    a2 = get_action_by_name(actions, "move r2 start roomB")
    sim.advance(a2)

    assert F("free r1") in sim.state.fluents
    assert F("free r2") in sim.state.fluents
    assert F("at r1 start") not in sim.state.fluents
    assert F("at r2 start") not in sim.state.fluents
    assert F("at r1 roomA") in sim.state.fluents
    assert F("at r2 roomB") not in sim.state.fluents
    assert F("at r2 r2_loc") in sim.state.fluents
    assert F("at r1 r1_loc") not in sim.state.fluents
    assert sim.state.time == 10.0
    assert len(sim.ongoing_actions) == 0
    assert np.all(sim.environment.locations["r2_loc"] == (0.0, 10.0))


def test_search_action():
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": {"start", "roomA", "roomB", "roomC"},
        "object": {"objA", "objB"}
    }

    search_time = lambda r, l: 10 if r == "r1" else 15
    object_find_prob = lambda r, l, o: 0.8 if l == "roomA" else 0.2

    initial_state = State(
            time=0,
            fluents={
                F("at", "r1", "roomA"),
                F("at", "r2", "roomB"),
                F("free", "r1"),
                F("free", "r2"),
            },
    )
    env = TestEnvironment()
    search_op = environments.actions.construct_search_operator(object_find_prob=object_find_prob,
                                                               search_time=search_time)

    sim = Simulator(initial_state, objects_by_type, [search_op], env)

    actions = sim.get_actions()
    a1 = get_action_by_name(actions, "search r1 roomA objA")
    sim.advance(a1)
    a2 = get_action_by_name(actions, "search r2 roomB objB")
    sim.advance(a2)

    assert F("free r1") in sim.state.fluents
    assert F("at r1 roomA") in sim.state.fluents
    assert F("searched roomA objA") in sim.state.fluents
    assert F("revealed roomA") in sim.state.fluents
    assert F("found objA") in sim.state.fluents
    assert F("found objC") in sim.state.fluents
    assert F("at objA roomA") in sim.state.fluents
    assert F("at objC roomA") in sim.state.fluents
    assert F("lock-search roomA") not in sim.state.fluents

    assert F("free r2") not in sim.state.fluents
    assert F("at r2 roomB") in sim.state.fluents
    assert F("searched roomB objB") not in sim.state.fluents
    assert F("found objB") not in sim.state.fluents
    assert F("lock-search roomB") in sim.state.fluents
    assert sim.state.time == 10

    assert len(sim.ongoing_actions) == 1


def test_pick_and_place_action():
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": {"start", "roomA", "roomB", "roomC"},
        "object": {"objA", "objB"}
    }

    pick_time = lambda r, l, o: 10 if r == "r1" else 15
    place_time = lambda r, l, o: 10 if r == "r1" else 15

    initial_state = State(
            time=0,
            fluents={
                F("at", "r1", "roomA"), F("at", "objA", "roomA"), F("at", "objC", "roomA"),
                F("at", "r2", "roomB"), F("at", "objB", "roomB"),
                F("free", "r1"),
                F("free", "r2"),
            },
    )
    env = TestEnvironment()
    pick_op = environments.actions.construct_pick_operator(pick_time=pick_time)
    place_op = environments.actions.construct_place_operator(place_time=place_time)

    sim = Simulator(initial_state, objects_by_type, [pick_op, place_op], env)

    actions = sim.get_actions()
    a1 = get_action_by_name(actions, "pick r1 roomA objA")
    sim.advance(a1)
    a2 = get_action_by_name(actions, "pick r2 roomB objB")
    sim.advance(a2)

    # R1 finishes picking objA first
    assert F("free r1") in sim.state.fluents
    assert F("at r1 roomA") in sim.state.fluents
    assert F("holding r1 objA") in sim.state.fluents
    assert F("at objA roomA") not in sim.state.fluents
    assert "objA" not in env.objects_at_locations["roomA"]["object"]
    # R2 is still picking objB
    assert F("free r2") not in sim.state.fluents
    assert F("at r2 roomB") in sim.state.fluents
    assert F("holding r2 objB") not in sim.state.fluents
    assert F("at objB roomB") not in sim.state.fluents
    assert sim.state.time == 10
    assert len(sim.ongoing_actions) == 1

    a3 = get_action_by_name(actions, "place r1 roomA objA")
    sim.advance(a3)
    # R2 finishes picking objB
    assert F("free r2") in sim.state.fluents
    assert F("at r2 roomB") in sim.state.fluents
    assert F("holding r2 objB") in sim.state.fluents
    assert F("at objB roomB") not in sim.state.fluents
    assert "objB" not in env.objects_at_locations["roomB"]["object"]
    # R1 is still placing objA
    assert F("free r1") not in sim.state.fluents
    assert F("at r1 roomA") in sim.state.fluents
    assert F("holding r1 objA") not in sim.state.fluents
    assert F("at objA roomA") not in sim.state.fluents
    assert sim.state.time == 15
    assert len(sim.ongoing_actions) == 1

    a4 = get_action_by_name(actions, "place r2 roomB objB")
    sim.advance(a4)
    # R1 finishes placing objA
    assert F("free r1") in sim.state.fluents
    assert F("at r1 roomA") in sim.state.fluents
    assert F("holding r1 objA") not in sim.state.fluents
    assert F("at objA roomA") in sim.state.fluents
    assert "objA" in env.objects_at_locations["roomA"]["object"]
    # R2 is still placing objB
    assert F("free r2") not in sim.state.fluents
    assert F("at r2 roomB") in sim.state.fluents
    assert F("holding r2 objB") not in sim.state.fluents
    assert F("at objB roomB") not in sim.state.fluents
    assert sim.state.time == 20
    assert len(sim.ongoing_actions) == 1

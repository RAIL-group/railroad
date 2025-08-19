import pytest
from copy import copy
from mrppddl.core import Fluent, State, get_action_by_name
from mrppddl.core import transition
from mrppddl.helper import construct_move_operator

from mrppddl.core import OptCallable, Operator, Effect
from mrppddl.helper import _make_callable, _invert_prob

from typing import Dict, Set, List, Tuple
import itertools

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
                       F("not lock-search ?loc_to"),
                       F("not searched ?loc_to ?object"),
                       F("not revealed ?loc_to"),
                       F("not found ?object")],
        effects=[
            Effect(time=0, resulting_fluents={
                F("not free ?robot"),
                F("lock-search ?loc_to"),
                F("not at ?robot ?loc_from")}),
            Effect(time=(move_time, ["?robot", "?loc_from", "?loc_to"]),
                   resulting_fluents={F("at ?robot ?loc_to"),
                                      F("not lock-search ?loc_to"),
                                      F("searched ?loc_to ?object"),
                                      F("free ?robot"),},
                   prob_effects=[(
                       (object_find_prob, ["?robot", "?loc_to", "?object"]),
                       [Effect(time=0, resulting_fluents={F("found ?object")})]
                   ), (
                       (inv_object_find_prob, ["?robot", "?loc_to", "?object"]), [],
                   )])
        ],
    )

class UpcomingAction:
    def __init__(self, time, action):
        self.time = time
        self.name = action.name
        self._action = action
        self._upcoming_effects = sorted([
            (time + eff.time, eff) for eff in action.effects
        ], key=lambda el: el[0])

    @property
    def time_to_next_event(self):
        if self._upcoming_effects:
            return self._upcoming_effects[0][0]
        else:
            return float('inf')

    @property
    def is_done(self):
        return not self.upcoming_effects

    @property
    def upcoming_effects(self):
        # Return remaining upcoming events
        return self._upcoming_effects

    def advance(self, time):
        # Update the internal time
        self.time = time

        # Pop and return all effects scheduled at or before the new time
        new_effects = []
        EPS = 1e-9
        i = 0
        while i < len(self._upcoming_effects) and self._upcoming_effects[i][0] <= time + EPS:
            new_effects.append(self._upcoming_effects[i])
            i += 1

        # remove the new_effects from upcoming_effects
        self._upcoming_effects = self._upcoming_effects[i:]
        return new_effects

    def __str__(self):
        return f"UpcomingAction<{self.name}, {self.time}, {self.upcoming_effects}>"
    

class Simulator:
    """Tiny wrapper around the PDDL core that (i) applies an action via
    `transition` and (ii) deterministically 'reveals' searched locations by
    intersecting outcomes across probabilistic branches.
    """

    def __init__(
        self,
        initial_state: State,
        objects_by_type: Dict[str, Set[str]],
        objects_at_locations: Dict[str, Dict[str, Set[str]]],
        operators: List[Operator],
    ):
        self._state = initial_state
        self.objects_by_type = {k: set(v) for k, v in objects_by_type.items()}
        self.objects_at_locations = objects_at_locations
        self.operators = operators
        self.ongoing_actions = []

    def get_actions(self) -> List:
        """Instantiate an Operator under the *current* objects_by_type."""
        return list(itertools.chain.from_iterable(
            operator.instantiate(self.objects_by_type)
            for operator in self.operators
        ))

    def advance(self, action) -> State:
        """Add a new action and then advance as much as possible using both `transition` and
        also `_reveal` as needed once areas are searched."""
        self.ongoing_actions.append(UpcomingAction(self._state.time, action))

        def _any_free_robots(state):
            return any(f.name == "free" for f in state.fluents)

        # Loop at least once to advance "time == now" actions
        robot_free = False
        while not robot_free and self.ongoing_actions:
            # If there are no actions that are not done, break.
            if not any(not act.is_done for act in self.ongoing_actions):
                break

            # Get the time we need to advance. If a robot needs an action the time is "now"
            robot_free = _any_free_robots(self._state)
            if robot_free:
                adv_time = self.time
            else:
                adv_time = min(act.time_to_next_event for act in self.ongoing_actions)

            # Get new active effects to add to the state, then transition and _reveal.
            new_effects = list(itertools.chain.from_iterable(
                [act.advance(adv_time) for act in self.ongoing_actions]
            ))
            new_state = State(self._state.time,
                              self._state.fluents,
                              sorted(self._state.upcoming_effects + new_effects,
                                     key=lambda el: el[0]))
            self._state, self.objects_by_type = self._reveal(transition(new_state, None))
            robot_free = _any_free_robots(self._state)

            # Remove any actions that are now done
            self.ongoing_actions = [act for act in self.ongoing_actions
                                    if not act.is_done]

        # Return the resulting state (for planning)
        return self.state

    @property
    def time(self):
        return self._state.time

    @property
    def state(self):
        """The state is the internal state with future effects added."""
        effects = []
        for act in self.ongoing_actions:
            effects += act.upcoming_effects
        self.ongoing_actions = [
            act for act in self.ongoing_actions
            if not act.is_done
        ]
        return State(
            self._state.time,
            self._state.fluents,
            sorted(self._state.upcoming_effects + effects,
                   key=lambda el: el[0])
        )

    def _reveal(self, states_and_probs: List[Tuple[State, float]]):
        """Determinize by intersecting outcomes and reveal searched locations."""
        states = [s for s, _ in states_and_probs]

        # Intersection over outcomes (pessimistic/common information)
        base = states[0]
        new_fluents = {fl for fl in base.fluents if all(fl in s.fluents for s in states[1:])}
        new_upcoming = [ue for ue in base.upcoming_effects if all(ue in s.upcoming_effects for s in states[1:])]

        def _is_location_searched(location: str, fluents: Set[Fluent]) -> bool:
            # Fluent 'searched' looks like: F("searched <loc> <obj>")
            return any(f.name == "searched" and f.args[0] == location for f in fluents)

        # Locations that have been searched but not yet marked revealed
        newly_revealed_locations = [
            loc
            for loc in self.objects_by_type.get("location", set())
            if _is_location_searched(loc, new_fluents) and F(f"revealed {loc}") not in new_fluents
        ]

        new_objects_by_type = copy(self.objects_by_type)
        for loc in newly_revealed_locations:
            new_fluents.add(F(f"revealed {loc}"))

            # Add every object present at this location to type universe and mark as found/located
            for obj_type, objs in self.objects_at_locations.get(loc, {}).items():
                new_objects_by_type.setdefault(obj_type, set()).update(objs)
                for obj in objs:
                    new_fluents.add(F(f"found {obj}"))
                    new_fluents.add(F(f"at {loc} {obj}"))

        new_state = State(states[0].time, new_fluents, new_upcoming)
        return new_state, new_objects_by_type


def test_search_reveal_simple_with_simulator():
    # Dynamics
    def move_time(robot: str, loc_from: str, loc_to: str) -> float:
        return 3.0 if robot == "r1" else 5.0

    def object_find_prob(robot: str, loc: str, obj: str):
        return 0.5

    # World
    objects_at_locations = {
        "start": dict(),
        "roomA": {"object": {"objA", "objC"}},
        "roomB": {"object": {"objB"}},
        "roomC": {"object": {"objD"}},
    }
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": {"start", "roomA", "roomB", "roomC"},
        "object": {"objA", "objB"},  # objC will be revealed from roomA
    }

    # Operators
    search_op = construct_search_operator(object_find_prob, move_time)

    # Initial state
    init_state = State(
        time=0,
        fluents={
            F("revealed start"),
            F("at r1 start"), F("free r1"),
            F("at r2 start"), F("free r2"),
        },
    )

    # Simulator
    sim = Simulator(init_state, objects_by_type, objects_at_locations, [search_op])

    # 1) Assign r1 to search roomA for objA
    actions = sim.instantiate(search_op)
    a1 = get_action_by_name(actions, "search r1 start roomA objA")
    sim.advance(a1)
    assert F("free r1") not in sim.state.fluents
    assert F("free r2") in sim.state.fluents

    # 2) Assign r2 to search roomB for objB; time propagates until r1 is free
    actions = sim.instantiate(search_op)  # re-instantiate after object set may have changed
    a2 = get_action_by_name(actions, "search r2 start roomB objB")
    sim.advance(a2)
    assert F("free r1") in sim.state.fluents
    assert F("free r2") not in sim.state.fluents
    assert F("revealed roomA") in sim.state.fluents
    assert F("found objA") in sim.state.fluents
    assert F("found objC") in sim.state.fluents  # revealed with roomA

    # 3) Assign r1 to search roomB for objB (from roomA to roomB)
    actions = sim.instantiate(search_op)
    a3 = get_action_by_name(actions, "search r1 roomA roomC objB")
    sim.advance(a3)
    assert F("free r1") not in sim.state.fluents
    assert F("free r2") in sim.state.fluents
    assert F("revealed roomA") in sim.state.fluents
    assert F("revealed roomB") in sim.state.fluents
    assert F("found objA") in sim.state.fluents
    assert F("found objC") in sim.state.fluents

    print(a3)
    print(sim.state)

    raise NotImplementedError()

# TODO:
# - Pass operators to the simulator and make a 'get all actions' based on the revealed objects.
# - Add special

def test_upcoming_action():
    # Dynamics
    def move_time(robot: str, loc_from: str, loc_to: str) -> float:
        return 3.0 if robot == "r1" else 5.0

    def object_find_prob(robot: str, loc: str, obj: str):
        return 0.5

    # World
    objects_at_locations = {
        "start": dict(),
        "roomA": {"object": {"objA", "objC"}},
        "roomB": {"object": {"objB"}},
        "roomC": {"object": {"objD"}},
    }
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": {"start", "roomA", "roomB", "roomC"},
        "object": {"objA", "objB"},
    }

    # Operators
    search_op = construct_search_operator(object_find_prob, move_time)

    # Initial state
    init_state = State(
        time=0,
        fluents={
            F("revealed start"),
            F("at r1 start"), F("free r1"),
            F("at r2 start"), F("free r2"),
        },
    )

    # Simulator
    sim = Simulator(init_state, objects_by_type, objects_at_locations, [search_op])

    # 1) r1 searches roomA for objA
    actions = sim.get_actions()
    a1 = get_action_by_name(actions, "search r1 start roomA objA")
    sim.advance(a1)
    assert F("free r1") not in sim.state.fluents
    assert F("free r2") in sim.state.fluents
    assert pytest.approx(sim.state.time) == 0.0

    # 2) r2 searches roomB for objB
    actions = sim.get_actions()
    a2 = get_action_by_name(actions, "search r2 start roomB objB")
    sim.advance(a2)
    assert F("free r1") in sim.state.fluents
    assert F("free r2") not in sim.state.fluents
    assert F("revealed roomA") in sim.state.fluents
    assert F("found objA") in sim.state.fluents
    assert F("found objC") in sim.state.fluents  # revealed with roomA
    assert pytest.approx(sim.state.time) == 3.0

    # 3) r1 searches roomC for objB (moving from roomA to roomC)
    actions = sim.get_actions()
    a3 = get_action_by_name(actions, "search r1 roomA roomC objB")
    sim.advance(a3)
    print(sim.state)
    assert pytest.approx(sim.state.time) == 5.0
    assert F("free r1") not in sim.state.fluents
    assert F("free r2") in sim.state.fluents
    assert F("revealed roomA") in sim.state.fluents
    assert F("revealed roomB") in sim.state.fluents
    assert F("found objA") in sim.state.fluents
    assert F("found objC") in sim.state.fluents

import pytest
import numpy as np
from dataclasses import dataclass
from copy import copy

from mrppddl.core import Fluent, State, get_action_by_name
from mrppddl.core import transition

from mrppddl.core import OptCallable, Operator, Effect
from mrppddl.helper import _make_callable, _invert_prob

from typing import Dict, Set, List, Tuple, Callable
import itertools
from .actions import OngoingMoveAction, OngoingSearchAction, OngoingPickAction, OngoingPlaceAction

F = Fluent

class Simulator:
    """Tiny wrapper around the PDDL core that (i) applies an action via
    `transition` and (ii) deterministically 'reveals' searched locations by
    intersecting outcomes across probabilistic branches.
    """

    def __init__(
        self,
        initial_state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator],
        environment
    ):
        self._state = initial_state
        self.objects_by_type = {k: set(v) for k, v in objects_by_type.items()}
        self.operators = operators
        self.ongoing_actions = []
        self.environment = environment
        self.storage = dict()
        self.trajectory = dict()

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

    def get_actions(self) -> List:
        """Instantiate an Operator under the *current* objects_by_type."""
        objects_with_rloc = {k: set(v) for k, v in self.objects_by_type.items()}
        objects_with_rloc["location"] |= set(
            f"{rob}_loc"
            for rob in self.objects_by_type["robot"]
            if F(f"at {rob} {rob}_loc") in self._state.fluents
            )
        all_actions = list(itertools.chain.from_iterable(
            operator.instantiate(objects_with_rloc)
            for operator in self.operators
        ))

        def filter_fn(action):
            ans = action.name.split()
            if ans[0] == 'move' and '_loc' in ans[3]:
                # (move robot1 robot1_loc other_loc)
                return False
            if ans[0] == 'place' and '_loc' in ans[2]:
                # (place robot1 robot1_loc Pillow)
                return False
            if ans[0] == 'search' and '_loc' in ans[2]:
                # (search robot1 robot1_loc Pillow)
                return False

            return True

        # Filter out actions that should not be allowed.
        filtered_actions = [a for a in all_actions if filter_fn(a)]
        return filtered_actions

    def advance(self, action, do_interrupt=True) -> State:
        if not self.state.satisfies_precondition(action):
            raise ValueError(f"Action preconditions not satisfied: {action.name} in state {self.state}")

        """Add a new action and then advance as much as possible using both `transition` and
        also `_reveal` as needed once areas are searched."""
        action_name = action.name.split()[0]
        if action_name not in {"move", "pick", "place", "search"}:
            raise ValueError(f"Action {action.name} not supported in simulator.")
        if action_name == "move":
            new_act = OngoingMoveAction(self._state.time, action, self.environment)
        elif action_name == "search":
            new_act = OngoingSearchAction(self._state.time, action, self.environment)
        elif action_name == "pick":
            new_act = OngoingPickAction(self._state.time, action, self.environment)
        elif action_name == "place":
            new_act = OngoingPlaceAction(self._state.time, action, self.environment)

        self.ongoing_actions.append(new_act)

        def _any_free_robots(state):
            return any(f.name == "free" for f in state.fluents)

        # Loop at least once to advance "time == now" actions
        robot_free = False
        while not robot_free and self.ongoing_actions:
            # If there are no actions that are not done, break.
            if not any(not act.is_done for act in self.ongoing_actions):
                break

            # Get the time we need to advance. If a robot needs an action the time is "now"
            if _any_free_robots(self._state):
                adv_time = self.time
            else:
                adv_time = min(act.time_to_next_event for act in self.ongoing_actions)

            # Get new active effects to add to the state, then transition and _reveal.
            new_effects = list(itertools.chain.from_iterable(
                [act.advance(adv_time) for act in self.ongoing_actions]))
            new_state = State(adv_time,
                              self._state.fluents,
                              sorted(self._state.upcoming_effects + new_effects,
                                     key=lambda el: el[0]))
            self._state, self.objects_by_type = self._reveal(transition(new_state, None))


            # Remove any actions that are now done
            self.ongoing_actions = [act for act in self.ongoing_actions if not act.is_done]

            robot_free = _any_free_robots(self._state)

            # Try to store the location of the robots (debug)
            try:
                self.storage[self.time] = {
                    "locations": copy(self.environment.locations),
                    "fluents": copy(self.state.fluents)
                }
            except:
                pass

        # Interrupt actions as needed
        # - if any action can be interrupted, interrupt it, getting the
        # Example: for the move action, interrupting it
        for act in self.ongoing_actions:
            new_fluents = act.interrupt()
            self._state.update_fluents(new_fluents)

        # Remove any actions that are now done
        self.ongoing_actions = [act for act in self.ongoing_actions if not act.is_done]

        # Return the resulting state (for planning)
        return self.state


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
            for obj_type, objs in self.environment.get_objects_at_location(loc).items():
                new_objects_by_type.setdefault(obj_type, set()).update(objs)
                for obj in objs:
                    new_fluents.add(F(f"found {obj}"))
                    new_fluents.add(F(f"at {obj} {loc}"))

        new_state = State(states[0].time, new_fluents, new_upcoming)
        return new_state, new_objects_by_type

    def is_goal_reached(self, goal_fluents: Set[Fluent]) -> bool:
        return all(f in self.state.fluents for f in goal_fluents)

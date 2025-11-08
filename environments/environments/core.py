import itertools
from copy import copy

from mrppddl.core import Fluent as F, State
from mrppddl.core import transition
from mrppddl.core import Operator
from environments import BaseEnvironment
from typing import Dict, Set, List


IDLE = -1
MOVING = 0
REACHED = 1


class EnvironmentInterface():
    def __init__(
            self,
            initial_state: State,
            objects_by_type: Dict[str, Set[str]],
            operators: List[Operator],
            environment: BaseEnvironment):
        self._state = initial_state
        self.objects_by_type = {k: set(v) for k, v in objects_by_type.items()}
        self.operators = operators
        self.ongoing_actions = []
        self.environment = environment

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
        objects_with_rloc = {k: set(v)
                             for k, v in self.objects_by_type.items()}
        objects_with_rloc["location"] |= set(
            f"{rob}_loc"
            for rob in self.objects_by_type["robot"]
            if F(f"at {rob} {rob}_loc") in self._state.fluents
        )
        return list(itertools.chain.from_iterable(
            operator.instantiate(objects_with_rloc)
            for operator in self.operators
        ))

    def _any_free_robots(self):
        return any(f.name == "free" for f in self._state.fluents)

    def advance(self, action):
        # Check preconditions for safety (helps in debugging)
        if not self.state.satisfies_precondition(action):
            raise ValueError(f"Action preconditions not satisfied: {action.name} in state {self.state}")

        new_act = self._get_ongoing_action(action)
        self.ongoing_actions.append(new_act)

        robot_free = False
        while not robot_free and self.ongoing_actions:
            # if every action is done, break
            if all(act.is_done for act in self.ongoing_actions):
                break

            adv_time = self._get_advance_time()

            new_effects = list(itertools.chain.from_iterable(
                [act.advance(adv_time) for act in self.ongoing_actions])
            )
            new_state = State(adv_time,
                              self._state.fluents,
                              sorted(self._state.upcoming_effects + new_effects,
                                     key=lambda el: el[0])
                              )
            # Add new effects to state
            self._state = self._get_new_state_by_intersection(transition(new_state, None))

            # Reveal and get new fluents and objects_by_type
            new_fluents, self.objects_by_type = self._reveal()
            self._state.update_fluents(new_fluents)

            # Remove any actions that are now done
            self.ongoing_actions = [act for act in self.ongoing_actions if not act.is_done]

            # Update environment time (used only in simulators to measure robot's action progress)
            self.environment.time = self._state.time

            robot_free = self._any_free_robots()

        # interrupt ongoing actions if needed
        for act in self.ongoing_actions:
            new_fluents = act.interrupt()
            self._state.update_fluents(new_fluents)

        self.ongoing_actions = [act for act in self.ongoing_actions if act.upcoming_effects]
        return self.state

    def _get_new_state_by_intersection(self, states_and_probs):
        '''Determine new state by intersecting outcomes'''
        states = [s for s, _ in states_and_probs]
        base = states[0]
        new_fluents = {fl for fl in base.fluents if all(fl in s.fluents for s in states[1:])}
        new_upcoming = [ue for ue in base.upcoming_effects if all(ue in s.upcoming_effects for s in states[1:])]

        new_state = State(states[0].time, new_fluents, new_upcoming)
        return new_state

    def _get_ongoing_action(self, action):
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
        return new_act

    def _get_advance_time(self):
        for act in self.ongoing_actions:
            if act.is_action_complete:
                return act.time_to_next_event
        return self.time

    def goal_reached(self, goal_fluents):
        if all(fluent in self.state.fluents for fluent in goal_fluents):
            return True
        return False

    def _reveal(self):
        new_fluents = {fl for fl in self._state.fluents}

        def _is_location_searched(location: str, fluents: Set[F]) -> bool:
            # Fluent 'searched' looks like: F("searched <loc> <obj>")
            return any(f.name == "searched" and f.args[0] == location for f in fluents)

        # Locations that have been searched but not yet marked revealed
        newly_revealed_locations = [
            loc
            for loc in self.objects_by_type.get("location", set())
            if _is_location_searched(loc, new_fluents) and F(f"revealed {loc}") not in new_fluents
        ]

        updated_objects_by_type = copy(self.objects_by_type)
        for loc in newly_revealed_locations:
            new_fluents.add(F(f"revealed {loc}"))

            # Add every object present at this location to type universe and mark as found/located
            for obj_type, objs in self.environment.get_objects_at_location(loc).items():
                updated_objects_by_type.setdefault(obj_type, set()).update(objs)
                for obj in objs:
                    new_fluents.add(F(f"found {obj}"))
                    new_fluents.add(F(f"at {obj} {loc}"))

        return new_fluents, updated_objects_by_type


class OngoingAction:
    def __init__(self, time, action, environment=None):
        self.time = time
        self.name = action.name
        self._start_time = time
        self._action = action
        self._upcoming_effects = sorted([
            (time + eff.time, eff) for eff in action.effects
        ], key=lambda el: el[0])
        self.environment = environment
        self.is_action_called = False
        self.robot = self.name.split()[1]  # (e.g., move r1 locA locB)

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
    def is_action_complete(self):
        if self.is_action_called:
            action_name = self.name.split()[0]
            action_status = self.environment.get_action_status(self.robot, action_name)
            if action_status == REACHED:
                return True
        return False

    @property
    def upcoming_effects(self):
        # Return remaining upcoming events
        return self._upcoming_effects

    def advance(self, time):
        # Update the internal time
        self.time = time

        # Pop and return all effects scheduled at or before the new time
        new_effects = [effect for effect in self._upcoming_effects
                       if effect[0] <= time + 1e-9]
        # Remove the new_effects from upcoming_effects (effects are sorted)
        self._upcoming_effects = self._upcoming_effects[len(new_effects):]

        return new_effects

    def interrupt(self):
        """Cannot interrupt this action. Nothing happens."""
        return set()

    def __str__(self):
        return f"OngoingAction<{self.name}, {self.time}, {self.upcoming_effects}>"


class OngoingSearchAction(OngoingAction):
    def advance(self, time):
        if not self.is_action_called:
            self.environment.search_robot(self.robot)
            self.is_action_called = True
        return super().advance(time)


class OngoingPickAction(OngoingAction):
    def advance(self, time):
        if not self.is_action_called:
            self.environment.pick_robot(self.robot)
            self.is_action_called = True

        new_effects = super().advance(time)
        if self.is_done:
            _, _, loc, obj = self.name.split()  # (e.g., pick r1 locA objA)
            # remove the object from the location
            self.environment.remove_object_from_location(obj, loc)
        return new_effects


class OngoingPlaceAction(OngoingAction):
    def advance(self, time):
        if not self.is_action_called:
            self.environment.place_robot(self.robot)
            self.is_action_called = True

        new_effects = super().advance(time)
        if self.is_done:
            _, _, loc, obj = self.name.split()  # (e.g., place r1 locA objA)
            # add the object to the location
            self.environment.add_object_at_location(obj, loc)
        return new_effects


class OngoingMoveAction(OngoingAction):
    def __init__(self, time, action, environment=None):
        super().__init__(time, action, environment)
        # Keep track of initial start and end locations
        _, self.robot, self.start, self.end = self.name.split()  # (e.g., move r1 locA locB)

    def advance(self, time):
        if not self.is_action_called:
            self.environment.move_robot(self.robot, self.end)
            self.is_action_called = True
        return super().advance(time)

    def interrupt(self):
        if self.time <= self._start_time:
            return set()  # Cannot interrupt before start time

        # stop robot
        self.environment.stop_robot(self.robot)
        robot = self.robot
        old_target = self.end
        new_target = f"{robot}_loc"
        new_fluents = set()

        for _, eff in self._upcoming_effects:
            if eff.is_probabilistic:
                raise ValueError("Probabilistic effects cannot be interrupted.")
            for fluent in eff.resulting_fluents:
                new_fluents.add(
                    F(" ".join(
                        [fluent.name]
                      + [fa if fa != old_target else new_target for fa in fluent.args]),
                      negated=fluent.negated)
                )

        self._upcoming_effects = []
        return new_fluents

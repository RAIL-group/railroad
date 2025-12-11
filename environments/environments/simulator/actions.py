from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl.helper import _make_callable, _invert_prob
from mrppddl.core import OptCallable, Operator, Effect


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


class OngoingMoveAction(OngoingAction):
    def __init__(self, time, action, environment=None):
        super().__init__(time, action, environment)
        # Keep track of initial start and end locations
        _, self.robot, self.start, self.end = self.name.split()  # (e.g., move r1 locA locB)

    def advance(self, time):
        new_effects = super().advance(time)
        delta_time = self.time - self._start_time
        intermediate_coords = self.environment.get_intermediate_coordinates(delta_time, self.start, self.end)
        self.environment.locations[f"{self.robot}_loc"] = intermediate_coords
        return new_effects

    def interrupt(self):
        """If the time > start_time, it can be interrupted. The robot location
        is updated, this action is marked as done, and the new fluents are
        returned."""

        if self.time <= self._start_time:
            return set() # Cannot interrupt before start time

        # This action is done. Treat this as having "reached" the destination
        # but where the destination is robot_loc, which means we must replace
        # all the old "target location" with "robot_loc". While this may seem
        # like a fair bit of needless complexity, it means that we don't need to
        # have a custom function for each new move action: all it's
        # post-conditions are added automatically.

        robot = self.robot
        old_target = self.end
        new_target = f"{robot}_loc"
        new_fluents = set()

        for _, eff in self._upcoming_effects:
            if eff.is_probabilistic:
                raise ValueError("Probabilistic effects cannot be interrupted.")
            for fluent in eff.resulting_fluents:
                if (~fluent) in new_fluents:
                    new_fluents.remove(~fluent)
                new_fluents.add(
                    F(" ".join(
                        [fluent.name]
                      + [fa if fa != old_target else new_target for fa in fluent.args]),
                      negated=fluent.negated)
                )
        self._upcoming_effects = []
        return new_fluents


class OngoingSearchAction(OngoingAction):
    pass


class OngoingPickAction(OngoingAction):
    def advance(self, time):
        new_effects = super().advance(time)
        if self.is_done:
            _, _, loc, obj = self.name.split()  # (e.g., pick r1 locA objA)
            # remove the object from the location
            self.environment.remove_object_from_location(obj, loc)
        return new_effects


class OngoingPlaceAction(OngoingAction):
    def advance(self, time):
        new_effects = super().advance(time)
        if self.is_done:
            _, _, loc, obj = self.name.split()  # (e.g., place r1 locA objA)
            # add the object to the location
            self.environment.add_object_at_location(obj, loc)
        return new_effects


def construct_move_operator(move_time: OptCallable):
    move_time = _make_callable(move_time)
    move_time_plus_eps = lambda *args: move_time(*args) + 0.1
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r"), ~F("just-moved ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?r ?from")}),
            Effect(
                time=(move_time, ["?r", "?from", "?to"]),
                resulting_fluents={F("free ?r"), F("at ?r ?to"), F("just-moved ?r")},
            ),
            Effect(
                time=(move_time_plus_eps, ["?r", "?from", "?to"]),
                resulting_fluents={~F("just-moved ?r")},
            ),
        ],
    )


def construct_search_operator(object_find_prob: OptCallable, search_time: OptCallable) -> Operator:
    object_find_prob = _make_callable(object_find_prob)
    inv_object_find_prob = _invert_prob(object_find_prob)
    search_time = _make_callable(search_time)
    return Operator(
        name="search",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("not revealed ?loc"), F("not searched ?loc ?obj"), F("not found ?obj"), F("not lock-search ?loc")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("lock-search ?loc")}),
            Effect(time=(search_time, ["?r", "?loc"]),
                    resulting_fluents={F("free ?r"),
                                      F("searched ?loc ?obj"),
                                      F("not lock-search ?loc")
                                      },
                   prob_effects=[((object_find_prob, ["?r", "?loc", "?obj"]),
                                  [Effect(time=0, resulting_fluents={F("found ?obj"), F("at ?obj ?loc")})]),
                                 ((inv_object_find_prob, ["?r", "?loc", "?obj"]),
                                  [])]
                   )
        ]
    )

def construct_pick_operator(pick_time: OptCallable) -> Operator:
    pick_time = _make_callable(pick_time)
    pick_time_plus_eps = lambda *args: pick_time(*args) + 0.1
    return Operator(
        name="pick",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("at ?obj ?loc"), ~F("hand-full ?r"), ~F("just-placed ?r ?obj")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?obj ?loc")}),
            Effect(time=(pick_time, ["?r", "?loc", "?obj"]),
                   resulting_fluents={F("free ?r"), F("holding ?r ?obj"), F("hand-full ?r"), F("just-picked ?r ?obj")}),
            Effect(time=(pick_time_plus_eps, ["?r", "?loc", "?obj"]),
                   resulting_fluents={~F("just-picked ?r ?obj")}),
        ],
    )

def construct_place_operator(place_time: OptCallable) -> Operator:
    place_time = _make_callable(place_time)
    place_time_plus_eps = lambda *args: place_time(*args) + 0.1
    return Operator(
        name="place",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("holding ?r ?obj"), F("hand-full ?r"), ~F("just-picked ?r ?obj")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not holding ?r ?obj")}),
            Effect(time=(place_time, ["?r", "?loc", "?obj"]),
                   resulting_fluents={F("free ?r"), F("at ?obj ?loc"), ~F("hand-full ?r"), F("just-placed ?r ?obj")}),
            Effect(time=(place_time_plus_eps, ["?r", "?loc", "?obj"]),
                   resulting_fluents={~F("just-placed ?r ?obj")}),
        ],
    )

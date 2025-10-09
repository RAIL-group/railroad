from mrppddl.core import Fluent, State, get_action_by_name
from mrppddl.helper import _make_callable, _invert_prob
from mrppddl.core import OptCallable, Operator, Effect


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

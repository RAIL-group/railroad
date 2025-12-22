from mrppddl.core import Fluent as F
from mrppddl.helper import _make_callable, _invert_prob
from mrppddl.core import OptCallable, Operator, Effect


def construct_move_operator_nonblocking(move_time: OptCallable):
    move_time = _make_callable(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?r ?from")}),
            Effect(
                time=(move_time, ["?r", "?from", "?to"]),
                resulting_fluents={F("free ?r"), F("at ?r ?to")},
            ),
        ],
    )


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

def construct_pick_operator_nonblocking(pick_time: OptCallable) -> Operator:
    pick_time = _make_callable(pick_time)
    return Operator(
        name="pick",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("at ?obj ?loc"), ~F("hand-full ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?obj ?loc")}),
            Effect(time=(pick_time, ["?r", "?loc", "?obj"]),
                   resulting_fluents={F("free ?r"), F("holding ?r ?obj"), F("hand-full ?r")}),
        ],
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

def construct_place_operator_nonblocking(place_time: OptCallable) -> Operator:
    place_time = _make_callable(place_time)
    return Operator(
        name="place",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("holding ?r ?obj"), F("hand-full ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not holding ?r ?obj")}),
            Effect(time=(place_time, ["?r", "?loc", "?obj"]),
                   resulting_fluents={F("free ?r"), F("at ?obj ?loc"), ~F("hand-full ?r")}),
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


def construct_no_op_operator(no_op_time, extra_cost: float = 0.0) -> Operator:
    '''Sometimes, patience is a virtuous skill.'''
    return Operator(
        name="no_op",
        parameters=[("?r", "robot")],
        preconditions=[F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(time=no_op_time, resulting_fluents={F("free ?r")}),
        ],
        extra_cost=extra_cost,
    )


# REAL WORLD OPERATORS
def construct_move_visited_operator(move_time: OptCallable):
    move_time = _make_callable(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r"), F("not visited ?to")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(
                time=(move_time, ["?r", "?from", "?to"]),
                resulting_fluents={
                    F("free ?r"),
                    F("not at ?r ?from"),
                    F("at ?r ?to"),
                    F("visited ?to"),
                },
            ),
        ],
    )

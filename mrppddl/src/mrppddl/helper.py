from .core import Operator, Fluent, Effect, OptCallable, Num
from typing import Callable

F = Fluent


def _make_callable(opt_expr: OptCallable) -> Callable[..., float]:
    if isinstance(opt_expr, Num):
        return lambda *args: opt_expr
    else:
        return lambda *args: opt_expr(*args)


def _invert_prob(opt_expr: OptCallable) -> Callable[..., float]:
    if isinstance(opt_expr, Num):
        return lambda *args: 1 - opt_expr
    else:
        return lambda *args: 1 - opt_expr(*args)


def construct_move_operator(move_time: OptCallable):
    move_time = _make_callable(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={Fluent("not free ?r")}),
            Effect(
                time=(move_time, ["?r", "?from", "?to"]),
                resulting_fluents={F("free ?r"), F("not at ?r ?from"), F("at ?r ?to")},
            ),
        ],
    )

def construct_move_visited_operator(move_time: OptCallable):
    move_time = _make_callable(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={Fluent("not free ?r")}),
            Effect(
                time=(move_time, ["?r", "?from", "?to"]),
                resulting_fluents={F("free ?r"), F("not at ?r ?from"), F("at ?r ?to"), F("visited ?to")},
            ),
        ],
    )

def construct_search_operator(
    object_find_prob: OptCallable, move_time: OptCallable, pick_time: OptCallable
) -> Operator:
    object_find_prob = _make_callable(object_find_prob)
    inv_object_find_prob = _invert_prob(object_find_prob)
    move_time = _make_callable(move_time)
    pick_time = _make_callable(pick_time)
    return Operator(
        name="search",
        parameters=[
            ("?robot", "robot"),
            ("?loc_from", "location"),
            ("?loc_to", "location"),
            ("?object", "object"),
        ],
        preconditions=[
            Fluent("free ?robot"),
            Fluent("at ?robot ?loc_from"),
            ~Fluent("searched ?loc_to ?object"),
            ~Fluent("found ?object"),
        ],
        effects=[
            Effect(
                time=0,
                resulting_fluents={
                    ~Fluent("free ?robot"),
                    ~Fluent("at ?robot ?loc_from"),
                    Fluent("searched ?loc_to ?object"),
                },
            ),
            Effect(
                time=(move_time, ["?robot", "?loc_from"]),
                resulting_fluents={Fluent("at ?robot ?loc_to")},
                prob_effects=[
                    (
                        (object_find_prob, ["?robot", "?loc_to", "?object"]),
                        [
                            Effect(time=0, resulting_fluents={Fluent("found ?object")}),
                            Effect(
                                time=(pick_time, ["?robot", "?object"]),
                                resulting_fluents={
                                    Fluent("holding ?robot ?object"),
                                    Fluent("free ?robot"),
                                },
                            ),
                        ],
                    ),
                    (
                        (inv_object_find_prob, ["?robot", "?loc_to", "?object"]),
                        [Effect(time=0, resulting_fluents={Fluent("free ?robot")})],
                    ),
                ],
            ),
        ],
    )

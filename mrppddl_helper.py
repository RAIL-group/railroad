from .mrppddl import Operator, Fluent, Effect, ProbEffect
from typing import Callable, Union

OptCallable = Union[float, Callable]

def _invert_prob(prob: OptCallable) -> OptCallable:
    if isinstance(prob, Callable):
        return lambda *args: 1 - prob(*args)
    else:
        return 1 - prob

def _opt_to_callable(optional_callable: OptCallable) -> Callable:
    if isinstance(optional_callable, Callable):
        return optional_callable

    return lambda *args: optional_callable

def specify_args(optional_callable: OptCallable, *args: str) -> OptCallable:
    if isinstance(optional_callable, Callable):
        fn = optional_callable
        return lambda b: fn(*[b.get(arg, arg) for arg in args])
    else:
        return optional_callable


def construct_move_operator(move_time: OptCallable):
    return Operator(
        name="move",
        parameters=[("?robot", "robot"),
                    ("?loc_from", "location"),
                    ("?loc_to", "location")],
        preconditions=[
            Fluent("at", "?robot", "?loc_from"),
            Fluent("free", "?robot")],
        effects=[
            Effect(time=0,
                   resulting_fluents={~Fluent("free", "?robot")}),
            Effect(time=specify_args(move_time, "?robot", "?loc_from", "?loc_to"),
                   resulting_fluents={
                       Fluent("free", "?robot"),
                       ~Fluent("at", "?robot", "?loc_from"),
                       Fluent("at", "?robot", "?loc_to")})
        ])


def construct_search_operator(object_find_prob: OptCallable, move_time: OptCallable, pick_time: OptCallable) -> Operator:
    return Operator(
        name="search",
        parameters=[
            ("?robot", "robot"),
            ("?loc_from", "location"),
            ("?loc_to", "location"),
            ("?object", "object")
        ],
        preconditions=[
            Fluent("free ?robot"),
            Fluent("at ?robot ?loc_from"),
            ~Fluent("searched ?loc_to ?object"),
            ~Fluent("found ?object")
        ],
        effects=[
            Effect(
                time=0,
                resulting_fluents={~Fluent("free ?robot"), ~Fluent("at ?robot ?loc_from"), Fluent("searched ?loc_to ?object")}
            ),
            ProbEffect(
                time=specify_args(move_time, "?robot ?loc_from"),
                resulting_fluents={Fluent("at ?robot ?loc_to")},
                prob_effects=[
                    (specify_args(object_find_prob, "?robot", "?loc_to", "?object"),
                     [Effect(time=0, resulting_fluents={Fluent("found ?object")}),
                      Effect(time=specify_args(pick_time, "?robot", "?object"), 
                             resulting_fluents={Fluent("holding ?robot ?object"), Fluent("free ?robot")})]),
                    (specify_args(_invert_prob(object_find_prob), "?robot", "?loc_to", "?object"),
                     [Effect(time=0, resulting_fluents={Fluent("free ?robot")})]),
                ],
            )
        ],
    )

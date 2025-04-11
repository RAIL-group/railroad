from .mrppddl import Operator, Fluent, Effect, ProbEffect, OptExpr
from typing import Callable, Union

# OptCallable = Union[float, Callable]
F = Fluent

def _invert_prob(expr: OptExpr) -> OptExpr:
    if isinstance(expr, Union[float, int]):
        return 1 - expr
    else:
        return (lambda *args: 1 - expr[0](*args), expr[1])

# def _opt_to_callable(optional_callable: OptCallable) -> Callable:
#     if isinstance(optional_callable, Callable):
#         return optional_callable

#     return lambda *args: optional_callable

# def specify_args(optional_callable: OptCallable, *args: str) -> OptCallable:
#     if isinstance(optional_callable, Callable):
#         fn = optional_callable
#         return lambda b: fn(*[b.get(arg, arg) for arg in args])
#     else:
#         return optional_callable


def construct_move_operator(move_time: Callable[[str, str, str], float]):
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?robot ?from"), F("free ?robot")],
        effects=[
            Effect(time=0, resulting_fluents={Fluent("not free ?robot")}),
            Effect(time=(move_time, ("?robot", "?from", "?to")),
                   resulting_fluents={F("free ?robot"), 
                                      F("not at ?robot ?from"),
                                      F("at ?robot ?to")})
        ])


def construct_search_operator(object_find_prob: OptExpr, move_time: OptExpr, pick_time: OptExpr) -> Operator:
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
                      Effect(time=(pick_time, "?robot", "?object"), 
                             resulting_fluents={Fluent("holding ?robot ?object"), Fluent("free ?robot")})]),
                    (specify_args(_invert_prob(object_find_prob), "?robot", "?loc_to", "?object"),
                     [Effect(time=0, resulting_fluents={Fluent("free ?robot")})]),
                ],
            )
        ],
    )

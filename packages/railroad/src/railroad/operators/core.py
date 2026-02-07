"""Core operator constructors for PDDL planning.

This module provides operator constructors for common robot planning actions:
- Movement operators (with and without blocking preconditions)
- Search operators (search-only and search-and-pick variants)
- Pick and place operators (with and without blocking preconditions)
- Wait and no-op operators for multi-robot coordination

Naming convention:
- Default operators (without suffix) are non-blocking
- `_blocking` suffix indicates operators with just-moved/just-picked/just-placed preconditions
- `_constrained` suffix indicates operators with additional constraints (e.g., "not visited")
"""

from railroad.core import Fluent, Operator, Effect

from ._utils import OptNumeric, _to_numeric

F = Fluent


# =============================================================================
# Move Operators
# =============================================================================


def construct_move_operator(move_time: OptNumeric) -> Operator:
    """Construct a basic move operator (non-blocking).

    Args:
        move_time: Time or function to compute movement duration.
            Function signature: (robot, from_location, to_location) -> float

    Returns:
        Operator for moving a robot between locations.
    """
    move_time_fn = _to_numeric(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?r ?from")}),
            Effect(
                time=(move_time_fn, ["?r", "?from", "?to"]),
                resulting_fluents={F("free ?r"), F("at ?r ?to")},
            ),
        ],
    )


def construct_move_operator_blocking(move_time: OptNumeric) -> Operator:
    """Construct a move operator with just-moved blocking precondition.

    This prevents immediate consecutive moves by the same robot.

    Args:
        move_time: Time or function to compute movement duration.
            Function signature: (robot, from_location, to_location) -> float

    Returns:
        Operator for moving with blocking precondition.
    """
    move_time_fn = _to_numeric(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r"), ~F("just-moved ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?r ?from")}),
            Effect(
                time=(move_time_fn, ["?r", "?from", "?to"]),
                resulting_fluents={F("free ?r"), F("at ?r ?to"), F("just-moved ?r")},
            ),
            Effect(
                time=(move_time_fn + 0.1, ["?r", "?from", "?to"]),
                resulting_fluents={~F("just-moved ?r")},
            ),
        ],
    )


def construct_move_visited_operator(move_time: OptNumeric) -> Operator:
    """Construct a move operator that tracks visited locations.

    Does not have a "not visited" precondition, allowing revisits.

    Args:
        move_time: Time or function to compute movement duration.
            Function signature: (robot, from_location, to_location) -> float

    Returns:
        Operator that marks destinations as visited upon arrival.
    """
    move_time_fn = _to_numeric(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(
                time=(move_time_fn, ["?r", "?from", "?to"]),
                resulting_fluents={
                    F("free ?r"),
                    F("not at ?r ?from"),
                    F("at ?r ?to"),
                    F("visited ?to"),
                },
            ),
        ],
    )


def construct_move_visited_operator_constrained(move_time: OptNumeric) -> Operator:
    """Construct a move operator that only allows visiting unvisited locations.

    Has a "not visited ?to" precondition, preventing revisits.

    Args:
        move_time: Time or function to compute movement duration.
            Function signature: (robot, from_location, to_location) -> float

    Returns:
        Operator that only allows moves to unvisited locations.
    """
    move_time_fn = _to_numeric(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r"), F("not visited ?to")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(
                time=(move_time_fn, ["?r", "?from", "?to"]),
                resulting_fluents={
                    F("free ?r"),
                    F("not at ?r ?from"),
                    F("at ?r ?to"),
                    F("visited ?to"),
                },
            ),
        ],
    )


# =============================================================================
# Search Operators
# =============================================================================


def construct_search_operator(
    object_find_prob: OptNumeric, search_time: OptNumeric
) -> Operator:
    """Construct a search-only operator.

    Searches a location for an object without moving or picking.

    Args:
        object_find_prob: Probability or function for finding the object.
            Function signature: (robot, location, object) -> float
        search_time: Time or function for search duration.
            Function signature: (robot, location, object) -> float

    Returns:
        Operator for searching a location.
    """
    object_find_prob_fn = _to_numeric(object_find_prob)
    search_time_fn = _to_numeric(search_time)
    return Operator(
        name="search",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[
            F("at ?r ?loc"),
            F("free ?r"),
            F("not revealed ?loc"),
            F("not searched ?loc ?obj"),
            F("not found ?obj"),
            F("not lock-search ?loc"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("lock-search ?loc")}),
            Effect(
                time=(search_time_fn, ["?r", "?loc", "?obj"]),
                resulting_fluents={
                    F("free ?r"),
                    F("searched ?loc ?obj"),
                    F("not lock-search ?loc"),
                },
                prob_effects=[
                    (
                        (object_find_prob_fn, ["?r", "?loc", "?obj"]),
                        [Effect(time=0, resulting_fluents={F("found ?obj"), F("at ?obj ?loc")})],
                    ),
                    (
                        (1 - object_find_prob_fn, ["?r", "?loc", "?obj"]),
                        [],
                    ),
                ],
            ),
        ],
    )


def construct_search_and_pick_operator(
    object_find_prob: OptNumeric, move_time: OptNumeric, pick_time: OptNumeric
) -> Operator:
    """Construct a combined search-move-pick operator.

    Moves to a location, searches for an object, and picks it up if found.

    Args:
        object_find_prob: Probability or function for finding the object.
            Function signature: (robot, location, object) -> float
        move_time: Time or function for movement duration.
            Function signature: (robot, from_location, to_location) -> float
        pick_time: Time or function for pick duration.
            Function signature: (robot, object) -> float

    Returns:
        Operator combining search, move, and pick actions.
    """
    object_find_prob_fn = _to_numeric(object_find_prob)
    move_time_fn = _to_numeric(move_time)
    pick_time_fn = _to_numeric(pick_time)
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
                time=(move_time_fn, ["?robot", "?loc_from", "?loc_to"]),
                resulting_fluents={Fluent("at ?robot ?loc_to")},
                prob_effects=[
                    (
                        (object_find_prob_fn, ["?robot", "?loc_to", "?object"]),
                        [
                            Effect(time=0, resulting_fluents={Fluent("found ?object")}),
                            Effect(
                                time=(pick_time_fn, ["?robot", "?object"]),
                                resulting_fluents={
                                    Fluent("holding ?robot ?object"),
                                    Fluent("free ?robot"),
                                },
                            ),
                        ],
                    ),
                    (
                        (1 - object_find_prob_fn, ["?robot", "?loc_to", "?object"]),
                        [Effect(time=0, resulting_fluents={Fluent("free ?robot")})],
                    ),
                ],
            ),
        ],
    )


# =============================================================================
# Pick Operators
# =============================================================================


def construct_pick_operator(pick_time: OptNumeric) -> Operator:
    """Construct a basic pick operator (non-blocking).

    Args:
        pick_time: Time or function for pick duration.
            Function signature: (robot, location, object) -> float

    Returns:
        Operator for picking up an object.
    """
    pick_time_fn = _to_numeric(pick_time)
    return Operator(
        name="pick",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("at ?obj ?loc"), ~F("hand-full ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?obj ?loc")}),
            Effect(
                time=(pick_time_fn, ["?r", "?loc", "?obj"]),
                resulting_fluents={F("free ?r"), F("holding ?r ?obj"), F("hand-full ?r")},
            ),
        ],
    )


def construct_pick_operator_blocking(pick_time: OptNumeric) -> Operator:
    """Construct a pick operator with just-placed blocking precondition.

    Prevents immediately picking up an object that was just placed.

    Args:
        pick_time: Time or function for pick duration.
            Function signature: (robot, location, object) -> float

    Returns:
        Operator for picking with blocking precondition.
    """
    pick_time_fn = _to_numeric(pick_time)
    return Operator(
        name="pick",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[
            F("at ?r ?loc"),
            F("free ?r"),
            F("at ?obj ?loc"),
            ~F("hand-full ?r"),
            ~F("just-placed ?r ?obj"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?obj ?loc")}),
            Effect(
                time=(pick_time_fn, ["?r", "?loc", "?obj"]),
                resulting_fluents={
                    F("free ?r"),
                    F("holding ?r ?obj"),
                    F("hand-full ?r"),
                    F("just-picked ?r ?obj"),
                },
            ),
            Effect(
                time=(pick_time_fn + 0.1, ["?r", "?loc", "?obj"]),
                resulting_fluents={~F("just-picked ?r ?obj")},
            ),
        ],
    )


# =============================================================================
# Place Operators
# =============================================================================


def construct_place_operator(place_time: OptNumeric) -> Operator:
    """Construct a basic place operator (non-blocking).

    Args:
        place_time: Time or function for place duration.
            Function signature: (robot, location, object) -> float

    Returns:
        Operator for placing an object.
    """
    place_time_fn = _to_numeric(place_time)
    return Operator(
        name="place",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("holding ?r ?obj"), F("hand-full ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not holding ?r ?obj")}),
            Effect(
                time=(place_time_fn, ["?r", "?loc", "?obj"]),
                resulting_fluents={F("free ?r"), F("at ?obj ?loc"), ~F("hand-full ?r")},
            ),
        ],
    )


def construct_place_operator_blocking(place_time: OptNumeric) -> Operator:
    """Construct a place operator with just-picked blocking precondition.

    Prevents immediately placing an object that was just picked up.

    Args:
        place_time: Time or function for place duration.
            Function signature: (robot, location, object) -> float

    Returns:
        Operator for placing with blocking precondition.
    """
    place_time_fn = _to_numeric(place_time)
    return Operator(
        name="place",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[
            F("at ?r ?loc"),
            F("free ?r"),
            F("holding ?r ?obj"),
            F("hand-full ?r"),
            ~F("just-picked ?r ?obj"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not holding ?r ?obj")}),
            Effect(
                time=(place_time_fn, ["?r", "?loc", "?obj"]),
                resulting_fluents={
                    F("free ?r"),
                    F("at ?obj ?loc"),
                    ~F("hand-full ?r"),
                    F("just-placed ?r ?obj"),
                },
            ),
            Effect(
                time=(place_time_fn + 0.1, ["?r", "?loc", "?obj"]),
                resulting_fluents={~F("just-placed ?r ?obj")},
            ),
        ],
    )


# =============================================================================
# Wait and No-Op Operators
# =============================================================================


def construct_wait_operator() -> Operator:
    """Construct a wait operator for multi-robot coordination.

    Allows one robot to wait for another robot to become free.

    Returns:
        Operator for waiting on another robot.
    """
    return Operator(
        name="wait",
        parameters=[("?r1", "robot"), ("?r2", "robot")],
        preconditions=[F("free ?r1"), ~F("free ?r2"), ~F("waiting ?r2 ?r1")],
        effects=[Effect(time=0, resulting_fluents={F("not free ?r1"), F("waiting ?r1 ?r2")})],
    )


def construct_no_op_operator(no_op_time: OptNumeric, extra_cost: float = 0.0) -> Operator:
    """Construct a no-op (do nothing) operator.

    Sometimes patience is a virtuous skill.

    Args:
        no_op_time: Time or function for no-op duration.
            Function signature: (robot,) -> float
        extra_cost: Additional cost for the no-op action (default: 0.0).

    Returns:
        Operator for waiting in place.
    """
    no_op_time_fn = _to_numeric(no_op_time)
    return Operator(
        name="no_op",
        parameters=[("?r", "robot")],
        preconditions=[F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(time=(no_op_time_fn, ["?r"]), resulting_fluents={F("free ?r")}),
        ],
        extra_cost=extra_cost,
    )

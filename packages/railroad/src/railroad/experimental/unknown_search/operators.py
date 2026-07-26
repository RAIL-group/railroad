"""Operator constructors for frontier-based exploration and unknown-space navigation."""

from railroad.core import Fluent, Operator, Effect
from railroad.operators._utils import OptNumeric, _to_numeric

from .statistics import ObjectFindLike, as_find_probability

F = Fluent


def construct_move_navigable_operator(move_time: OptNumeric) -> Operator:
    """Construct a move operator with just-moved throttling.

    Suitable for unknown-space navigation where the environment controls the
    set of valid locations through ``objects_by_type["location"]``.

    Multiple robots may target the same destination concurrently: there is no
    destination reservation here. Exclusivity that actually matters --- two
    robots must not both *explore* the same frontier --- is enforced at the
    explore action via ``lock-explore``, not by pre-claiming travel targets.
    (A robot is still free to drive to a shared destination such as the goal.)

    Args:
        move_time: Time or function to compute movement duration.
            Function signature: (robot, from_location, to_location) -> float

    Returns:
        Operator for moving to a known target location.
    """
    move_time_fn = _to_numeric(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[
            F("at ?r ?from"),
            F("free ?r"),
            ~F("just-moved ?r"),
        ],
        effects=[
            Effect(
                time=0,
                resulting_fluents={F("not free ?r"), F("not at ?r ?from")},
            ),
            Effect(
                time=(move_time_fn, ["?r", "?from", "?to"]),
                resulting_fluents={
                    F("free ?r"),
                    F("at ?r ?to"),
                    F("just-moved ?r"),
                },
            ),
            Effect(
                time=(move_time_fn + 0.1, ["?r", "?from", "?to"]),
                resulting_fluents={~F("just-moved ?r")},
            ),
        ],
    )


def construct_search_at_site_operator(
    object_find_prob: ObjectFindLike,
    search_time: OptNumeric,
    *,
    container_type: str | None = None,
) -> Operator:
    """Construct a search operator for hidden candidate sites.

    Like the standard search operator but restricts search to known
    hidden-object candidate locations.

    Args:
        object_find_prob: An
            :class:`~railroad.experimental.unknown_search.statistics.ObjectFindEstimator`
            (whose ``container_probability`` is read per grounding, so the
            estimator can be swapped on a built environment), or — as before —
            a constant or a ``(robot, location, object) -> float`` callable.
        search_time: Time or function for search duration.
            Function signature: (robot, location, object) -> float
        container_type: If set, use this as the parameter type for the
            location (e.g. ``"container"``) instead of requiring a
            ``(candidate-site ?loc)`` fluent precondition.

    Returns:
        Operator for searching a candidate site.
    """
    prob_fn = _to_numeric(as_find_probability(object_find_prob, kind="container"))
    time_fn = _to_numeric(search_time)

    loc_param_type = container_type or "location"
    preconditions = [
        F("at ?r ?loc"),
        F("free ?r"),
        ~F("revealed ?loc"),
        ~F("searched ?loc ?obj"),
        ~F("found ?obj"),
        ~F("lock-search ?loc"),
    ]
    if container_type is None:
        preconditions.insert(2, F("candidate-site ?loc"))

    return Operator(
        name="search",
        parameters=[("?r", "robot"), ("?loc", loc_param_type), ("?obj", "object")],
        preconditions=preconditions,
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("lock-search ?loc")}),
            Effect(
                time=(time_fn, ["?r", "?loc", "?obj"]),
                resulting_fluents={
                    F("free ?r"),
                    F("searched ?loc ?obj"),
                    F("not lock-search ?loc"),
                },
                prob_effects=[
                    (
                        (prob_fn, ["?r", "?loc", "?obj"]),
                        [Effect(time=0, resulting_fluents={F("found ?obj"), F("at ?obj ?loc")})],
                    ),
                    (
                        (1 - prob_fn, ["?r", "?loc", "?obj"]),
                        [],
                    ),
                ],
            ),
        ],
    )


def construct_explore_frontier_operator(
    explore_time: OptNumeric = 5.0,
    completion_prob: OptNumeric = 0.2,
) -> Operator:
    """Construct ``(explore ?robot ?frontier)`` for pure exploration.

    Exploring a frontier may symbolically complete exploration with
    probability ``completion_prob``. This sampled ``exploration-complete``
    outcome is only a planning prior: after each action the environment's
    ``refresh_frontiers()`` overwrites the fluent from the true frontier set,
    so execution terminates exactly when no reachable frontiers remain.

    Args:
        explore_time: Time or function for the explore duration.
            Function signature: (robot, frontier) -> float
        completion_prob: Probability or function that exploring this frontier
            completes exploration. Function signature: (robot, frontier) -> float

    Returns:
        Operator for exploring a frontier.
    """
    prob_fn = _to_numeric(completion_prob)
    time_fn = _to_numeric(explore_time)

    return Operator(
        name="explore",
        parameters=[("?r", "robot"), ("?f", "frontier")],
        preconditions=[
            F("at ?r ?f"),
            F("free ?r"),
            ~F("explored ?f"),
            ~F("lock-explore ?f"),
        ],
        effects=[
            Effect(
                time=0,
                resulting_fluents={F("not free ?r"), F("lock-explore ?f")},
            ),
            Effect(
                time=(time_fn, ["?r", "?f"]),
                resulting_fluents={
                    F("free ?r"),
                    F("explored ?f"),
                    F("not lock-explore ?f"),
                },
                prob_effects=[
                    (
                        (prob_fn, ["?r", "?f"]),
                        [Effect(time=0, resulting_fluents={F("exploration-complete")})],
                    ),
                    (
                        (1 - prob_fn, ["?r", "?f"]),
                        [],
                    ),
                ],
            ),
        ],
    )


def construct_search_frontier_operator(
    object_find_prob: ObjectFindLike,
    search_time: OptNumeric,
) -> Operator:
    """Construct ``(search-frontier ?robot ?frontier ?object)``.

    This is a frontier-conditioned object-search action. On success the
    object is marked found and assigned to the frontier location for symbolic
    planning purposes.

    Args:
        object_find_prob: An
            :class:`~railroad.experimental.unknown_search.statistics.ObjectFindEstimator`
            (whose ``frontier_probability`` is read per grounding, so the
            estimator can be swapped on a built environment), or — as before —
            a constant or a ``(robot, frontier, object) -> float`` callable.
        search_time: Time or function for search duration.
            Function signature: (robot, frontier, object) -> float

    Returns:
        Operator for searching an object from a frontier.
    """
    prob_fn = _to_numeric(as_find_probability(object_find_prob, kind="frontier"))
    time_fn = _to_numeric(search_time)

    return Operator(
        name="search-frontier",
        parameters=[
            ("?r", "robot"),
            ("?frontier", "frontier"),
            ("?obj", "object"),
        ],
        preconditions=[
            F("at ?r ?frontier"),
            F("free ?r"),
            ~F("found ?obj"),
            ~F("searched-frontier ?frontier ?obj"),
            ~F("lock-search-frontier ?obj"),
        ],
        effects=[
            Effect(
                time=0,
                resulting_fluents={F("not free ?r"), F("lock-search-frontier ?obj")},
            ),
            Effect(
                time=(time_fn, ["?r", "?frontier", "?obj"]),
                resulting_fluents={
                    F("free ?r"),
                    F("searched-frontier ?frontier ?obj"),
                    F("not lock-search-frontier ?obj"),
                },
                prob_effects=[
                    (
                        (prob_fn, ["?r", "?frontier", "?obj"]),
                        [
                            Effect(
                                time=0,
                                resulting_fluents={
                                    F("found ?obj"),
                                    F("at ?obj ?frontier"),
                                },
                            )
                        ],
                    ),
                    (
                        (1 - prob_fn, ["?r", "?frontier", "?obj"]),
                        [],
                    ),
                ],
            ),
        ],
    )

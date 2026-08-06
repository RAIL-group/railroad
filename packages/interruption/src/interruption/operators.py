from railroad.core import Effect, Operator
from railroad.core import Fluent as F
from railroad.operators._utils import OptNumeric, _to_numeric


# custom operator functions
def construct_assemble_operator(assemble_time: int):
    """
    Constructs an assemble sandwhich operator.
    """
    assemble = Operator(
            name="assemble",
            parameters=[
                ("?r", "robot"), ("?o1", "object"), ("?o2", "object"),
                ("?o3", "object"), ("?l", "location")
            ],
            preconditions=[
                F("free ?r"), F("is-turkey ?o1"), F("is-bread ?o2"), F("at ?o1 ?l"),
                F("at ?o2 ?l"), F("at ?r ?l"), ~F("hand-full ?r"), F("prep-station ?l"),
                F("is-sandwhich ?o3")
            ],
            effects=[
                Effect(time=0, resulting_fluents={F("not free ?r"), F("hand-full ?r")}),
                Effect(time=assemble_time, resulting_fluents={
                    F("free ?r"), F("not at ?o1 ?l"), F("not at ?o2 ?l"),
                    F("sandwhich-made"), ~F("hand-full ?r"),
                    F("at ?o3 ?l")
                })
            ]
        )
    return assemble


def construct_gripper_pick_operator(pick_time: OptNumeric) -> Operator:
    """Construct a basic pick operator (non-blocking).

    Args:
        pick_time: Time or function for pick duration.
            Function signature: (robot, gripper, location, object) -> float

    Returns:
        Operator for picking up an object.
    """
    pick_time_fn = _to_numeric(pick_time)
    return Operator(
        name="pick",
        parameters=[("?r", "robot"), ("?g", "gripper"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("at ?obj ?loc"), ~F("hand-full ?g")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?obj ?loc")}),
            Effect(
                time=(pick_time_fn, ["?r", "?g", "?loc", "?obj"]),
                resulting_fluents={F("free ?r"), F("holding ?g ?obj"), F("hand-full ?g")},
            ),
        ],
    )


def construct_gripper_place_operator(place_time: OptNumeric) -> Operator:
    """Construct a basic place operator (non-blocking).

    Args:
        place_time: Time or function for place duration.
            Function signature: (robot, gripper, location, object) -> float

    Returns:
        Operator for placing an object.
    """
    place_time_fn = _to_numeric(place_time)
    return Operator(
        name="place",
        parameters=[("?r", "robot"), ("?g", "gripper"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("holding ?g ?obj"), F("hand-full ?g")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not holding ?g ?obj")}),
            Effect(
                time=(place_time_fn, ["?r", "?g", "?loc", "?obj"]),
                resulting_fluents={F("free ?r"), F("at ?obj ?loc"), ~F("hand-full ?g")},
            ),
        ],
    )

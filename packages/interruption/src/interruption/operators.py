from railroad.core import Effect, Operator
from railroad.core import Fluent as F
from railroad.operators._utils import OptNumeric, _to_numeric


# The hand-specific pick/place operators also keep `at-category ?obj ?cat` up to
# date: the object is somewhere of category ?cat (e.g. diningtable), whichever
# instance. Tasks are stated as at-category goals, so any instance of a location
# category satisfies them. ?cat is pinned by the static fact `category ?loc ?cat`,
# which grounding compiles away, so it adds no actions. It comes right after ?r so
# the location and object stay the last two words of the action name, which
# get_updated_scene_graph parses.

# Blocking facts, as railroad's `_blocking` operators use them: true for 0.1 after
# the action that sets them ends, and a precondition of the actions they block.
#   just-moved ?r         - no move straight after a move (the environment is fully
#                           connected, so moving twice is never cheaper than once)
#   just-picked ?r ?obj   - no placing an object straight after picking it up
#   just-placed ?r ?obj   - no picking an object straight up after placing it
# They only constrain plans; a planner's starting state drops them (see
# planning_framework.planning_state), since a new plan may need to start with a
# move even if the robot has just moved.
BLOCKING_PREDICATES = frozenset({"just-moved", "just-picked", "just-placed"})
# how long after the blocking action ends its blocking fact stays true
BLOCKING_DURATION = 0.1


def only_clears_blocking_facts(effect) -> bool:
    """Whether a (pending) effect only makes blocking facts false: a clean-up."""
    return (
        not effect.is_probabilistic and not effect.is_conditional
        and all(f.negated and f.name in BLOCKING_PREDICATES for f in effect.resulting_fluents)
    )

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


def construct_pick_with_left_hand_operator(pick_time: OptNumeric) -> Operator:
    """Construct a basic pick with left hand operator (non-blocking).

    Args:
        pick_time: Time or function for pick duration.
            Function signature: (robot, location, object) -> float

    Returns:
        Operator for picking up an object with the robot's left hand.
        See the note at the top of this module for ?cat.
    """
    return _hand_pick_operator("left", pick_time, blocking=False)


def construct_pick_with_right_hand_operator(pick_time: OptNumeric) -> Operator:
    """Construct a basic pick with right hand operator (non-blocking).

    Args:
        pick_time: Time or function for pick duration.
            Function signature: (robot, location, object) -> float

    Returns:
        Operator for picking up an object with the robot's right hand.
        See the note at the top of this module for ?cat.
    """
    return _hand_pick_operator("right", pick_time, blocking=False)


def construct_place_with_left_hand_operator(place_time: OptNumeric) -> Operator:
    """Construct a basic place with left hand operator (non-blocking).

    Args:
        place_time: Time or function for place duration.
            Function signature: (robot, location, object) -> float

    Returns:
        Operator for placing an object that was in the robot's left hand.
        See the note at the top of this module for ?cat.
    """
    return _hand_place_operator("left", place_time, blocking=False)


def construct_place_with_right_hand_operator(place_time: OptNumeric) -> Operator:
    """Construct a basic place with right hand operator (non-blocking).

    Args:
        place_time: Time or function for place duration.
            Function signature: (robot, location, object) -> float

    Returns:
        Operator for placing an object that was in the robot's right hand.
        See the note at the top of this module for ?cat.
    """
    return _hand_place_operator("right", place_time, blocking=False)


def construct_pick_with_left_hand_operator_blocking(pick_time: OptNumeric) -> Operator:
    """
    construct_pick_with_left_hand_operator with just-placed blocking: an object
    can't be picked straight back up after being placed, and picking sets
    just-picked (see BLOCKING_PREDICATES).
    """
    return _hand_pick_operator("left", pick_time, blocking=True)


def construct_pick_with_right_hand_operator_blocking(pick_time: OptNumeric) -> Operator:
    """construct_pick_with_right_hand_operator with just-placed blocking."""
    return _hand_pick_operator("right", pick_time, blocking=True)


def construct_place_with_left_hand_operator_blocking(place_time: OptNumeric) -> Operator:
    """
    construct_place_with_left_hand_operator with just-picked blocking: an object
    can't be placed straight after being picked up, and placing sets just-placed
    (see BLOCKING_PREDICATES).
    """
    return _hand_place_operator("left", place_time, blocking=True)


def construct_place_with_right_hand_operator_blocking(place_time: OptNumeric) -> Operator:
    """construct_place_with_right_hand_operator with just-picked blocking."""
    return _hand_place_operator("right", place_time, blocking=True)


# ?cat right after ?r, so the location and object are the action name's last two words
_HAND_PARAMETERS = [("?r", "robot"), ("?cat", "category"), ("?loc", "location"), ("?obj", "object")]


def _hand_pick_operator(hand: str, pick_time: OptNumeric, blocking: bool) -> Operator:
    """pick-left / pick-right; the right hand is only used once the left is full."""
    pick_time_fn = _to_numeric(pick_time)
    timing_args = ["?r", "?loc", "?obj"]
    hand_preconditions = (
        [~F("left-hand-full ?r")] if hand == "left"
        else [F("left-hand-full ?r"), ~F("right-hand-full ?r")]
    )
    preconditions = [
        F("at ?r ?loc"), F("free ?r"), F("at ?obj ?loc"), *hand_preconditions,
        F("category ?loc ?cat"),
    ]
    end_fluents = {F("free ?r"), F(f"holding-in-{hand} ?r ?obj"), F(f"{hand}-hand-full ?r")}
    effects = [
        Effect(time=0, resulting_fluents={
            F("not free ?r"), F("not at ?obj ?loc"), F("not at-category ?obj ?cat")
        }),
    ]
    if blocking:
        preconditions.append(~F("just-placed ?r ?obj"))
        end_fluents.add(F("just-picked ?r ?obj"))
    effects.append(Effect(time=(pick_time_fn, timing_args), resulting_fluents=end_fluents))
    if blocking:
        effects.append(Effect(
            time=(pick_time_fn + BLOCKING_DURATION, timing_args),
            resulting_fluents={~F("just-picked ?r ?obj")},
        ))
    return Operator(
        name=f"pick-{hand}",
        parameters=_HAND_PARAMETERS,
        preconditions=preconditions,
        effects=effects,
    )


def _hand_place_operator(hand: str, place_time: OptNumeric, blocking: bool) -> Operator:
    """place-left / place-right."""
    place_time_fn = _to_numeric(place_time)
    timing_args = ["?r", "?loc", "?obj"]
    preconditions = [
        F("at ?r ?loc"), F("free ?r"), F(f"holding-in-{hand} ?r ?obj"), F(f"{hand}-hand-full ?r"),
        F("category ?loc ?cat"),
    ]
    end_fluents = {
        F("free ?r"), F("at ?obj ?loc"), F("at-category ?obj ?cat"), ~F(f"{hand}-hand-full ?r"),
    }
    effects = [
        Effect(time=0, resulting_fluents={F("not free ?r"), F(f"not holding-in-{hand} ?r ?obj")}),
    ]
    if blocking:
        preconditions.append(~F("just-picked ?r ?obj"))
        end_fluents.add(F("just-placed ?r ?obj"))
    effects.append(Effect(time=(place_time_fn, timing_args), resulting_fluents=end_fluents))
    if blocking:
        effects.append(Effect(
            time=(place_time_fn + BLOCKING_DURATION, timing_args),
            resulting_fluents={~F("just-placed ?r ?obj")},
        ))
    return Operator(
        name=f"place-{hand}",
        parameters=_HAND_PARAMETERS,
        preconditions=preconditions,
        effects=effects,
    )

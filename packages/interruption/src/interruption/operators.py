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


def construct_pick_with_left_hand_operator_blocking(pick_time: OptNumeric) -> Operator:
    """
    pick-left: the robot picks an object up with its left hand. Blocking: an
    object can't be picked straight back up after being placed, and picking sets
    just-picked (see BLOCKING_PREDICATES). See the note at the top of this module
    for ?cat.

    Args:
        pick_time: Time or function for pick duration.
            Function signature: (robot, location, object) -> float
    """
    return _hand_pick_operator("left", pick_time)


def construct_place_with_left_hand_operator_blocking(place_time: OptNumeric) -> Operator:
    """
    place-left: the robot places the object in its left hand. Blocking: an object
    can't be placed straight after being picked up, and placing sets just-placed
    (see BLOCKING_PREDICATES). See the note at the top of this module for ?cat.

    Args:
        place_time: Time or function for place duration.
            Function signature: (robot, location, object) -> float
    """
    return _hand_place_operator("left", place_time)


# ?cat right after ?r, so the location and object are the action name's last two words
_HAND_PARAMETERS = [("?r", "robot"), ("?cat", "category"), ("?loc", "location"), ("?obj", "object")]


def _hand_pick_operator(hand: str, pick_time: OptNumeric) -> Operator:
    """pick-left / pick-right; the right hand is only used once the left is full."""
    pick_time_fn = _to_numeric(pick_time)
    timing_args = ["?r", "?loc", "?obj"]
    hand_preconditions = (
        [~F("left-hand-full ?r")] if hand == "left"
        else [F("left-hand-full ?r"), ~F("right-hand-full ?r")]
    )
    preconditions = [
        F("at ?r ?loc"), F("free ?r"), F("at ?obj ?loc"), *hand_preconditions,
        F("category ?loc ?cat"), ~F("just-placed ?r ?obj"),
    ]
    effects = [
        Effect(time=0, resulting_fluents={
            F("not free ?r"), F("not at ?obj ?loc"), F("not at-category ?obj ?cat")
        }),
        Effect(time=(pick_time_fn, timing_args), resulting_fluents={
            F("free ?r"), F(f"holding-in-{hand} ?r ?obj"), F(f"{hand}-hand-full ?r"),
            F("just-picked ?r ?obj"),
        }),
        Effect(
            time=(pick_time_fn + BLOCKING_DURATION, timing_args),
            resulting_fluents={~F("just-picked ?r ?obj")},
        ),
    ]
    return Operator(
        name=f"pick-{hand}",
        parameters=_HAND_PARAMETERS,
        preconditions=preconditions,
        effects=effects,
    )


def _hand_place_operator(hand: str, place_time: OptNumeric) -> Operator:
    """place-left / place-right."""
    place_time_fn = _to_numeric(place_time)
    timing_args = ["?r", "?loc", "?obj"]
    preconditions = [
        F("at ?r ?loc"), F("free ?r"), F(f"holding-in-{hand} ?r ?obj"), F(f"{hand}-hand-full ?r"),
        F("category ?loc ?cat"), ~F("just-picked ?r ?obj"),
    ]
    effects = [
        Effect(time=0, resulting_fluents={F("not free ?r"), F(f"not holding-in-{hand} ?r ?obj")}),
        Effect(time=(place_time_fn, timing_args), resulting_fluents={
            F("free ?r"), F("at ?obj ?loc"), F("at-category ?obj ?cat"),
            ~F(f"{hand}-hand-full ?r"), F("just-placed ?r ?obj"),
        }),
        Effect(
            time=(place_time_fn + BLOCKING_DURATION, timing_args),
            resulting_fluents={~F("just-placed ?r ?obj")},
        ),
    ]
    return Operator(
        name=f"place-{hand}",
        parameters=_HAND_PARAMETERS,
        preconditions=preconditions,
        effects=effects,
    )

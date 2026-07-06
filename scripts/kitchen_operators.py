"""Kitchen domain operators for the multi-robot simulator.

Follows the same construction pattern as railroad.operators.core:
each function returns an Operator whose preconditions and effects
are expressed using railroad.core.Fluent / Effect.

Design intent
-------------
The only valid plan for goal "chopped herbs" is:
  chop carrot → place knife → stir → pick knife again → chop_garnish herbs

The knife gap (place then re-pick) must emerge from planning:
- stir requires free ?r  (empty hands, so robot must put knife down first)
- chop_garnish requires stirred bowl  (stir must have completed first)
"""

from railroad.core import Fluent, Operator, Effect

F = Fluent


def construct_chop_operator(chop_time: float = 5.0) -> Operator:
    """Chop an item at the counter.

    Preconditions: holding ?r knife, at ?r counter, at ?item counter
    Add effects:   chopped ?item
    Duration:      chop_time (default 5.0)

    Note: no delete effects here — chop does not consume the item.
    The knife stays held; knife placement is a separate place action.
    """
    return Operator(
        name="chop",
        parameters=[("?r", "robot"), ("?item", "object")],
        preconditions=[
            F("holding ?r knife"),
            F("at ?r counter"),
            F("at ?item counter"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(
                time=chop_time,
                resulting_fluents={F("chopped ?item"), F("free ?r")},
            ),
        ],
    )


def construct_stir_operator(stir_time: float = 12.0) -> Operator:
    """Stir the bowl at the counter.

    Preconditions: free ?r (empty hands), at ?r counter, chopped carrot
    Add effects:   stirred bowl
    Duration:      stir_time (default 12.0)

    Requiring free ?r forces the robot to put the knife down before stirring,
    which creates the two-knife-use structure in any valid plan for chopped herbs.
    """
    return Operator(
        name="stir",
        parameters=[("?r", "robot")],
        preconditions=[
            F("free ?r"),
            F("at ?r counter"),
            F("chopped carrot"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(
                time=stir_time,
                resulting_fluents={F("stirred bowl"), F("free ?r")},
            ),
        ],
    )


def construct_chop_garnish_operator(chop_time: float = 5.0) -> Operator:
    """Chop herbs (garnish) at the counter after the bowl has been stirred.

    Preconditions: holding ?r knife, at ?r counter, at herbs counter, stirred bowl
    Add effects:   chopped herbs
    Duration:      chop_time (default 5.0)

    The stirred bowl precondition serialises chop_garnish after stir,
    preventing the robot from skipping the stir step.
    """
    return Operator(
        name="chop-garnish",
        parameters=[("?r", "robot")],
        preconditions=[
            F("holding ?r knife"),
            F("at ?r counter"),
            F("at herbs counter"),
            F("stirred bowl"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(
                time=chop_time,
                resulting_fluents={F("chopped herbs"), F("free ?r")},
            ),
        ],
    )

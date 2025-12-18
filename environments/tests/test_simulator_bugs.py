import logging
import pytest
import numpy as np
from mrppddl.core import Fluent as F, State, get_action_by_name
import environments
from environments.core import EnvironmentInterface as Simulator
from environments import SimpleEnvironment

# Fancy error handling; shows local vars
from rich.traceback import install
install(show_locals=True)


@pytest.mark.timeout(15)
def test_simulator_looping_bug():
    LOCATIONS = {
        "living_room": np.array([0, 0]),
        "kitchen": np.array([10, 0]),
        "den": np.array([15, 5]),
    }

    OBJECTS_AT_LOCATIONS = {
        "living_room": {"object": {"Remote"}},
        "kitchen": {"object": {"Cookie", "Plate"}},
        "den": {"object": set()},
    }
    # Initialize environment
    env = SimpleEnvironment(LOCATIONS, OBJECTS_AT_LOCATIONS, num_robots=2)

    # Define the objects we're looking for
    objects_of_interest = ["Remote", "Cookie", "Plate"]

    # Define initial state
    initial_state = State(
        time=0,
        fluents={
            # Robots free and start in (revealed) living room
            F("free robot1"),
            F("free robot2"),
            F("at robot1 living_room"),
            F("at robot2 living_room"),
            F("revealed living_room"),
            F("at Remote living_room"),
            F("found Remote"),
            F("revealed den"),
        },
    )

    # Initial objects by type (robot only knows about some objects initially)
    objects_by_type = {
        "robot": ["robot1", "robot2"],
        "location": list(LOCATIONS.keys()),
        "object": objects_of_interest,  # Robot knows these objects exist
    }

    # Create operators
    move_op = environments.operators.construct_move_operator(
        move_time=env.get_move_cost_fn()
    )

    # Search operator with 80% success rate when object is actually present
    object_find_prob = lambda r, loc, o: 0.8 if o in OBJECTS_AT_LOCATIONS.get(loc, dict()).get("object", dict()) else 0.2

    random_cost = np.random.uniform(0, 0.1)
    search_op = environments.operators.construct_search_operator(
        object_find_prob=object_find_prob,
        search_time=lambda r, loc: 5.0 + random_cost
    )

    from mrppddl.core import Operator, Effect
    no_op = Operator(
        name="no-op",
        parameters=[("?r", "robot")],
        preconditions=[F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(time=5, resulting_fluents={F("free ?r")}),
        ],
        extra_cost=100,
    )
    # add 5.0 and random small noise cost
    random_cost = np.random.uniform(0, 0.1)
    pick_op = environments.operators.construct_pick_operator(
        pick_time=lambda r, l, o: 5.0 + random_cost
    )

    random_cost = np.random.uniform(0, 0.1)
    place_op = environments.operators.construct_place_operator(
        place_time=lambda r, l, o: 5.0 + random_cost
    )

    # Create simulator
    sim = Simulator(
        initial_state,
        objects_by_type,
        [no_op, pick_op, place_op, move_op, search_op],
        env
    )

    actions_to_take = [
        "pick robot2 living_room Remote",
        "move robot1 living_room kitchen",
        "move robot2 living_room kitchen",
        "search robot1 kitchen Cookie",
        "pick robot1 kitchen Cookie",
        "place robot2 kitchen Remote",
        "pick robot2 kitchen Plate",
        "move robot1 kitchen den",
    ]

    all_actions = sim.get_actions()

    for action_name in actions_to_take:
        action = get_action_by_name(all_actions, action_name)
        sim.advance(action, do_interrupt=False)

    # if doesn't loop forever, test passes
    assert True
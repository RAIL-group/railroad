import pytest
import numpy as np
from mrppddl.core import Fluent as F, State, get_action_by_name
import environments
from environments.core import EnvironmentInterface
from environments import SimpleEnvironment

# Fancy error handling; shows local vars
from rich.traceback import install
install(show_locals=True)


@pytest.mark.timeout(15)
def test_simulator_advance_blocking_operators():
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
    robot_locations = {'robot1': 'living_room', 'robot2': 'living_room'}
    # Initialize environment
    env = SimpleEnvironment(LOCATIONS, OBJECTS_AT_LOCATIONS, robot_locations)

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
        "robot": robot_locations.keys(),
        "location": list(LOCATIONS.keys()),
        "object": objects_of_interest,  # Robot knows these objects exist
    }

    # Create operators
    move_time_fn = env.get_skills_cost_fn(skill_name='move')
    search_time = env.get_skills_cost_fn(skill_name='search')
    pick_time = env.get_skills_cost_fn(skill_name='pick')
    place_time = env.get_skills_cost_fn(skill_name='place')
    object_find_prob = lambda r, loc, o: 0.8 if o in OBJECTS_AT_LOCATIONS.get(loc, dict()).get("object", dict()) else 0.2

    move_op = environments.operators.construct_move_operator(move_time_fn)
    search_op = environments.operators.construct_search_operator(object_find_prob, search_time)
    pick_op = environments.operators.construct_pick_operator(pick_time)
    place_op = environments.operators.construct_place_operator(place_time)
    no_op = environments.operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Create simulator
    sim = EnvironmentInterface(
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
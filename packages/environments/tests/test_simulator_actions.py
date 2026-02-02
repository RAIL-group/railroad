import numpy as np
from railroad.core import Fluent as F, State, get_action_by_name, LiteralGoal
from railroad.planner import MCTSPlanner
from railroad import operators
from railroad.experimental.environment import EnvironmentInterface, SimpleEnvironment


LOCATIONS = {
    "start": np.array([0, 0]),
    "roomA": np.array([10, 0]),
    "roomB": np.array([0, 15]),
    "roomC": np.array([15, 15]),
}

OBJECTS_AT_LOCATIONS = {
    "start": dict(),
    "roomA": {"object": {"objA", "objC"}},
    "roomB": {"object": {"objB"}},
    "roomC": {"object": {"objD"}},
}

SKILLS_TIME = {
    'r1': {
        'pick': 15,
        'place': 15,
        'search': 15},
    'r2': {
        'pick': 10,
        'place': 10,
        'search': 10}
}


class TestEnvironment(SimpleEnvironment):
    __test__ = False  # Tell pytest this is not a test class

    def get_skills_time_fn(self, skill_name: str):
        if skill_name == 'move':
            return super()._get_move_cost_fn()
        else:
            def get_skill_time(robot_name, *args, **kwargs):
                return SKILLS_TIME[robot_name][skill_name]
            return get_skill_time


def test_move_action():
    '''Test that move action is interrupted correctly.'''
    robot_locations = {"r1": "start", "r2": "start"}
    objects_by_type = {
        "robot": robot_locations.keys(),
        "location": {"start", "roomA", "roomB", "roomC"},
    }

    initial_state = State(
        time=0,
        fluents={
            F("at", "r1", robot_locations["r1"]),
            F("at", "r2", robot_locations["r2"]),
            F("free", "r1"),
            F("free", "r2"),
        },
    )
    env = SimpleEnvironment(
        locations=LOCATIONS, objects_at_locations=OBJECTS_AT_LOCATIONS, robot_locations=robot_locations)

    move_time_fn = env.get_skills_time_fn(skill_name='move')
    move_op = operators.construct_move_operator_blocking(move_time_fn)

    sim = EnvironmentInterface(initial_state, objects_by_type, [move_op], env)
    actions = sim.get_actions()
    a1 = get_action_by_name(actions, "move r1 start roomA")
    sim.advance(a1)
    a2 = get_action_by_name(actions, "move r2 start roomB")
    sim.advance(a2)

    assert F("free r1") in sim.state.fluents
    assert F("free r2") in sim.state.fluents
    assert F("at r1 start") not in sim.state.fluents
    assert F("at r2 start") not in sim.state.fluents
    assert F("at r1 roomA") in sim.state.fluents
    assert F("at r2 roomB") not in sim.state.fluents
    assert F("at r2 r2_loc") in sim.state.fluents
    assert F("at r1 r1_loc") not in sim.state.fluents
    assert sim.state.time == 10.0
    assert len(sim.ongoing_actions) == 0
    assert np.all(sim.environment.locations["r2_loc"] == (0.0, 10.0))  # type: ignore[union-attr]


def test_search_action():
    robot_locations = {"r1": "start", "r2": "start"}
    objects_by_type = {
        "robot": robot_locations.keys(),
        "location": {"start", "roomA", "roomB", "roomC"},
        "object": {"objA", "objB"}
    }

    initial_state = State(
        time=0,
        fluents={
            F("at", "r1", "roomA"),
            F("at", "r2", "roomB"),
            F("free", "r1"),
            F("free", "r2"),
        },
    )
    env = TestEnvironment(
        locations=LOCATIONS, objects_at_locations=OBJECTS_AT_LOCATIONS, robot_locations=robot_locations)

    search_time = env.get_skills_time_fn(skill_name='search')
    object_find_prob = lambda r, l, o: 0.8 if l == "roomA" else 0.2  # noqa: E731, E741
    search_op = operators.construct_search_operator(object_find_prob, search_time)

    sim = EnvironmentInterface(initial_state, objects_by_type, [search_op], env)

    actions = sim.get_actions()
    a1 = get_action_by_name(actions, "search r1 roomA objA")
    sim.advance(a1)
    a2 = get_action_by_name(actions, "search r2 roomB objB")
    sim.advance(a2)

    assert F("free r1") not in sim.state.fluents
    assert F("at r1 roomA") in sim.state.fluents
    assert F("searched roomA objA") not in sim.state.fluents
    assert F("revealed roomA") not in sim.state.fluents
    assert F("found objA") not in sim.state.fluents
    assert F("found objC") not in sim.state.fluents
    assert F("at objA roomA") not in sim.state.fluents
    assert F("at objC roomA") not in sim.state.fluents
    assert F("lock-search roomA") in sim.state.fluents

    assert F("free r2") in sim.state.fluents
    assert F("at r2 roomB") in sim.state.fluents
    assert F("searched roomB objB") in sim.state.fluents
    assert F("found objB") in sim.state.fluents
    assert F("at objB roomB") in sim.state.fluents
    assert F("lock-search roomB") not in sim.state.fluents
    assert sim.state.time == 10

    assert len(sim.ongoing_actions) == 1


def test_pick_and_place_action():
    robot_locations = {"r1": "start", "r2": "start"}
    objects_by_type = {
        "robot": robot_locations.keys(),
        "location": {"start", "roomA", "roomB", "roomC"},
        "object": {"objA", "objB"}
    }

    initial_state = State(
        time=0,
        fluents={
            F("at", "r1", "roomA"), F("at", "objA",
                                      "roomA"), F("at", "objC", "roomA"),
            F("at", "r2", "roomB"), F("at", "objB", "roomB"),
            F("free", "r1"), F("free", "r2"),
        },
    )
    env = TestEnvironment(
        locations=LOCATIONS, objects_at_locations=OBJECTS_AT_LOCATIONS, robot_locations=robot_locations)

    pick_time = env.get_skills_time_fn(skill_name='pick')
    place_time = env.get_skills_time_fn(skill_name='place')
    pick_op = operators.construct_pick_operator(pick_time)
    place_op = operators.construct_place_operator(place_time)

    sim = EnvironmentInterface(initial_state, objects_by_type, [pick_op, place_op], env)

    actions = sim.get_actions()
    a1 = get_action_by_name(actions, "pick r1 roomA objA")
    sim.advance(a1)

    assert F("free r1") not in sim.state.fluents
    # assert F("free-arm r1") not in sim.state.fluents
    # assert F("free-arm r2") in sim.state.fluents
    assert sim.state.time == 0
    assert len(sim.ongoing_actions) == 1

    a2 = get_action_by_name(actions, "pick r2 roomB objB")
    sim.advance(a2)

    # R2 finishes picking objB first
    assert F("free r2") in sim.state.fluents
    # assert F("free-arm r2") not in sim.state.fluents
    assert F("at r2 roomB") in sim.state.fluents
    assert F("holding r2 objB") in sim.state.fluents
    assert F("at objB roomB") not in sim.state.fluents
    assert "objB" not in env._objects_at_locations["roomB"]["object"]
    assert sim.state.time == 10
    assert len(sim.ongoing_actions) == 1
    # R1 is still picking objA
    assert F("free r1") not in sim.state.fluents
    assert F("at r1 roomA") in sim.state.fluents
    assert F("holding r1 objA") not in sim.state.fluents
    assert F("at objA roomA") not in sim.state.fluents

    a3 = get_action_by_name(actions, "place r2 roomB objB")
    sim.advance(a3)
    # R1 finishes picking objA
    assert F("free r1") in sim.state.fluents
    # assert F("free-arm r1") not in sim.state.fluents
    assert F("at r1 roomA") in sim.state.fluents
    assert F("holding r1 objA") in sim.state.fluents
    assert F("at objA roomA") not in sim.state.fluents
    assert "objA" not in env._objects_at_locations["roomA"]["object"]

    # R2 is still placing objB
    assert F("free r2") not in sim.state.fluents
    assert F("at r2 roomB") in sim.state.fluents
    assert F("holding r2 objB") not in sim.state.fluents
    assert F("at objB roomB") not in sim.state.fluents
    assert sim.state.time == 15
    assert len(sim.ongoing_actions) == 1

    a4 = get_action_by_name(actions, "place r1 roomA objA")
    sim.advance(a4)
    # R2 finishes placing objB
    assert F("free r2") in sim.state.fluents
    # assert F("free-arm r2") in sim.state.fluents
    assert F("at r2 roomB") in sim.state.fluents
    assert F("holding r2 objB") not in sim.state.fluents
    assert F("at objB roomB") in sim.state.fluents
    assert "objB" in env._objects_at_locations["roomB"]["object"]
    # R1 is still placing objA
    assert F("free r1") not in sim.state.fluents
    assert F("at r1 roomA") in sim.state.fluents
    assert F("holding r1 objA") not in sim.state.fluents
    assert F("at objA roomA") not in sim.state.fluents
    assert sim.state.time == 20
    assert len(sim.ongoing_actions) == 1


# @pytest.mark.timeout(15)
def test_no_oscillation_pick_place_move_search():
    robot_locations = {"r1": "start"}

    OTHER_LOCATIONS = {
        "start": np.array([0, 0]),
        "bed_2": np.array([10, 0]),
        "dresser_3": np.array([0, 15]),
        "desk_4": np.array([15, 15]),
        "garbagecan_5": np.array([15, 0]),
    }

    OTHER_OBJECTS_AT_LOCATIONS = {
        "start": dict(),
        "bed_2": {"object": {"teddybear_6"}},
        "dresser_3": dict(),
        "desk_4": {"object": {"pencil_17"}},
        "garbagecan_5": dict(),
    }
    objects_by_type = {
        "robot": ["r1"],
        "location": ["start", "bed_2", "dresser_3", "desk_4", "garbagecan_5"],
        "object": ["teddybear_6", "pencil_17"],
    }

    # Check2: initial_state
    initial_fluents = {
        F("revealed start"),
        # F("at pencil_17 desk_4"),
        F("at r1 start"), F("free r1"),
    }
    initial_state = State(
        time=0,
        fluents=initial_fluents
    )

    # Check3: goal function
    goal = LiteralGoal(F("at pencil_17 garbagecan_5"))  # used to oscillate before fix

    env = TestEnvironment(locations=OTHER_LOCATIONS,
                          objects_at_locations=OTHER_OBJECTS_AT_LOCATIONS, robot_locations=robot_locations)

    move_time_fn = env.get_skills_time_fn(skill_name='move')
    search_time = env.get_skills_time_fn(skill_name='search')
    pick_time = env.get_skills_time_fn(skill_name='pick')
    place_time = env.get_skills_time_fn(skill_name='place')
    object_find_prob = lambda r, loc, o: 1.0

    move_op = operators.construct_move_operator_blocking(move_time_fn)
    search_op = operators.construct_search_operator(object_find_prob, search_time)
    pick_op = operators.construct_pick_operator_blocking(pick_time)
    place_op = operators.construct_place_operator_blocking(place_time)

    sim = EnvironmentInterface(initial_state, objects_by_type, [move_op, search_op, pick_op, place_op], env)

    all_actions = sim.get_actions()
    mcts = MCTSPlanner(all_actions)
    actions_taken = []

    # time the planning
    import time
    for _ in range(15):
        t1 = time.time()
        action_name = mcts(sim.state, goal, max_iterations=2000, c=20)
        t2 = time.time()
        print(f'Planning time: {t2 - t1:.2f} seconds')
        print(f'{action_name=}')
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action)
            # print(sim.state.fluents)
            actions_taken.append(action_name)

        if goal.evaluate(sim.state.fluents):
            print("Goal reached!")
            break

    print(f"Actions taken: {actions_taken}")
    assert goal.evaluate(sim.state.fluents)

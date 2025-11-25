import pytest
import numpy as np
import environments
from environments import Simulator
from mrppddl.core import State, Fluent as F
from mrppddl.core import get_action_by_name
from mrppddl.planner import MCTSPlanner


LOCATIONS = {
    "start": np.array([0, 0]),
    "bed_2": np.array([10, 0]),
    "dresser_3": np.array([0, 15]),
    "desk_4": np.array([15, 15]),
    "garbagecan_5": np.array([15, 0]),
}

OBJECTS_AT_LOCATIONS = {
    "start": dict(),
    "bed_2": {"object": {"teddybear_6"}},
    "dresser_3": dict(),
    "desk_4": {"object": {"pencil_17"}},
    "garbagecan_5": dict(),
}


class TestEnvironment(environments.BaseEnvironment):
    '''This is how the environment wrapper should look like for any simulator.'''
    def __init__(self, locations):
        super().__init__()
        self.locations = locations
        self._objects_at_locations = {loc: {"object": set()} for loc in locations}

    def get_objects_at_location(self, location):
        objects_found = OBJECTS_AT_LOCATIONS.get(location, {})
        for obj in objects_found:
            # update internal state
            self.add_object_at_location(obj, location)
        return objects_found

    def get_move_cost_fn(self):
        def get_move_time(robot, loc_from, loc_to):
            distance = np.linalg.norm(self.locations[loc_from] - self.locations[loc_to])
            return distance
        return get_move_time

    def get_intermediate_coordinates(self, time, loc_from, loc_to):
        coord_from = self.locations[loc_from]
        coord_to = self.locations[loc_to]
        direction = (coord_to - coord_from) / np.linalg.norm(coord_to - coord_from)
        new_coord = coord_from + direction * time
        return new_coord

    def remove_object_from_location(self, obj, location, object_type="object"):
        self._objects_at_locations[location][object_type].discard(obj)

    def add_object_at_location(self, obj, location, object_type="object"):
        self._objects_at_locations[location][object_type].add(obj)


@pytest.mark.timeout(15)
def test_more_heuristic():
    objects_by_type = {
        "robot": ["r1"],
        "location": ["start", "bed_2", "dresser_3", "desk_4", "garbagecan_5"],
        "object": ["teddybear_6", "pencil_17"],
    }

    # Check2: initial_state
    initial_fluents = {
        F("revealed start"),
        # F("at pencil_17 desk_4"),
        F("at r1 start"), F("free r1"), F("free-arm r1"),
    }
    initial_state = State(
        time=0,
        fluents=initial_fluents
    )

    # Check3: goal function
    goal_fluents = {
        # F("at teddybear_6 garbagecan_5"),   # works
        F("at pencil_17 garbagecan_5"),  # used to osccillate before fix
        # F("at pencil_17 bed_2"),  # used to osccillate before fix
    }

    env = TestEnvironment(locations=LOCATIONS)

    move_time_fn = env.get_move_cost_fn()
    search_time = lambda r, l: 10 if r == "r1" else 15
    pick_time = lambda r, l, o: 5 if r == "r1" else 7
    place_time = lambda r, l, o: 5 if r == "r1" else 7
    object_find_prob = lambda r, l, o: 1.0
    move_op = environments.simulator.actions.construct_move_operator(move_time_fn)
    search_op = environments.simulator.actions.construct_search_operator(object_find_prob, search_time)
    pick_op = environments.simulator.actions.construct_pick_operator(pick_time)
    place_op = environments.simulator.actions.construct_place_operator(place_time)

    sim = Simulator(initial_state, objects_by_type, [move_op, search_op, pick_op, place_op], env)

    all_actions = sim.get_actions()
    mcts = MCTSPlanner(all_actions)
    actions_taken = []

    # time the planning
    import time
    for _ in range(15):
        t1 = time.time()
        action_name = mcts(sim.state, goal_fluents, max_iterations=2000, c=20)
        t2 = time.time()
        print(f'Planning time: {t2 - t1:.2f} seconds')
        print(f'{action_name=}')
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            sim.advance(action)
            # print(sim.state.fluents)
            actions_taken.append(action_name)

        if sim.is_goal_reached(goal_fluents):
            print("Goal reached!")
            break

    print(f"Actions taken: {actions_taken}")
    assert sim.is_goal_reached(goal_fluents)

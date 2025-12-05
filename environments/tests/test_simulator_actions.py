import pytest
import numpy as np
from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl.planner import MCTSPlanner
import environments
from environments import BaseEnvironment, Robot, ActionStatus
from environments.core import EnvironmentInterface as Simulator


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


class TestEnvironment(BaseEnvironment):
    __test__ = False
    '''This is how the environment wrapper should look like for any simulator.'''
    def __init__(self, locations, object_oracle_locations, num_robots=2):
        super().__init__()
        self.locations = locations
        self._object_oracle_locations = object_oracle_locations
        self._objects_at_locations = {loc: {"object": set()} for loc in locations}
        self.robots = {
            f"r{i + 1}": Robot(name=f"r{i + 1}",
                               pose=locations["start"].copy(),
                               skills_time=SKILLS_TIME[f'r{i + 1}']) for i in range(num_robots)
        }

    def get_objects_at_location(self, location):
        objects_found = self._object_oracle_locations.get(location, {})
        for obj in objects_found:
            # update internal state
            self.add_object_at_location(obj, location)
        return objects_found

    def get_move_cost_fn(self):
        def get_move_time(robot, loc_from, loc_to):
            distance = np.linalg.norm(self.locations[loc_from] - self.locations[loc_to])
            return distance
        return get_move_time

    def get_intermediate_coordinates(self, time, loc_from, loc_to, is_coords=False):
        if not is_coords:
            coord_from = self.locations[loc_from]
            coord_to = self.locations[loc_to]
        else:
            coord_from = loc_from
            coord_to = loc_to
        direction = (coord_to - coord_from) / np.linalg.norm(coord_to - coord_from)
        new_coord = coord_from + direction * time
        return new_coord

    def remove_object_from_location(self, obj, location, object_type="object"):
        self._objects_at_locations[location][object_type].discard(obj)

    def add_object_at_location(self, obj, location, object_type="object"):
        self._objects_at_locations[location][object_type].add(obj)

    def move_robot(self, robot_name, location):
        target_coords = self.locations[location]
        self.robots[robot_name].move(target_coords, self.time)

    def pick_robot(self, robot_name):
        self.robots[robot_name].pick(self.time)

    def place_robot(self, robot_name):
        self.robots[robot_name].place(self.time)

    def search_robot(self, robot_name):
        self.robots[robot_name].search(self.time)

    def stop_robot(self, robot_name):
        self.robots[robot_name].stop()

    def _get_move_status(self, robot_name):
        all_robots_assigned = all(not r.is_free for r in self.robots.values())
        if not all_robots_assigned:
            return ActionStatus.IDLE
        robots_progress = np.array([self.time - r.start_time for r in self.robots.values()])
        time_to_target = [(n, np.linalg.norm(r.pose - r.target_pose)) for n, r in self.robots.items()]

        remaining_times = [(n, t - p) for (n, t), p in zip(time_to_target, robots_progress)]
        min_robot, min_distance = min(remaining_times, key=lambda x: x[1])

        if min_robot != robot_name:
            return ActionStatus.RUNNING

        # compute intermediate pose for all robots
        for r_name in self.robots:
            r_pose = self.get_intermediate_coordinates(
                min_distance, self.robots[r_name].pose, self.robots[r_name].target_pose, is_coords=True)
            self.robots[r_name].pose = r_pose
            self.locations[f'{r_name}_loc'] = r_pose

        # stop the robot that has reached its target
        self.robots[robot_name].stop()
        return ActionStatus.DONE

    def _get_pick_place_search_status(self, robot_name, action_name):
        all_robots_assigned = all(not r.is_free for r in self.robots.values())
        if not all_robots_assigned:
            return ActionStatus.IDLE
        robots_progress = np.array([self.time - r.start_time for r in self.robots.values()])
        time_to_action = [(n, r.skills_time[action_name]) for n, r in self.robots.items()]

        remaining_times = [(n, t - p) for (n, t), p in zip(time_to_action, robots_progress)]
        min_robot, _ = min(remaining_times, key=lambda x: x[1])

        if min_robot != robot_name:
            return ActionStatus.RUNNING

        self.stop_robot(robot_name)
        return ActionStatus.DONE

    def get_action_status(self, robot_name, action_name):
        if action_name == 'move':
            return self._get_move_status(robot_name)
        if action_name in ['pick', 'place', 'search']:
            return self._get_pick_place_search_status(robot_name, action_name)
        raise ValueError(f"Unknown action name: {action_name}")


def test_move_action():
    '''Test that move action is interrupted correctly.'''
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": {"start", "roomA", "roomB", "roomC"},
    }

    initial_state = State(
        time=0,
        fluents={
            F("at", "r1", "start"),
            F("at", "r2", "start"),
            F("free", "r1"),
            F("free", "r2"),
        },
    )
    env = TestEnvironment(locations=LOCATIONS, object_oracle_locations=OBJECTS_AT_LOCATIONS)
    move_op = environments.operators.construct_move_operator(move_time=env.get_move_cost_fn())

    sim = Simulator(initial_state, objects_by_type, [move_op], env)
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
    assert np.all(sim.environment.locations["r2_loc"] == (0.0, 10.0))


def test_search_action():
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": {"start", "roomA", "roomB", "roomC"},
        "object": {"objA", "objB"}
    }

    search_time = lambda r, l: SKILLS_TIME[r]['search']  # noqa: E731, E741
    object_find_prob = lambda r, l, o: 0.8 if l == "roomA" else 0.2  # noqa: E731, E741

    initial_state = State(
        time=0,
        fluents={
            F("at", "r1", "roomA"),
            F("at", "r2", "roomB"),
            F("free", "r1"),
            F("free", "r2"),
        },
    )
    env = TestEnvironment(locations=LOCATIONS, object_oracle_locations=OBJECTS_AT_LOCATIONS)
    search_op = environments.operators.construct_search_operator(object_find_prob=object_find_prob,
                                                                 search_time=search_time)

    sim = Simulator(initial_state, objects_by_type, [search_op], env)

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
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": {"start", "roomA", "roomB", "roomC"},
        "object": {"objA", "objB"}
    }

    pick_time = lambda r, l, o: SKILLS_TIME[r]['pick']  # noqa: E731, E741
    place_time = lambda r, l, o: SKILLS_TIME[r]['place']  # noqa: E731, E741

    initial_state = State(
        time=0,
        fluents={
            F("at", "r1", "roomA"), F("at", "objA",
                                      "roomA"), F("at", "objC", "roomA"),
            F("at", "r2", "roomB"), F("at", "objB", "roomB"),
            F("free", "r1"), F("free", "r2"),
        },
    )
    env = TestEnvironment(locations=LOCATIONS, object_oracle_locations=OBJECTS_AT_LOCATIONS)
    pick_op = environments.operators.construct_pick_operator(pick_time=pick_time)
    place_op = environments.operators.construct_place_operator(place_time=place_time)

    sim = Simulator(initial_state, objects_by_type, [pick_op, place_op], env)

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


# @pytest.mark.timeout(15)
def test_no_oscillation_pick_place_move_search():
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
    goal_fluents = {
        # F("at teddybear_6 garbagecan_5"),   # works
        F("at pencil_17 garbagecan_5"),  # used to osccillate before fix
        # F("at pencil_17 bed_2"),  # used to osccillate before fix
    }

    env = TestEnvironment(locations=OTHER_LOCATIONS, object_oracle_locations=OTHER_OBJECTS_AT_LOCATIONS, num_robots=1)

    move_time_fn = env.get_move_cost_fn()
    search_time = lambda r, l: 10 if r == "r1" else 15
    pick_time = lambda r, l, o: 5 if r == "r1" else 7
    place_time = lambda r, l, o: 5 if r == "r1" else 7
    object_find_prob = lambda r, l, o: 1.0
    move_op = environments.operators.construct_move_operator(move_time_fn)
    search_op = environments.operators.construct_search_operator(object_find_prob, search_time)
    pick_op = environments.operators.construct_pick_operator(pick_time)
    place_op = environments.operators.construct_place_operator(place_time)

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

        if sim.goal_reached(goal_fluents):
            print("Goal reached!")
            break

    print(f"Actions taken: {actions_taken}")
    assert sim.goal_reached(goal_fluents)

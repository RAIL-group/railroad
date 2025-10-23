import itertools
import roslibpy
import environments
from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl.core import transition

from mrppddl.core import OptCallable, Operator, Effect
from mrppddl.helper import _make_callable, _invert_prob

from mrppddl.planner import MCTSPlanner
from environments import BaseEnvironment

from typing import Dict, Set, List, Tuple, Callable

IDLE = -1
MOVING = 0
REACHED = 1



def construct_move_visited_operator(move_time: OptCallable):
    move_time = _make_callable(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r"), F("not visited ?to")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(
                time=(move_time, ["?r", "?from", "?to"]),
                resulting_fluents={
                    F("free ?r"),
                    F("not at ?r ?from"),
                    F("at ?r ?to"),
                    F("visited ?to"),
                },
            ),
        ],
    )

class RealEnvironment(BaseEnvironment):
    def __init__(self, client):
        super().__init__()
        self._get_locations_service = roslibpy.Service(client, '/get_locations', 'GetLocations')
        self._move_robot_service = roslibpy.Service(client, '/move_robot', 'MoveRobot')
        self._get_distance_service = roslibpy.Service(client, '/get_distance', 'GetDistance')
        self._move_status_service = roslibpy.Service(client, '/get_move_status', 'MoveStatus')
        self._stop_robot_service = roslibpy.Service(client, '/stop_robot', 'StopRobot')
        self.locations = self._get_locations()

    def get_move_cost_fn(self):
        locations = set(self.locations) - {'r1_loc', 'r2_loc'}
        location_distances = {}
        for loc1, loc2 in itertools.combinations(locations, 2):
            distance = self._get_distance(loc1, loc2)
            location_distances[frozenset([loc1, loc2])] = distance

        def get_move_time(robot, loc_from, loc_to):
            if frozenset([loc_from, loc_to]) in location_distances:
                return location_distances[frozenset([loc_from, loc_to])]
            distance = self._get_distance(loc_from, loc_to)
            return distance
        return get_move_time

    def _get_distance(self, location1, location2):
        request = roslibpy.ServiceRequest({'location1': location1, 'location2': location2})
        result = self._get_distance_service.call(request)
        return result['distance']

    def _get_locations(self):
        request = roslibpy.ServiceRequest()
        result = self._get_locations_service.call(request)
        return result['locations']

    def move_robot(self, robot_name, location):
        print("Moving robot", robot_name, "to", location)
        request = roslibpy.ServiceRequest({'robot_name': robot_name, 'location': location})
        result = self._move_robot_service.call(request)
        return result['status']

    def get_move_status(self, robot_name):
        request = roslibpy.ServiceRequest({'robot_name': robot_name})
        result = self._move_status_service.call(request)
        return result['status']

    def stop_robot(self, robot_name):
        print("Stopping robot", robot_name)
        request = roslibpy.ServiceRequest({'robot_name': robot_name})
        result = self._stop_robot_service.call(request)
        return result['success']


class OngoingAction:
    def __init__(self, time, action, environment=None):
        self.time = time
        self.name = action.name
        self._start_time = time
        self._action = action
        self._upcoming_effects = sorted([
            (time + eff.time, eff) for eff in action.effects
        ], key=lambda el: el[0])
        self.environment = environment

    @property
    def time_to_next_event(self):
        if self._upcoming_effects:
            return self._upcoming_effects[0][0]
        else:
            return float('inf')

    @property
    def is_done(self):
        return not self.upcoming_effects

    @property
    def upcoming_effects(self):
        # Return remaining upcoming events
        return self._upcoming_effects

    def advance(self, time):
        # Update the internal time
        self.time = time
        # Pop and return all effects scheduled at or before the new time
        new_effects = [effect for effect in self._upcoming_effects
                       if effect[0] <= time + 1e-9]
        # Remove the new_effects from upcoming_effects (effects are sorted)
        self._upcoming_effects = self._upcoming_effects[len(new_effects):]
        return new_effects


class OngoingMoveAction(OngoingAction):
    def __init__(self, time, action, environment=None):
        super().__init__(time, action, environment)
        # Keep track of initial start and end locations
        _, self.robot, self.start, self.end = self.name.split()  # (e.g., move r1 locA locB)
        self._action_status = IDLE
        self._action_progress = 0.0
        self._action_progress_prev = 0.0
        self.move_called = False

    def advance(self, time):
        if not self.move_called:
            self.environment.move_robot(self.robot, self.end)
            self.move_called = True
        return super().advance(time)

    @property
    def move_complete(self):
        if self.move_called:
            action_status = self.environment.get_move_status(self.robot)
            if action_status == REACHED:
                return True
        return False

    def interrupt(self):
        if self.time <= self._start_time:
            return set()  # Cannot interrupt before start time

        # stop robot
        self.environment.stop_robot(self.robot)
        robot = self.robot
        old_target = self.end
        new_target = f"{robot}_loc"
        new_fluents = set()

        for _, eff in self._upcoming_effects:
            if eff.is_probabilistic:
                raise ValueError("Probabilistic effects cannot be interrupted.")
            for fluent in eff.resulting_fluents:
                new_fluents.add(
                    F(" ".join(
                        [fluent.name]
                      + [fa if fa != old_target else new_target for fa in fluent.args]),
                      negated=fluent.negated)
                )

        self._upcoming_effects = []
        return new_fluents


class PlanningLoop():
    def __init__(
            self,
            initial_state: State,
            objects_by_type: Dict[str, Set[str]],
            operators: List[Operator],
            environment: BaseEnvironment):
        self._state = initial_state
        self.objects_by_type = {k: set(v) for k, v in objects_by_type.items()}
        self.operators = operators
        self.ongoing_actions = []
        self.environment = environment

    @property
    def time(self):
        return self._state.time

    @property
    def state(self):
        """The state is the internal state with future effects added."""
        effects = []
        for act in self.ongoing_actions:
            effects += act.upcoming_effects
        self.ongoing_actions = [
            act for act in self.ongoing_actions
            if not act.is_done
        ]
        return State(
            self._state.time,
            self._state.fluents,
            sorted(self._state.upcoming_effects + effects,
                   key=lambda el: el[0])
        )

    def get_actions(self) -> List:
        """Instantiate an Operator under the *current* objects_by_type."""
        objects_with_rloc = {k: set(v)
                             for k, v in self.objects_by_type.items()}
        objects_with_rloc["location"] |= set(
            f"{rob}_loc"
            for rob in self.objects_by_type["robot"]
            if F(f"at {rob} {rob}_loc") in self._state.fluents
        )
        return list(itertools.chain.from_iterable(
            operator.instantiate(objects_with_rloc)
            for operator in self.operators
        ))

    def advance(self, action):
        action_name = action.name.split()[0]
        if action_name == "move":
            new_act = OngoingMoveAction(self._state.time, action, self.environment)

        self.ongoing_actions.append(new_act)

        def _any_free_robots(state):
            return any(f.name == "free" for f in state.fluents)

        robot_free = False
        while not robot_free and self.ongoing_actions:
            # if every action is done, break
            if all(act.is_done for act in self.ongoing_actions):
                break

            adv_time = self.get_advance_time()

            new_effects = list(itertools.chain.from_iterable(
                [act.advance(adv_time) for act in self.ongoing_actions])
            )
            new_state = State(adv_time,
                              self._state.fluents,
                              sorted(self._state.upcoming_effects + new_effects,
                                     key=lambda el: el[0])
                              )
            # Add new effects to state
            self._state = self._get_new_state_by_intersection(transition(new_state, None))

            # Remove any actions that are now done
            self.ongoing_actions = [act for act in self.ongoing_actions if not act.is_done]

            robot_free = _any_free_robots(self._state)

        # interrupt ongoing actions if needed
        for act in self.ongoing_actions:
            new_fluents = act.interrupt()
            self._state.update_fluents(new_fluents)

        self.ongoing_actions = [act for act in self.ongoing_actions if act.upcoming_effects]
        return self.state

    def _get_new_state_by_intersection(self, states_and_probs):
        '''Determine new state by intersecting outcomes'''
        states = [s for s, _ in states_and_probs]
        base = states[0]
        new_fluents = {fl for fl in base.fluents if all(fl in s.fluents for s in states[1:])}
        new_upcoming = [ue for ue in base.upcoming_effects if all(ue in s.upcoming_effects for s in states[1:])]

        new_state = State(states[0].time, new_fluents, new_upcoming)
        return new_state

    def get_advance_time(self):
        # if any robot is done moving, advancing to that time
        for act in self.ongoing_actions:
            if act.move_complete:
                return act.time_to_next_event
        return self.state.time


    def goal_reached(self, goal_fluents):
        if all(fluent in self.state.fluents for fluent in goal_fluents):
            return True
        return False


if __name__ == '__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    env = RealEnvironment(client)

    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": env.locations,
    }
    initial_state = State(
        time=0.0,
        fluents={
            F("at", "r1", "r1_loc"), F('visited', 'r1_loc'),
            F("at", "r2", "r2_loc"), F('visited', 'r2_loc'),
            F("free", "r1"),
            F("free", "r2"),
        },
    )

    move_op = construct_move_visited_operator(move_time=env.get_move_cost_fn())
    planning_loop = PlanningLoop(initial_state, objects_by_type, [move_op], env)
    goal_fluents = {F("visited roomA"), F("visited roomB"), F("visited roomC"), F("visited roomD")}

    actions_taken = []
    for _ in range(1000):
        all_actions = planning_loop.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(planning_loop.state, goal_fluents, max_iterations=20000, c=10)
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            print(action_name)
            planning_loop.advance(action)
            print(planning_loop.state.fluents)
            actions_taken.append(action_name)
        else:
            print("No action.")

        if planning_loop.goal_reached(goal_fluents):
            print("Goal reached!")
            break
    client.terminate()

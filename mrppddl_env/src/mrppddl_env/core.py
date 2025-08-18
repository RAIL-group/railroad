from mrppddl.core import State
from mrppddl.core import Fluent as F
import itertools

class Robot:
    def __init__(self, name, location, fluents=None, time=0.0):
        self.name = name
        self.location = location
        self.current_action = None
        if fluents:
            self.fluents = set(fluents)
        else:
            self.fluents = set([
                F(f"free {name}"),
                F(f"at {name} {location}"),
            ])
        self._state = State(time, fluents)

    def assign(self, action, time):
        self.current_action_assigned_time = self._state.time
        self.current_action = action
        """Assigning an action selects the corresponding skill and updates the
        current fluents and upcoming effects as necessary."""
        raise NotImplementedError()

    def get_when_free(self):
        """Advance the robot's state until it's free and return that time."""

    def get_state(self):
        """Based on the robot's internal fluents, the action it's selected, and
        the time, get the 'State'."""
        if not self.current_action:
            return State(self._state.time, self._state.fluents)
        raise NotImplementedError()

class Simulator:
    def __init__(self, robots, env_fluents, env_upcoming=[], start_time=0.0):
        self.robots = robots
        self.env_fluents = env_fluents
        self.env_upcoming = env_upcoming
        self.time = start_time

    def get_state(self):
        """Make the 'state' of the current state of the world."""
        robot_states = [robot.get_state() for robot in self.robots]
        print([self.env_fluents] + [s.fluents for s in robot_states])
        all_fluents = set().union(*(
            [self.env_fluents] +
            [s.fluents for s in robot_states]
        ))
        all_upcoming_effects = list(itertools.chain.from_iterable(
            [self.env_upcoming] +
            [s.upcoming_effects for s in robot_states]
        ))
        return State(time=self.time,
                     fluents=all_fluents,
                     upcoming_effects=all_upcoming_effects)

    def assign(self, action):
        """Take in an action and assign it to the corresponding robot(s)."""
        for robot in self.robots:
            if robot.name in action.name.split(" "):
                robot.assign(action)
        pass

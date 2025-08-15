import numpy as np
from mrppddl.core import State, Fluent, transition, get_next_actions, get_action_by_name
from environment import get_location_object_likelihood


get_action_details = lambda s: (s.split()[0], s.split()[1], s.split()[2], s.split()[3], s.split()[4])


def get_current_fluents(robots, found_objects=None):
    fluents = set()
    for r in robots:
        if r.targeted:
            fluents.add(~Fluent(f"free {r.name}"))
            fluents.add(~Fluent(f"at {r.name} {r.at}"))
        else:
            fluents.add(Fluent(f"free {r.name}"))
            fluents.add(Fluent(f"at {r.name} {r.at}"))

    for obj in found_objects or []:
        fluents.add(Fluent(f"found {obj}"))

    return fluents

class SymbolicToRealSimulator():
    def __init__(self, map, robots, goal_fluents):
        self.map = map
        self.robots = robots
        self.state = State(
            time=0,
            fluents=get_current_fluents(self.robots),
        )
        self.goal_fluents = goal_fluents
        self.object_of_interest = [f.args[0] for f in self.goal_fluents]
        self.locations = map.locations + [r.start for r in robots]

        self.symb_robot_to_robot = {r.name: r for r in robots}
        self.symb_loc_to_location_coords = {
            loc.name: (lambda l=loc: l.location)
            for loc in self.locations
        }
        self.location_symb_to_objects_symb = {
            loc.name: loc.objects for loc in self.locations
        }

        self.visited_locations = []


    def is_goal(self):
        return all(gf in self.state.fluents for gf in self.goal_fluents)

    def get_move_cost(self, robot, from_loc, to_loc):
        location_from = self.symb_loc_to_location_coords[from_loc]()
        location_to = self.symb_loc_to_location_coords[to_loc]()
        return np.linalg.norm(np.array(location_from) - np.array(location_to))

    def get_likelihood_of_object(self, robot, location, object):
        robot_locations = [r.start.name for r in self.robots]
        if location in robot_locations:
            return 0.0
        return get_location_object_likelihood(location, object)

    def execute_action(self, action):
        if not self.state.satisfies_precondition(action):
            raise ValueError("Precondition not satisfied")

        # Target robot towards action.
        _, robot_name, _, to_loc_name, _ = get_action_details(action.name)

        robot = self.symb_robot_to_robot[robot_name]
        target_coords = self.symb_loc_to_location_coords[to_loc_name]()
        robot.retarget(action, target_coords)

        # get new fluents after retargeting
        new_fluents = get_current_fluents(self.robots)
        self.update_state(delta_t=0, new_fluents=new_fluents)

        # one of the robots reaches the target pose (all robots have to stop)
        if not np.any([r.target_pose == None for r in self.robots]):
            # robot reaches target pose
            robot = min(self.robots, key=lambda r: r.distance_to_target)
            move_distance = robot.distance_to_target
            [r.move(move_distance) for r in self.robots]

            # add to visited locations so that it is not planned with again
            if robot.target not in [r.start.name for r in self.robots]:
                self.visited_locations.append(robot.target)

            # Get current fluents after moving and getting observations
            found_objects = self.location_symb_to_objects_symb[robot.target]
            current_fluents = get_current_fluents(self.robots, found_objects)

            self.update_state(delta_t=move_distance, new_fluents=current_fluents)


    def update_state(self, delta_t, new_fluents):
        new_state = self.state.copy()

        new_state.update_fluents(new_fluents)
        new_state.set_time(self.state.time + delta_t) # TODO: verify this

        self.state = new_state


class Robot:
    def __init__(self, name=None, start=None):
        self.name = name
        self.start = start
        self.pose = self.start.location
        self.targeted = False
        self.action = None
        self.at = f'{self.name}_start'
        self.target = None
        self.target_pose = None
        self.target_object = None
        self.net_motion = 0

    def retarget(self, action, target_pose):
        self.targeted = True
        self.target_pose = target_pose
        self.action = action
        _, _, _, self.target, self.target_object = get_action_details(action.name)
        self.distance_to_target = np.linalg.norm(np.array(self.pose) - np.array(self.target_pose))


    def move(self, distance):
        direction = np.array(self.target_pose) - np.array(self.pose)
        if not np.all(direction) == 0:
            self.pose = self.pose + distance * direction / np.linalg.norm(direction)
            self.net_motion += distance
        # After moving certain distance, robot needs to retargeted, and it's start location is the pose.
        self.targeted = False
        self.start.location = self.pose

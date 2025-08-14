import numpy as np
from mrppddl.core import State, Fluent, transition, get_next_actions, get_action_by_name
from environment import get_location_object_likelihood


get_action_details = lambda s: (s.split()[0], s.split()[1], s.split()[2], s.split()[3], s.split()[4])


class SymbolicToRealSimulator():
    def __init__(self, map, start, robots, initial_fluents, goal_fluents):
        self.map = map
        self.all_objects = self.map.objects_in_environment
        self.robots = robots
        self.locations = [start] + map.locations

        self.state = State(
            time=0,
            fluents=initial_fluents
        )
        self.goal_fluents = goal_fluents

        self.symbolic_to_robot = {r.name: r for r in robots}
        self.symbolic_to_location_coords = {loc.name: loc.location for loc in self.locations}
        self.location_coords_to_symbolic = {tuple(loc.location): loc.name for loc in self.locations}
        self.location_coords_to_objects = {tuple(loc.location): loc.objects for loc in self.locations}
        self.location_symb_to_objects_symb = {
            loc.name: loc.objects for loc in self.locations
        }

    def get_fluents(self):
        # fs = set()
        fs = []
        for r in self.robots:
            fs.append(Fluent("free", r.name))
        fs.append(Fluent("at"))

    def is_goal(self):
        return all(gf in self.state.fluents for gf in self.goal_fluents)

    def get_move_cost(self, robot, from_loc, to_loc):
        location_from = self.symbolic_to_location_coords[from_loc]
        location_to = self.symbolic_to_location_coords[to_loc]
        return np.linalg.norm(np.array(location_from) - np.array(location_to))

    def get_likelihood_of_object(self, robot, location, object):
        return get_location_object_likelihood(location, object)

    def execute_action(self, action):

        if not self.state.satisfies_precondition(action):
            raise ValueError("Precondition not satisfied")

        # Target robot towards action.
        action_type, robot_name, from_loc_name, to_loc_name, obj_name = get_action_details(action.name)
        # print(action_type, robot_name, from_loc_name, to_loc_name, obj_name)

        robot = self.symbolic_to_robot[robot_name]
        target_pose = self.symbolic_to_location_coords[to_loc_name]

        robot.retarget(action, target_pose)
        new_fluents = set()
        new_fluents.update(action.effects[0].resulting_fluents)
        print("Starting state fluents", self.state.fluents)
        print(f'New fluents: {new_fluents}')
        self.update_state(delta_t=0, new_fluents=new_fluents)
        print("Updated state fluents", self.state.fluents)

        if not np.any([r.target_pose == None for r in self.robots]):
            # robot reaches target pose
            print([r.target for r in self.robots])
            robot = min(self.robots, key=lambda r: r.distance_to_target)
            min_distance = robot.distance_to_target
            [r.move(min_distance) for r in self.robots]

            print("Robot poses after move:")
            print([r.pose for r in self.robots])

            for effect in robot.action.effects:
                if effect.is_probabilistic:
                    print("Probabilistic Effect:")
                    new_fluents = set()
                    new_fluents.update(effect.resulting_fluents)

                    idx = 0 if robot.target_object in self.location_symb_to_objects_symb[robot.target] else 1
                    found_fluents = {Fluent(f"found {item}") for item in self.location_symb_to_objects_symb[robot.target]}
                    new_fluents.update(found_fluents)
                    _, effects = effect.prob_effects[idx]
                    for e in effects:
                        new_fluents.update(e.resulting_fluents)

                    print("Starting state fluents", self.state.fluents)
                    print(f'New fluents: {new_fluents}')
                    self.update_state(delta_t=min_distance, new_fluents=new_fluents)
                    print("Updated state fluents", self.state.fluents)


    def update_state(self, delta_t, new_fluents):
        new_state = self.state.copy()

        new_state.update_fluents(new_fluents)
        new_state.set_time(self.state.time + delta_t) # TODO: verify this

        self.state = new_state





class Robot:
    def __init__(self, name=None, pose=None):
        self.name = name
        self.pose = pose
        self.prev_pose = None
        self.action = None
        self.target_pose = None
        self.target_object = None
        self.loc_to = None
        self.loc_from = None
        self.net_motion = 0

    def retarget(self, action, target):
        self.action = action
        _, r_name, loc_from, loc_to, target_object = get_action_details(action.name)
        assert r_name == self.name
        print(f"Robot: Retargeting {self.name} from {loc_from} to {loc_to} to find {target_object}")
        self.target_pose = target
        self.target_object = target_object
        self.target = loc_to
        self.start = loc_from
        self.distance_to_target = np.linalg.norm(np.array(self.pose) - np.array(self.target_pose))

    def move(self, distance):
        self.prev_pose = self.pose
        direction = np.array(self.target_pose) - np.array(self.pose)
        if not np.all(direction) == 0:
            self.pose = self.pose + distance * direction / np.linalg.norm(direction)
            self.net_motion += distance

    # def move(self, target_pose):
    #     self.prev_pose = self.pose
    #     self.pose = target_pose
    #     self.net_motion += np.linalg.norm(np.array(self.prev_pose) - np.array(self.pose))

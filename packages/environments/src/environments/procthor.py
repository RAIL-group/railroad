import numpy as np
import itertools
from procthor import ThorInterface
from . import utils
from .environments import BaseEnvironment, SimulatedRobot, SkillStatus

SKILLS_TIME = {
    'robot1': {
        'pick': 15,
        'place': 15,
        'search': 15,
        'no_op': 100},  # TODO: adjust aftering checking the distance costs
    'robot2': {
        'pick': 10,
        'place': 10,
        'search': 10,
        'no_op': 100}
}


class ProcTHOREnvironment(BaseEnvironment):
    def __init__(self, args, robot_locations):
        super().__init__()
        self.args = args
        self.thor_interface = ThorInterface(self.args)
        self.known_graph, self.grid, self.robot_pose, self.target_object_info = self.thor_interface.gen_map_and_poses()
        self.start_coords = (self.robot_pose.x, self.robot_pose.y)
        self.robots = {
            r_name: SimulatedRobot(name=r_name,
                                   pose=self.start_coords)
            for r_name in robot_locations.keys()
        }
        self.locations = self._get_location_to_coordinates_dict()
        self.all_objects = self._get_all_objects()
        self.target_object = f'{self.target_object_info['name']}_{self.target_object_info['idxs'][0]}'

        self.partial_graph = self.known_graph.get_object_free_graph()

        self.robot_skill_start_time_and_duration = {robot_name: (0, None) for robot_name in self.robots.keys()}
        self.min_time = None

    def _get_all_objects(self):
        all_objects = set()
        for container_idx in self.known_graph.container_indices:
            object_idxs = self.known_graph.get_adjacent_nodes_idx(container_idx, filter_by_type=3)
            all_objects.update({f'{self.known_graph.get_node_name_by_idx(obj_idx)}_{obj_idx}'
                                for obj_idx in object_idxs})
        return all_objects

    def get_objects_at_location(self, location):
        if location not in self.locations or location == 'start':
            raise ValueError(f"Location {location} is not valid.")
        location_node_idx = int(location.split('_')[1])

        # get names of objects already present in partial graph
        existing_object_idxs = self.partial_graph.get_adjacent_nodes_idx(location_node_idx, filter_by_type=3)
        object_names = set(self.partial_graph.nodes[idx]['object_name'] for idx in existing_object_idxs)

        # add ground truth objects at location from known graph
        object_idxs = self.known_graph.get_adjacent_nodes_idx(location_node_idx, filter_by_type=3)
        for obj_idx in object_idxs:
            obj_node = self.known_graph.nodes[obj_idx].copy()
            obj_name = f"{obj_node['name']}_{obj_idx}"
            obj_node['object_name'] = obj_name
            object_names.add(obj_name)
            new_obj_idx = self.partial_graph.add_node(obj_node)
            self.partial_graph.add_edge(location_node_idx, new_obj_idx)

        return {'object': object_names}

    def _get_location_to_coordinates_dict(self):
        loc_to_coords = {'start': self.start_coords}
        for container_idx in self.known_graph.container_indices:
            loc_name = self.known_graph.get_node_name_by_idx(container_idx)
            coords = self.known_graph.get_node_position_by_idx(container_idx)
            coords = coords if len(coords) == 3 else (coords[0], coords[1], 0)
            loc_to_coords[f"{loc_name}_{container_idx}"] = coords
        return loc_to_coords

    def _get_move_cost_fn(self):
        inter_container_distances = {}
        for cnt1_idx, cnt2_idx in itertools.combinations(self.known_graph.container_indices, 2):
            loc1 = f"{self.known_graph.get_node_name_by_idx(cnt1_idx)}_{cnt1_idx}"
            loc2 = f"{self.known_graph.get_node_name_by_idx(cnt2_idx)}_{cnt2_idx}"
            cnt1_id = self.known_graph.nodes[cnt1_idx]['id']
            cnt2_id = self.known_graph.nodes[cnt2_idx]['id']
            inter_container_distances[frozenset([loc1, loc2])] = self.thor_interface.known_cost[cnt1_id][cnt2_id]

        def move_cost_fn(robot, loc_from, loc_to):
            if frozenset([loc_from, loc_to]) in inter_container_distances:
                return inter_container_distances[frozenset([loc_from, loc_to])]
            coord_from = self.locations[loc_from]
            coord_to = self.locations[loc_to]
            cost = utils.get_cost_between_two_coords(self.grid, coord_from, coord_to)
            return cost
        return move_cost_fn

    def _get_intermediate_coordinates(self, time, coord_from, coord_to):
        _, path = utils.get_cost_between_two_coords(self.grid, coord_from, coord_to, return_path=True)
        return utils.get_coordinates_at_time(path, time)

    def remove_object_from_location(self, obj, location):
        location_node_idx = int(location.split('_')[1])
        objects_at_location_idxs = self.partial_graph.get_adjacent_nodes_idx(location_node_idx, filter_by_type=3)
        for obj_idx in objects_at_location_idxs:
            obj_name = self.partial_graph.nodes[obj_idx]['object_name']
            if obj_name == obj:
                self.partial_graph.delete_node(obj_idx)
                break

    def add_object_at_location(self, obj, location):
        location_node_idx = int(location.split('_')[1])
        obj_idx = int(obj.split('_')[1])
        obj_node = self.known_graph.nodes[obj_idx].copy()
        obj_node['object_name'] = obj
        new_obj_idx = self.partial_graph.add_node(obj_node)
        self.partial_graph.add_edge(location_node_idx, new_obj_idx)

    def execute_skill(self, robot_name, skill_name, *args, **kwargs):
        if skill_name == 'move':
            loc_from = args[0]
            loc_to = args[1]
            target_coords = self.locations[loc_to]
            self.robots[robot_name].move(target_coords)

            # Keep track of move start time and duration
            move_cost_fn = self._get_move_cost_fn()
            move_time = move_cost_fn(robot_name, loc_from, loc_to) / 1.0  # robot velocity = 1.0
            self.robot_skill_start_time_and_duration[robot_name] = (self.time, move_time)

        elif skill_name in ['pick', 'place', 'search', 'no_op']:
            getattr(self.robots[robot_name], skill_name)()

            # Keep track of skill start time and duration
            skill_time = self.get_skills_cost_fn(skill_name)(robot_name, *args, **kwargs)
            self.robot_skill_start_time_and_duration[robot_name] = (self.time, skill_time)
        else:
            raise ValueError(f"Skill '{skill_name}' not defined for robot '{robot_name}'.")

    def get_skills_cost_fn(self, skill_name: str):
        if skill_name == 'move':
            return self._get_move_cost_fn()
        else:
            def get_skill_time(robot_name, *args, **kwargs):
                return SKILLS_TIME[robot_name][skill_name]
            return get_skill_time

    def stop_robot(self, robot_name):
        # If the robot was moving, it's now at a new intermediate location
        robot = self.robots[robot_name]
        if robot.current_action_name == 'move':
            robot_pose = self._get_intermediate_coordinates(
                self.min_time, robot.pose, robot.target_pose)
            self.locations[f'{robot_name}_loc'] = robot_pose
            robot.pose = robot_pose

        self.robots[robot_name].stop()
        self.robot_skill_start_time_and_duration[robot_name] = (0, None)

    def get_robot_that_finishes_first_and_when(self):
        robots_progress = np.array(
            [self.time - start_time for start_time, _ in self.robot_skill_start_time_and_duration.values()])
        time_to_target = [(r_name, tc) for r_name, (_, tc) in self.robot_skill_start_time_and_duration.items()]

        remaining_times = [(r_name, t - p) for (r_name, t), p in zip(time_to_target, robots_progress)]
        _, min_time = min(remaining_times, key=lambda x: x[1])
        min_robots = [n for n, t in remaining_times if t == min_time]
        return min_robots, min_time

    def get_executed_skill_status(self, robot_name, action_name):
        if action_name not in ['move', 'pick', 'place', 'search', 'no_op']:
            print(f"Action: '{action_name}' not verified in Simulation!")

        # For simulation we do the following:
        # If all robots are not assigned, return IDLE
        # If some robots are assigned, but this robot is not the one finishing first, return RUNNING
        # If this robot is among the ones finishing first, return DONE
        all_robots_assigned = all(not r.is_free for r in self.robots.values())
        if not all_robots_assigned:
            return SkillStatus.IDLE
        min_robots, self.min_time = self.get_robot_that_finishes_first_and_when()
        if robot_name not in min_robots:
            return SkillStatus.RUNNING
        return SkillStatus.DONE

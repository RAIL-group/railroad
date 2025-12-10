from typing import Dict, List, Tuple, Callable, Union
import numpy as np

class BaseEnvironment:
    '''Abstract class for all environments.'''
    def __init__(self):
        pass

    def get_move_cost_fn(self) -> Callable[[str, str, str], float]:
        raise NotImplementedError()

    def get_intermediate_coordinates(self, time, loc_from, loc_to) -> Union[List, Tuple]:
        raise NotImplementedError()

    def get_objects_at_location(self, location) -> Dict[str, set]:
        '''This is supposed to be a perception method that updates _objects_at_locations. In simulators, we get this
        from ground truth. In real robots, this would be replaced by a perception module.'''
        raise NotImplementedError()

    def remove_object_from_location(self, obj, location):
        raise NotImplementedError()

    def add_object_at_location(self, obj, location):
        raise NotImplementedError()


class SimpleEnvironment(BaseEnvironment):
    """Simple household environment for testing multi-object manipulation."""

    def __init__(self, locations, objects_at_locations):
        super().__init__()
        self.locations = locations.copy()
        self._ground_truth = objects_at_locations
        self._objects_at_locations = {loc: {"object": set()} for loc in locations}

    def get_objects_at_location(self, location):
        """Return objects at a location (simulates perception)."""
        objects_found = self._ground_truth.get(location, {}).copy()
        # Update internal knowledge
        if "object" in objects_found:
            for obj in objects_found["object"]:
                self.add_object_at_location(obj, location)
        return objects_found

    def get_move_cost_fn(self):
        """Return a function that computes movement time between locations."""
        def get_move_time(robot, loc_from, loc_to):
            distance = np.linalg.norm(
                self.locations[loc_from] - self.locations[loc_to]
            )
            return distance  # 1 unit of distance = 1 second
        return get_move_time

    def get_intermediate_coordinates(self, time, loc_from, loc_to):
        """Compute intermediate position during movement (for visualization)."""
        coord_from = self.locations[loc_from]
        coord_to = self.locations[loc_to]
        dist = np.linalg.norm(coord_to - coord_from)
        if dist < 0.01:
            return coord_to
        elif time > dist:
            print("time dist", time, dist)
            return coord_to
        direction = (coord_to - coord_from) / dist
        new_coord = coord_from + direction * time
        return new_coord

    def remove_object_from_location(self, obj, location, object_type="object"):
        """Remove an object from a location (e.g., when picked up)."""
        self._objects_at_locations[location][object_type].discard(obj)
        # Also update ground truth
        if location in self._ground_truth and object_type in self._ground_truth[location]:
            self._ground_truth[location][object_type].discard(obj)

    def add_object_at_location(self, obj, location, object_type="object"):
        """Add an object to a location (e.g., when placed down)."""
        self._objects_at_locations[location][object_type].add(obj)
        # Also update ground truth
        if location not in self._ground_truth:
            self._ground_truth[location] = {}
        if object_type not in self._ground_truth[location]:
            self._ground_truth[location][object_type] = set()
        self._ground_truth[location][object_type].add(obj)

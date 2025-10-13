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

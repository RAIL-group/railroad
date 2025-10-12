

class BaseEnvironment:
    def __init__(self, locations):
        self.locations = locations

    def get_move_cost_fn(self):
        raise NotImplementedError()

    def get_intermediate_coordinates(self, time, loc_from, loc_to):
        raise NotImplementedError()

    def get_objects_at_location(self, location):
        '''This is supposed to be a perception method that updates _objects_at_locations. In simulators, we get this
        from ground truth. In real robots, this would be replaced by a perception module.'''
        raise NotImplementedError()

    def remove_object_from_location(self, obj, location):
        raise NotImplementedError()

    def add_object_at_location(self, obj, location):
        raise NotImplementedError()

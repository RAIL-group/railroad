import random
import numpy as np

GRID_SIZE = 500
LOCATIONS = ["kitchen", "desk", "shelf", "bedroom", "bathroom", "storage"]
OBJECTS = ["Knife", "Notebook", "Pen", "Lamp", "Clock", "Toolset", "Pillow"]

likelihoods = {
    "start": {"Knife": 0.0, "Notebook": 0.0, "Pen": 0.0, "Lamp": 0.0, "Clock": 0.0, "Toolset": 0.0, "Pillow": 0.0},
    "kitchen": {"Knife": 1.0, "Notebook": 0.0, "Pen": 0.0, "Lamp": 0.0, "Clock": 0.0, "Toolset": 0.0, "Pillow": 0.0},
    "desk": {"Knife": 0.0, "Notebook": 1.0, "Pen": 1.0, "Lamp": 0.0, "Clock": 0.0, "Toolset": 0.0, "Pillow": 0.0},
    "shelf": {"Knife": 0.0, "Notebook": 0.0, "Pen": 0.0, "Lamp": 0.0, "Clock": 1.0, "Toolset": 0.0, "Pillow": 0.0},
    "bedroom": {"Knife": 0.0, "Notebook": 0.0, "Pen": 0.0, "Lamp": 1.0, "Clock": 1.0, "Toolset": 0.0, "Pillow": 1.0},
    "bathroom": {"Knife": 0.0, "Notebook": 0.0, "Pen": 0.0, "Lamp": 0.0, "Clock": 0.0, "Toolset": 0.0, "Pillow": 0.0},
    "storage": {"Knife": 0.0, "Notebook": 0.0, "Pen": 0.0, "Lamp": 0.0, "Clock": 0.0, "Toolset": 1.0, "Pillow": 0.0}
}

def get_location_object_likelihood(location, object):
    return likelihoods[location][object]


class Map():
    def __init__(self, seed=1024):
        np.random.seed(seed)
        self.coords_locations, self.location_objects = self.generate_env()
        self.objects_in_environment = []
        for objs in self.location_objects.values():
            for obj in objs:
                if obj not in self.objects_in_environment:
                    self.objects_in_environment.append(obj)

    def generate_env(self):
        location_coords = {}
        location_objects = {loc: [] for loc in LOCATIONS}
        for loc in LOCATIONS:
            coords = [random.randint(0, GRID_SIZE)] * 2
            location_coords[loc] = coords
            # randomly sample objects in those locations
            for object in OBJECTS:
                ps = likelihoods[loc][object]
                if np.random.rand() < ps:
                    location_objects[loc].append(object)
        return location_coords, location_objects

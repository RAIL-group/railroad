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

class Location:
    def __init__(self, name=None, location=None, objects=[]):
        self.name = name
        self.location = location
        self.objects = objects

    def __repr__(self):
        return f"Location(name={self.name}, location={self.location}, objects={self.objects})"

class Map():
    def __init__(self, n=3, seed=1024):
        np.random.seed(seed)
        self.n = min(n, len(LOCATIONS))
        self.objects_in_environment = []
        self.locations = self.generate_env()

    def generate_env(self):
        locations = []
        for i, loc in enumerate(LOCATIONS):
            coords = [random.randint(0, GRID_SIZE)] * 2
            # randomly sample objects in those locations
            items = []
            for object in OBJECTS:
                ps = likelihoods[loc][object]
                if np.random.rand() < ps:
                    items.append(object)
                    self.objects_in_environment.append(object)
            locations.append(Location(loc, coords, set(items)))
            if i + 1 == self.n:
                break
        return locations

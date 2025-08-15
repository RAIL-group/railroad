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
    def __init__(self, n_robots=2, max_locations=3, seed=1024):
        np.random.seed(seed)
        self.n = n_robots
        self.max_locations = min(max_locations, len(LOCATIONS))
        self.objects_in_environment = []
        self.locations, self.robot_poses = self.generate_env()

    def generate_env(self):
        locations, robot_poses = [], []
        # Robot start locations
        for i in range(self.n):
            robot_poses.append(Location(f'r{i+1}_start', (0, 0), set()))

        for i, loc in enumerate(LOCATIONS):
            coords = [np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)]
            # randomly sample objects in those locations
            items = []
            for object in OBJECTS:
                ps = likelihoods[loc][object]
                if np.random.rand() < ps:
                    items.append(object)
                    self.objects_in_environment.append(object)
            locations.append(Location(loc, coords, set(items)))
            if i + 1 == self.max_locations:
                break

        return locations, robot_poses

    def plot_map(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        for loc in self.locations:
            plt.scatter(*loc.location, label=loc.name)
        plt.xlim(0, GRID_SIZE)
        plt.ylim(0, GRID_SIZE)
        plt.title("Environment Map")
        plt.legend()
        plt.savefig("environment_map.png")

from collections import Counter
from contextlib import nullcontext

from interruption.utilities import filter_procthor_scenes
from railroad.environment.procthor.scene import ProcTHORScene
from tqdm import tqdm


def procthor10k_exploration(
    num_rooms: set[int] | None,
    room_types: set[str] | None,
    locations: set[str] | None,
    objects: set[str] | None,
    k: int = 10
) -> None:
    """
    Basic dataset exploration function for the ProcTHOR-10k dataset.
    Outputs the number of seeds that match the provided filtering criteria,
    the counts of the inclusion of locations/objects in those scenes, and the
    top k locations and objects.
    """
    seeds = filter_procthor_scenes(num_rooms, room_types, locations, objects)

    # print out the number of seeds that match the filter
    print(f"Number of seeds that matched filter: {len(seeds)}")

    # compute the counts of objects/locations included in the scenes
    location_counts = Counter()
    object_counts = Counter()

    for seed in tqdm(seeds):
        scene = ProcTHORScene(seed)
        # accessing of a private member is necessary to not run out of memory 
        # if the scenes are not cached
        controller = scene._thor.controller if scene._thor.controller else nullcontext()
        with controller:
            locations = {get_generic_name(loc) for loc in scene.locations if loc != "start_loc"}
            objects = {get_generic_name(obj) for obj in scene.objects}

            # update counts
            for loc in locations:
                location_counts[loc]+=1
            for obj in objects:
                object_counts[obj]+=1

    # print out top-k objects/locations
    print(f"Top-{k} most common locations:")
    print(location_counts.most_common(k))
    print(f"Top-{k} most common objects:")
    print(object_counts.most_common(k))


def explore_seeds(seeds: set[int]) -> None:
    """
    Prints out the structure of a set of ProcTHOR-10k scenes at a symbolic level.
    """
    for seed in seeds:
        scene = ProcTHORScene(seed)
        # accessing of a private member is necessary to not run out of memory 
        # if the scenes are not cached
        controller = scene._thor.controller if scene._thor.controller else nullcontext()
        with controller:
            print(f"Seed {seed}:")
            print(f"Number of Locations: {len(scene.locations)}")
            print(f"Number of Objects: {len(scene.objects)}")
            print(f"Number of Unique Locations: {len({loc.split("_")[0] for loc in scene.locations})}")
            print(f"Number of Unique Objects: {len({obj.split("_")[0] for obj in scene.objects})}")
            print(f"Locations: {scene.locations}")
            print(f"Objects: {scene.objects}")
            print(f"Object Locations: {scene.object_locations}")



# helper functions
def get_generic_name(obj: str) -> str:
    """
    Custom helper function for getting the generic name of a ProcTHOR object/location
    with the railroad naming convention. (Ex. 'apple_6' -> 'apple')
    """
    return obj.split("_")[0]


def main():
    # dataset filtering parameters
    num_rooms = {2}
    room_types = {"Kitchen"}
    locations = None
    objects = None
    k = 10

    # procthor10k_exploration(num_rooms, room_types, locations, objects, k)

    # explore the first k 1-room kitchen scenes
    seeds = set(filter_procthor_scenes(num_rooms, room_types, locations, objects)[:k])
    explore_seeds(seeds)


if __name__ == "__main__":
    main()

"""
ProcTHOR-10k scene selection and remapping for the multi-scene experiments:
filtering scenes, picking exemplars, remapping a scene's objects and locations
onto a shared set, and loading a remapped run's scenes and task distribution.
"""
import copy
import json
import os
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Sequence

from shapely.geometry import Point, Polygon

from railroad.core import (
    Goal,
)
from railroad.environment.procthor.resources import REMAP_DIR_ENV_VAR, get_procthor_10k_dir
from railroad.environment.procthor.thor_interface import IGNORE_CONTAINERS
from railroad.environment.procthor.utils import get_generic_name


# Generic (lowercase, get_generic_name-style) AI2-THOR/iTHOR object types with
# the "pickupable" property, per iTHOR's published object-type documentation.
# NOTE: "pickupable" is only exposed at runtime via a live Controller's event
# metadata (e.g. thor_interface.py's obj["pickupable"]) - the static
# ProcTHOR-10k data.jsonl scene records don't carry it, so this list can't be
# derived from that file directly and isn't verified against a live
# simulator here. Treat it as a reasonable starting point and extend/correct
# it as needed.
PICKUPABLE_OBJECT_TYPES = frozenset({
    "alarmclock", "aluminumfoil", "apple", "baseballbat", "basketball",
    "book", "boots", "bottle", "bowl", "box", "bread", "butterknife",
    "candle", "cd", "cellphone", "cloth", "creditcard", "cup", "dishsponge",
    "dumbbell", "egg", "fork", "handtowel", "kettle",
    "keychain", "knife", "ladle", "laptop", "lettuce", "mug", "newspaper",
    "pan", "papertowelroll", "pen", "pencil", "peppershaker", "pillow",
    "plate", "plunger", "pot", "potato", "remotecontrol", "saltshaker",
    "scrubbrush", "soapbar", "soapbottle", "spatula", "spoon",
    "spraybottle", "statue", "teddybear", "tennisracket", "tissuebox",
    "toiletpaper", "tomato", "towel", "vase", "watch", "wateringcan",
    "winebottle",
})

# Which location/container categories are plausible in which ProcTHOR room
# type, derived empirically from ProcTHOR-10k itself (via
# assign_containers_to_rooms below) rather than curated by hand: for each
# roomType, the fraction of that room type's scenes containing a given
# container category, kept when that fraction is >= 10%.
# The 10% cutoff was picked by eyeballing each room type's frequency
# distribution - every room type has a clear gap around there separating
# common furnishings from noise (e.g. Bedroom: tvstand at 20.5% vs. the next
# category, sofa, at 4.0%).
# IGNORE_CONTAINERS categories (painting, houseplant, box, baseballbat,
# basketball, plunger, television, ...) are excluded from this derivation:
# ThorInterface._preprocess_containers strips them before env.scene.locations
# is ever built, regardless of what the raw ProcTHOR-10k JSON contains, so
# they'd never actually be usable planning locations even if included here.
ROOM_TYPE_COMPATIBLE_LOCATIONS: dict[str, set[str]] = {
    "Kitchen": {
        "chair", "countertop", "diningtable", "fridge", "garbagebag",
        "garbagecan", "shelvingunit", "stool",
    },
    "Bathroom": {
        "dresser", "garbagecan", "laundryhamper", "sidetable", "sink",
        "toilet",
    },
    "LivingRoom": {
        "armchair", "cart", "chair", "diningtable", "dogbed", "dresser",
        "garbagecan", "shelvingunit", "sidetable", "sofa", "tvstand",
    },
    "Bedroom": {
        "armchair", "bed", "chair", "desk", "diningtable", "dogbed",
        "dresser", "garbagecan", "sidetable", "tvstand",
    },
}


def _room_polygon(room: dict[str, Any]) -> Polygon:
    """
    Builds a shapely polygon from a room's floor boundary, projected onto the
    x/z floor plane (y is height in ProcTHOR's coordinate convention, and is
    constant across a floorPolygon's points).
    """
    return Polygon([(pt["x"], pt["z"]) for pt in room["floorPolygon"]])


def assign_containers_to_rooms(scene: dict[str, Any]) -> dict[str, str]:
    """
    Maps each top-level container's id to the roomType of the room its
    position falls inside.

    Uses covers() only when it cleanly resolves to a single room. When it
    resolves to zero rooms (floating-point boundary noise - e.g. a
    wall-mounted object positioned right at a room's boundary) OR more than
    one room (a genuinely borderline point near a shared wall, where
    different GEOS versions can disagree on which side of the line it's on),
    falls back to nearest-by-distance instead of trusting the boolean
    predicate - distance() is a continuous function of the coordinates, not
    a boundary-sensitive topological test, so it produces the same room
    assignment regardless of which GEOS version computed it (important since
    this runs on multiple systems that may resolve different shapely/GEOS
    versions).
    """
    rooms = [(room["roomType"], _room_polygon(room)) for room in scene["rooms"]]

    container_room_types = {}
    for container in scene["objects"]:
        point = Point(container["position"]["x"], container["position"]["z"])

        exact_matches = [room_type for room_type, polygon in rooms if polygon.covers(point)]
        if len(exact_matches) == 1:
            container_room_types[container["id"]] = exact_matches[0]
        else:
            nearest_room_type, _ = min(rooms, key=lambda rp: rp[1].distance(point))
            container_room_types[container["id"]] = nearest_room_type

    return container_room_types


def filter_procthor_scenes(**filter_criteria) -> dict[int, dict[str, Any]]:
    """
    Filters the scenes of the ProcTHOR-10k dataset based on:
    - Number of rooms in the scene
    - If 1 or more of the rooms in the scene have the desired roomType
    - Containing certain locations
    - Containing certain objects
    - Number of objects (of any kind, including non-pickupable ones)
    - Number of pickupable objects specifically (see PICKUPABLE_OBJECT_TYPES)
    - Number of locations (of any kind, including non-functional ones)
    - Number of valid (functional-receptacle) locations specifically (see
      IGNORE_CONTAINERS)
    """
    supported_filters = {
        "num_rooms", "room_types", "locations", "objects", "num_locations",
        "num_objects", "num_pickupable_objects", "num_valid_locations",
    }
    # verify that passed in filters are supported
    unsupported_filters = set(filter_criteria) - supported_filters
    assert not unsupported_filters, f"unsupported filter(s): {unsupported_filters}"

    # load procthor-10k dataset
    with open(get_procthor_10k_dir() / 'data.jsonl', 'r', encoding="utf-8") as f:
        json_list = list(f)

    filtered_scenes = {}

    for seed, scene_json in enumerate(json_list):
        scene = json.loads(scene_json)
        if _check_scene_match(scene, filter_criteria):
            filtered_scenes[seed] = scene
    return filtered_scenes

def _check_scene_match(scene: dict[str, Any], filters: dict[str, Any]) -> bool:
    rooms = scene["rooms"]
    containers = scene["objects"]

    # shared by both the membership check ("locations"/"objects") and the
    # count check ("num_locations"/"num_objects") for a category - computed
    # once per scene, and only when a filter actually needs it
    scene_locations = (
        {get_generic_name(container["id"]) for container in containers}
        if "locations" in filters or "num_locations" in filters else None
    )
    scene_objects = (
        {
            get_generic_name(child["id"])
            for container in containers
            for child in container.get("children", [])
        }
        if "objects" in filters or "num_objects" in filters else None
    )
    scene_pickupable_objects = (
        {
            get_generic_name(child["id"])
            for container in containers
            for child in container.get("children", [])
            if get_generic_name(child["id"]) in PICKUPABLE_OBJECT_TYPES
        }
        if "num_pickupable_objects" in filters else None
    )
    scene_valid_locations = (
        {
            get_generic_name(container["id"])
            for container in containers
            if get_generic_name(container["id"]) not in IGNORE_CONTAINERS
        }
        if "num_valid_locations" in filters else None
    )

    filter_functions = {
        "num_rooms": partial(_check_num_rooms, rooms),
        "room_types": partial(_check_scene_room_types, rooms),
        "locations": partial(_check_scene_locations, scene_locations),
        "objects": partial(_check_scene_objects, scene_objects),
        "num_locations": partial(_check_num_unique_locations, scene_locations),
        "num_objects": partial(_check_num_unique_objects, scene_objects),
        "num_pickupable_objects": partial(_check_num_unique_objects, scene_pickupable_objects),
        "num_valid_locations": partial(_check_num_unique_locations, scene_valid_locations),
    }

    for filter_type, criteria in filters.items():
        if not filter_functions[filter_type](criteria):
            return False
    return True

def _check_num_rooms(rooms: list[dict[str, Any]], num_rooms: set[int] | None) -> bool:
    """
    Helper function for checking if the number of rooms in a ProcTHOR scene
    matches the desired number of rooms.
    Returns True if the user doesn't specify the desired number of rooms or
    if a match is found. Otherwise, returns False.
    """
    return len(rooms) in num_rooms if num_rooms is not None else True

def _check_scene_room_types(rooms: list[dict[str, Any]], room_types: set[str] | None) -> bool:
    """
    Helper function for checking if 1 or more of the rooms in a ProcTHOR
    scene is of the desired type. (E.g., kitchen, bedroom, etc.)
    Returns True if the user doesn't specify the desired number of rooms or
    if a match is found. Otherwise, returns False.
    """
    if room_types is None:
        return True
    return bool([True for room in rooms if room["roomType"] in room_types])

def _check_scene_locations(scene_locations: set[str] | None, locations: set[str] | None) -> bool:
    """
    Helper function for checking if the ProcTHOR scene contains the
    desired locations. (E.g., countertop, fridge, etc.)
    Returns True if the user doesn't specify the desired number of rooms or
    if all locations are present. Otherwise, returns False.
    """
    if locations is None:
        return True
    assert scene_locations is not None
    return locations.issubset(scene_locations)

def _check_scene_objects(scene_objects: set[str] | None, objects: set[str] | None) -> bool:
    """
    Helper function for checking if the ProcTHOR scene contains the
    desired objects. (E.g., coffeemachine, egg, etc.)
    Returns True if the user doesn't specify the desired number of rooms or
    if all objects are present. Otherwise, returns False.
    """
    if objects is None:
        return True
    assert scene_objects is not None
    return objects.issubset(scene_objects)

def _check_num_unique_locations(scene_locations: set[str] | None, num_locations: int) -> bool:
    assert scene_locations is not None
    return len(scene_locations) >= num_locations

def _check_num_unique_objects(scene_objects: set[str] | None, num_objects: int) -> bool:
    assert scene_objects is not None
    return len(scene_objects) >= num_objects


def find_exemplar_scene(
    scenes: dict[int, dict[str, Any]],
    num_objects: int,
    num_locations: int,
) -> tuple[int, set[str], set[str]]:
    """
    Deterministically selects an exemplar scene from a candidate pool (e.g.
    the output of filter_procthor_scenes) to source a fixed target
    object/location vocabulary from, for use with
    remap_scene_objects_and_locations.

    Prefers a scene with EXACTLY num_objects distinct pickupable object
    categories and num_locations distinct location categories, to avoid an
    arbitrary truncation choice; falls back to every candidate scene if no
    exact match exists. Among the resulting pool, picks the lowest seed for
    reproducibility across runs, then (only relevant on the fallback path)
    truncates to exactly num_objects/num_locations categories in sorted
    order for a deterministic, well-defined selection.

    Returns (exemplar_seed, target_objects, target_locations).
    """
    exact_matches = []
    all_candidates = []

    for seed, scene in scenes.items():
        obj_categories = {
            get_generic_name(child["id"])
            for container in scene["objects"]
            for child in container.get("children", [])
            if get_generic_name(child["id"]) in PICKUPABLE_OBJECT_TYPES
        }
        loc_categories = {
            get_generic_name(container["id"]) for container in scene["objects"]
            if get_generic_name(container["id"]) not in IGNORE_CONTAINERS
        }
        all_candidates.append((seed, obj_categories, loc_categories))
        if len(obj_categories) == num_objects and len(loc_categories) == num_locations:
            exact_matches.append((seed, obj_categories, loc_categories))

    pool = exact_matches if exact_matches else all_candidates
    exemplar_seed, obj_categories, loc_categories = min(pool, key=lambda c: c[0])

    target_objects = set(sorted(obj_categories)[:num_objects])
    target_locations = set(sorted(loc_categories)[:num_locations])
    return exemplar_seed, target_objects, target_locations


class SceneRemappingError(Exception):
    """
    Raised when one or more scenes couldn't be fully remapped to the target
    object/location vocabulary - i.e., ran out of non-matching candidates
    before every missing target category could be filled in.

    `failures` maps scene seed -> {"objects": <unfilled categories>,
    "locations": <unfilled categories>}. `remapped_scenes` carries the full
    result dict (including scenes that succeeded), in case the successful
    ones are still usable rather than discarding the whole batch.
    """
    def __init__(
        self,
        failures: dict[int, dict[str, set[str]]],
        remapped_scenes: dict[int, dict[str, Any]],
    ):
        self.failures = failures
        self.remapped_scenes = remapped_scenes
        summary = "; ".join(
            f"scene {seed} (missing objects={fail['objects'] or None}, "
            f"missing locations={fail['locations'] or None})"
            for seed, fail in failures.items()
        )
        super().__init__(f"{len(failures)} scene(s) could not be fully remapped: {summary}")


def remap_scene_objects_and_locations(
    scenes: dict[int, dict[str, Any]],
    target_objects: set[str],
    target_locations: set[str],
    seed: int | None = None,
) -> dict[int, dict[str, Any]]:
    """
    For each scene, ensures every category in target_objects/target_locations
    is present by randomly remapping non-matching existing objects/locations
    onto whichever target categories the scene doesn't already have.
    Categories already present in a scene are left untouched. Does not
    mutate the input scenes.

    Location remapping prefers non-matching containers whose room type is
    compatible with the target category (see ROOM_TYPE_COMPATIBLE_LOCATIONS),
    falling back to any non-matching container if none are compatible.
    Object remapping has no such constraint - any non-matching child is an
    eligible candidate for any missing target object category.

    Raises SceneRemappingError if any scene didn't have enough non-matching
    objects/locations to cover every target category - all scenes are still
    processed first, so a single error reports every affected scene at once.
    """
    rng = random.Random(seed)
    remapped_scenes = {}
    failures = {}

    for scene_seed, scene in scenes.items():
        scene_copy = copy.deepcopy(scene)
        unfilled_locations = _remap_locations(scene_copy, target_locations, rng)
        unfilled_objects = _remap_objects(scene_copy, target_objects, rng)
        remapped_scenes[scene_seed] = scene_copy
        if unfilled_locations or unfilled_objects:
            failures[scene_seed] = {"objects": unfilled_objects, "locations": unfilled_locations}

    if failures:
        raise SceneRemappingError(failures, remapped_scenes)

    return remapped_scenes


def _spare_entities(entities: list[dict[str, Any]], target_set: set[str]) -> list[dict[str, Any]]:
    """
    Entities that can safely be relabeled: those not already a target
    category, plus all-but-one instance of each target category that has
    duplicates (only one instance is needed to satisfy that category).
    """
    by_category = defaultdict(list)
    for entity in entities:
        by_category[get_generic_name(entity["id"])].append(entity)
    spare = []
    for category, instances in by_category.items():
        spare.extend(instances[1:] if category in target_set else instances)
    return spare


def _remap_locations(
    scene: dict[str, Any], target_locations: set[str], rng: random.Random
) -> set[str]:
    """Mutates scene in place. Returns any target categories left unfilled."""
    existing = {get_generic_name(c["id"]) for c in scene["objects"]}
    # sorted, not list(): set iteration order varies with PYTHONHASHSEED, and
    # shuffling a differently-ordered list with the same seed gives a
    # different result - which made the remap differ between processes
    missing = sorted(target_locations - existing)
    rng.shuffle(missing)

    spare = _spare_entities(scene["objects"], target_locations)
    rng.shuffle(spare)
    room_types = assign_containers_to_rooms(scene)

    for idx, target_category in enumerate(missing):
        if not spare:
            return set(missing[idx:])
        compatible = [
            c for c in spare
            if target_category in ROOM_TYPE_COMPATIBLE_LOCATIONS.get(room_types[c["id"]], set())
        ]
        pool = compatible if compatible else spare
        chosen = rng.choice(pool)
        _relabel(chosen, target_category)
        spare.remove(chosen)
    return set()


def _remap_objects(
    scene: dict[str, Any], target_objects: set[str], rng: random.Random
) -> set[str]:
    """
    Mutates scene in place. Returns any target categories left unfilled.
    Children of IGNORE_CONTAINERS containers neither count as present nor get
    relabeled: the environment drops them along with their container.
    """
    all_children = [
        child for container in scene["objects"]
        if get_generic_name(container["id"]) not in IGNORE_CONTAINERS
        for child in container.get("children", [])
    ]
    existing = {get_generic_name(child["id"]) for child in all_children}
    missing = sorted(target_objects - existing)  # sorted: see _remap_locations
    rng.shuffle(missing)

    spare = _spare_entities(all_children, target_objects)
    rng.shuffle(spare)

    for child, target_category in zip(spare, missing):
        _relabel(child, target_category)
    return set(missing[len(spare):])


def _relabel(entity: dict[str, Any], target_category: str) -> None:
    """
    Renames entity's id to target_category, preserving every segment after
    the first '|' unchanged (room index, slot descriptor, instance index -
    see get_room_id/get_generic_name) so nothing that parses those segments
    breaks. Does not touch assetId/position/rotation - only the symbolic id.
    """
    _, *rest = entity["id"].split("|")
    entity["id"] = "|".join([target_category.capitalize(), *rest])


def _scene_categories(scene: dict[str, Any]) -> tuple[set[str], set[str]]:
    """
    (pickupable object categories, location categories) of a ProcTHOR scene
    JSON, as an environment built from it has them: objects inside an
    IGNORE_CONTAINERS container don't count, since it is dropped with them.
    """
    obj_categories = {
        get_generic_name(child["id"])
        for container in scene["objects"]
        if get_generic_name(container["id"]) not in IGNORE_CONTAINERS
        for child in container.get("children", [])
        if get_generic_name(child["id"]) in PICKUPABLE_OBJECT_TYPES
    }
    loc_categories = {
        get_generic_name(container["id"]) for container in scene["objects"]
        if get_generic_name(container["id"]) not in IGNORE_CONTAINERS
    }
    return obj_categories, loc_categories


def use_remapped_scenes(remap_hash: str, procthor_seed: int) -> Path:
    """
    Points REMAP_DIR_ENV_VAR at remapped_scenes/<remap_hash>/ for the rest of
    this process, so environments built for scenes in it load the remapped
    scene rather than the raw one. Returns the directory.

    Raises if procthor_seed isn't one of the directory's scenes: it would
    silently load the raw scene, whose objects and locations the remapped
    task distribution doesn't match. Environments must also be built with
    object_seed=None - ThorInterface bypasses remaps for an object seed.
    """
    remap_dir = get_procthor_10k_dir() / "remapped_scenes" / remap_hash
    seeds = sorted(int(path.stem.removeprefix("scene_")) for path in remap_dir.glob("scene_*.json"))
    if not seeds:
        raise FileNotFoundError(f"no scene_<seed>.json files in {remap_dir}")
    if procthor_seed not in seeds:
        raise ValueError(f"scene {procthor_seed} isn't in {remap_dir}; its scenes are {seeds}")
    os.environ[REMAP_DIR_ENV_VAR] = str(remap_dir)
    return remap_dir


def get_task_distribution_from_remap_dir(
    remap_dir: Path | str,
    scene_filter: dict[str, int | set[str]],
    num_rooms: set[int] | None,
    num_tasks: int | None = None,
) -> tuple[Sequence[Goal], list[float]]:
    """
    Rebuilds the task distribution a multi-scene data-generation run used, so
    experiments evaluate on the same tasks the model was trained on.

    Mirrors multiprocess_datagen._select_scenes: filters the dataset, takes the
    exemplar's object/location categories from find_exemplar_scene, and feeds
    them to get_alfred_task_distribution as main does. scene_filter and
    num_rooms are the run's own (constants.ONE_ROOM_FILTER / NUM_ROOMS_FILTER
    for the one-room run), including "num_scenes".

    remap_dir (remapped_scenes/<hash>/, holding scene_<seed>.json) is not the
    source of the categories - it can't be. The exemplar is the lowest seed of
    the whole filtered pool, which usually lies outside the first num_scenes
    scenes that get remapped, and remapping only *adds* target categories, so
    the remapped scenes keep extra ones and no longer expose the exemplar's
    exact vocabulary. It is instead a check that this filter is the one that
    produced the directory: the scene seeds must be exactly the first
    num_scenes of the filtered pool, and every scene must contain every
    category. A mismatch means the filter (or the directory) is wrong.

    num_tasks defaults to NUM_TASKS.
    """
    # deferred: constants and environments both import this module
    from .constants import NUM_TASKS
    from .environments import get_alfred_task_distribution

    remap_dir = Path(remap_dir)
    remapped_scenes = {
        int(path.stem.removeprefix("scene_")): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(remap_dir.glob("scene_*.json"))
    }
    if not remapped_scenes:
        raise FileNotFoundError(f"no scene_<seed>.json files in {remap_dir}")

    num_scenes = scene_filter["num_scenes"]
    filter_kwargs = {k: v for k, v in scene_filter.items() if k != "num_scenes"}
    filtered_scenes = filter_procthor_scenes(num_rooms=num_rooms, **filter_kwargs)

    assert (
        isinstance(scene_filter["num_pickupable_objects"], int) and
        isinstance(scene_filter["num_valid_locations"], int) and
        isinstance(num_scenes, int)
    )
    _, target_objects, target_locations = find_exemplar_scene(
        filtered_scenes,
        num_objects=scene_filter["num_pickupable_objects"],
        num_locations=scene_filter["num_valid_locations"],
    )

    expected_seeds = set(list(filtered_scenes)[:num_scenes])
    if set(remapped_scenes) != expected_seeds:
        raise ValueError(
            f"{remap_dir} holds scenes {sorted(remapped_scenes)}, but this filter selects "
            f"{sorted(expected_seeds)}: it isn't the filter that produced this directory"
        )
    for seed, scene in remapped_scenes.items():
        objects, locations = _scene_categories(scene)
        if not (target_objects <= objects and target_locations <= locations):
            raise ValueError(
                f"scene {seed} in {remap_dir} lacks categories of this filter's exemplar "
                f"(objects {sorted(target_objects - objects)}, "
                f"locations {sorted(target_locations - locations)}): "
                "it isn't the filter that produced this directory"
            )

    return get_alfred_task_distribution(
        target_objects,
        target_locations,
        size=NUM_TASKS if num_tasks is None else num_tasks,
        one_object_per_taskdist=True,
    )

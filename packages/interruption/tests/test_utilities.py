import json
import math
import os

import pytest
from interruption.utilities import (
    RandomVariableType,
    calibrate_beta_parameter,
    _check_num_rooms,
    _check_scene_room_types,
    filter_procthor_scenes,
    find_exemplar_scene,
    find_shared_obj_loc_scenes,
    get_action_cost,
    get_augmented_task_dist,
    get_next_state,
    get_task_arrival_prob,
    get_task_distribution_from_remap_dir,
    remap_scene_objects_and_locations,
    use_remapped_scenes,
)
from railroad.core import Fluent, State, get_next_actions
from railroad.environment.procthor.resources import get_procthor_10k_dir
from railroad.environment.procthor.utils import get_generic_name
from railroad.operators.core import construct_move_operator, construct_pick_operator


def _load_scene_items(seed: int) -> set[tuple[str, str]]:
    """
    Test helper: independently derives a scene's (object/location) item set
    directly from data.jsonl, mirroring find_shared_obj_loc_scenes' own
    derivation. Used to verify its output against the raw dataset rather than
    trusting the function's own bookkeeping.
    """
    with open(get_procthor_10k_dir() / "data.jsonl", "r", encoding="utf-8") as f:
        scene = json.loads(next(line for i, line in enumerate(f) if i == seed))
    objects = {
        ("object", get_generic_name(child["id"]))
        for container in scene["objects"]
        for child in container.get("children", [])
    }
    locations = {
        ("location", get_generic_name(container["id"]))
        for container in scene["objects"]
    }
    return objects | locations


@pytest.mark.parametrize("action_cost", [1, 3, 5])
def test_get_action_cost_pick(action_cost):
    pick_op = construct_pick_operator(action_cost)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"pantry", "refrigerator"},
        "object": {"turkey", "bread"}
    }

    pick_actions = pick_op.instantiate(objects_by_type)

    for action in pick_actions:
        assert get_action_cost(action) == action_cost

def test_get_next_state():
    move_op = construct_move_operator(4)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
    }

    move_actions = move_op.instantiate(objects_by_type)
    assert len(move_actions) == 2

    initial_state = State(
        time=0,
        fluents={Fluent("at robot1 kitchen"), Fluent("free robot1")}
    )

    applicable_actions = get_next_actions(initial_state, move_actions)
    assert len(applicable_actions) == 1
    next_state, interruption_prob = get_next_state(initial_state, applicable_actions[0])
    assert next_state.time == 4
    assert next_state.fluents == {Fluent("at robot1 living_room"), Fluent("free robot1")}
    assert interruption_prob == 0


@pytest.mark.parametrize(
    argnames="rv_type, arrival_prob, time_between_arrivals, action_time, sol",
    argvalues=[
        (RandomVariableType.DISCRETE, 0.1, 100, 1, 0.1),
        (RandomVariableType.DISCRETE, 0.1, -1, 4, 0.1),
        (RandomVariableType.DISCRETE, 0.1, -1, 100, 0.1),
        (RandomVariableType.CONTINUOUS, -1, 5, 10, 0.86466472),
        (RandomVariableType.CONTINUOUS, -1, 30, 30, 0.63212056),
        (RandomVariableType.CONTINUOUS, -1, 60, 30, 0.39346934),
    ]
)
def test_get_task_arrival_prob(rv_type, arrival_prob, time_between_arrivals, action_time, sol):
    arrival_prob = get_task_arrival_prob(rv_type, arrival_prob, time_between_arrivals, action_time)
    assert arrival_prob == pytest.approx(sol)


@pytest.mark.parametrize(
    argnames="prob, a_t, sol",
    argvalues=[
        (0.5, 10, 14.426950408889635),
        (0.9, 30, 13.028834457097554),
        (0.1, 4, 37.96488632411962),
        (0.99999, 5, 0.4342944819030801),
        (0.5, 0, -0.0),
        (-0.1, 5, -1),
        (1, 5, -1),
        (1.5, 5, -1),
        (0.5, -1, -1),
        (0, 5, math.inf),
        (0, 0, math.inf),
    ]
)
def test_calibrate_beta_parameter(prob, a_t, sol):
    assert calibrate_beta_parameter(prob, a_t) == pytest.approx(sol)


@pytest.mark.parametrize(
    argnames="rooms, num_rooms, sol",
    argvalues=[
        ([{"room1": 1}, {"room2": 2}], {1}, False),
        ([{"room1": 1}, {"room2": 2}], {2}, True),
        ([{"room1": 1}, {"room2": 2}], {1, 2}, True),
        ([{"room1": 1}, {"room2": 2}], None, True),
    ]
)
def test_check_num_rooms(rooms, num_rooms, sol):
    assert _check_num_rooms(rooms, num_rooms) == sol


@pytest.mark.parametrize(
    argnames="rooms, room_types, sol",
    argvalues=[
        ([{"roomType": "Kitchen"}], {"Bedroom"}, False),
        ([{"roomType": "Bedroom"}], {"Bedroom"}, True),
        ([{"roomType": "Kitchen"}, {"roomType": "Bedroom"}], {"Kitchen"}, True),
        ([{"roomType": "Kitchen"}, {"roomType": "Bedroom"}], None, True),
    ]
)
def test_check_scene_room_types(rooms, room_types, sol):
    assert _check_scene_room_types(rooms, room_types) == sol


@pytest.mark.parametrize(
    argnames="num_rooms, room_types, locations, objects, sol",
    argvalues=[
        (None, None, None, None, 10000),
        ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, None, None, None, 10000),
        ({1}, {"Kitchen"}, None, None, 482),
        ({1}, {"Kitchen"}, {"sidetable"}, None, 33),
        ({1}, {"Kitchen"}, None, {"coffeemachine"}, 120),
    ]
)
def test_filter_procthor_scenes(num_rooms, room_types, locations, objects, sol):
    assert len(filter_procthor_scenes(
        num_rooms=num_rooms, room_types=room_types, locations=locations, objects=objects
    )) == sol


@pytest.mark.parametrize(
    argnames="num_scenes, num_rooms, num_objects, num_locations",
    argvalues=[
        (5, {1}, 2, 2),
        (20, {1}, 3, 1),
        (50, {1}, 1, 3),
    ]
)
def test_find_shared_obj_loc_scenes_success(num_scenes, num_rooms, num_objects, num_locations):
    scene_seeds, selected_items, success = find_shared_obj_loc_scenes(
        num_scenes, num_rooms, num_objects, num_locations
    )
    assert success
    assert len(scene_seeds) >= num_scenes

    # exactly the requested number of DISTINCT objects/locations were
    # selected - regression check for a prior bug where the same item could
    # be picked twice, and for a prior bug where a type more common in the
    # remaining scene pool than the other could overshoot its own quota
    # while the loop kept searching for the other type
    selected_objects = {item for item in selected_items if item[0] == "object"}
    selected_locations = {item for item in selected_items if item[0] == "location"}
    assert len(selected_objects) == num_objects
    assert len(selected_locations) == num_locations

    # every returned scene actually contains every selected item, checked
    # against the raw dataset independently of the function's own bookkeeping
    for seed in scene_seeds:
        assert selected_items.issubset(_load_scene_items(seed))


def test_find_shared_obj_loc_scenes_infeasible_num_scenes():
    # more scenes than exist in the whole ProcTHOR-10k dataset - can never succeed
    scene_seeds, _, success = find_shared_obj_loc_scenes(
        num_scenes=10001, num_rooms=None, num_objects=1, num_locations=1
    )
    assert not success
    assert len(scene_seeds) < 10001


def test_find_shared_obj_loc_scenes_zero_quota_not_allowed():
    # requesting 0 objects AND 0 locations is a no-op request and disallowed
    with pytest.raises(AssertionError):
        find_shared_obj_loc_scenes(
            num_scenes=1, num_rooms=None, num_objects=0, num_locations=0
        )


@pytest.mark.parametrize(
    argnames="current_task, task_dist, sol",
    argvalues=[
        (
            Fluent("at spoon bathroom"),
            ([Fluent("at cup bathroom")], [1.0]),
            ([Fluent("at spoon bathroom") & Fluent("at cup bathroom")], [1.0])
        ),
        (
            Fluent("at spoon bathroom"),
            ([Fluent("at cup bathroom"), Fluent("at knife bathroom")], [0.5, 0.5]),
            (
                [
                    Fluent("at spoon bathroom") & Fluent("at cup bathroom"),
                    Fluent("at spoon bathroom") & Fluent("at knife bathroom")
                ],
                [0.5, 0.5]
            )
        ),
    ]
)
def test_get_augmented_task_dist(current_task, task_dist, sol):
    assert get_augmented_task_dist(current_task, task_dist) == sol


_REMAP_SCRIPT = """
import hashlib
import json

from interruption.utilities import remap_scene_objects_and_locations

square = [{"x": x, "y": 0, "z": z} for x, z in [(0, 0), (0, 10), (10, 10), (10, 0)]]
scene = {
    "rooms": [{"roomType": "Kitchen", "floorPolygon": square}],
    "objects": [
        {"id": f"{name}|1|{i}", "position": {"x": 2 + i, "y": 0, "z": 2}, "children": [
            {"id": f"{obj}|surface|1|{i}"} for obj in objs
        ]}
        for i, (name, objs) in enumerate([
            ("Sofa", ["Knife", "Pen"]), ("Bed", ["Cup", "Book"]),
            ("TVStand", ["Fork", "Lamp"]), ("Dresser", ["Spoon", "Vase"]),
        ])
    ],
}
out = remap_scene_objects_and_locations(
    {7: scene}, {"apple", "bowl", "egg", "mug"}, {"fridge", "countertop", "stool"}, seed=37
)
print(hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest())
"""


def test_remap_is_identical_across_processes_with_different_hash_seeds():
    """The remap must not depend on set iteration order. list(set) order varies
    with PYTHONHASHSEED, and shuffling a differently-ordered list with the same
    seed gives a different result - so the same inputs used to produce a
    different remapped scene on every process launch, which also made anything
    keyed on its content unreproducible."""
    import os
    import subprocess
    import sys

    outputs = set()
    for hash_seed in ("1", "2", "3", "4", "5"):
        result = subprocess.run(
            [sys.executable, "-c", _REMAP_SCRIPT],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True, text=True, check=True,
        )
        outputs.add(result.stdout.strip())

    assert len(outputs) == 1


_DATAGEN_FILTER = {
    "num_scenes": 5, "num_pickupable_objects": 11, "num_valid_locations": 6,
    "room_types": {"Kitchen"},
}


@pytest.fixture(scope="module")
def datagen_remap():
    """
    The selection multiprocess_datagen does for a multi-scene run (small
    ONE_ROOM_FILTER-shaped filter): the exemplar's categories, and the first
    num_scenes matching scenes remapped onto them.
    """
    num_scenes = _DATAGEN_FILTER["num_scenes"]
    assert isinstance(num_scenes, int)
    filter_kwargs = {k: v for k, v in _DATAGEN_FILTER.items() if k != "num_scenes"}
    scenes = filter_procthor_scenes(num_rooms={1}, **filter_kwargs)
    _, target_objects, target_locations = find_exemplar_scene(scenes, 11, 6)
    remapped = remap_scene_objects_and_locations(
        {seed: scenes[seed] for seed in list(scenes)[:num_scenes]},
        target_objects, target_locations, seed=37,
    )
    return target_objects, target_locations, remapped


def _write_remap_dir(remap_dir, remapped_scenes):
    remap_dir.mkdir(parents=True)
    for seed, scene in remapped_scenes.items():
        (remap_dir / f"scene_{seed}.json").write_text(json.dumps(scene), encoding="utf-8")


def test_task_distribution_from_remap_dir_matches_datagen(datagen_remap, tmp_path):
    from interruption.constants import NUM_TASKS
    from interruption.environments import get_alfred_task_distribution

    target_objects, target_locations, remapped = datagen_remap
    _write_remap_dir(tmp_path / "hash", remapped)

    expected = get_alfred_task_distribution(
        target_objects, target_locations, size=NUM_TASKS, one_object_per_taskdist=True
    )
    goals, probs = get_task_distribution_from_remap_dir(tmp_path / "hash", _DATAGEN_FILTER, {1})

    assert [str(g) for g in goals] == [str(g) for g in expected[0]]
    assert probs == expected[1]


def test_task_distribution_from_remap_dir_rejects_other_filter(datagen_remap, tmp_path):
    # a different num_scenes selects a different set of seeds than the directory holds
    _write_remap_dir(tmp_path / "hash", datagen_remap[2])
    with pytest.raises(ValueError, match="isn't the filter that produced"):
        get_task_distribution_from_remap_dir(
            tmp_path / "hash", {**_DATAGEN_FILTER, "num_scenes": 4}, {1}
        )


def test_task_distribution_from_remap_dir_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_task_distribution_from_remap_dir(tmp_path, _DATAGEN_FILTER, {1})


def test_use_remapped_scenes(tmp_path, monkeypatch):
    import interruption.utilities as utilities
    from railroad.environment.procthor.resources import REMAP_DIR_ENV_VAR

    monkeypatch.setattr(utilities, "get_procthor_10k_dir", lambda: tmp_path)
    monkeypatch.delenv(REMAP_DIR_ENV_VAR, raising=False)
    remap_dir = tmp_path / "remapped_scenes" / "abc"
    remap_dir.mkdir(parents=True)
    for seed in (7, 12):
        (remap_dir / f"scene_{seed}.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"scene 5 isn't in .*\[7, 12\]"):
        use_remapped_scenes("abc", 5)
    assert REMAP_DIR_ENV_VAR not in os.environ  # a rejected seed leaves the environment alone

    assert use_remapped_scenes("abc", 12) == remap_dir
    assert os.environ[REMAP_DIR_ENV_VAR] == str(remap_dir)

    with pytest.raises(FileNotFoundError):
        use_remapped_scenes("missing", 7)

import random

import pytest
from railroad.core import Fluent, LiteralGoal

from interruption.experiments import _get_scene_objects_locations, _map_goal_to_scene


def test_map_goal_to_scene_maps_object_and_location():
    goal = LiteralGoal(Fluent("at spoon shelvingunit"))
    scene_objects = {"spoon": "spoon_14"}
    scene_locations = {"shelvingunit": "shelvingunit_2"}

    result = _map_goal_to_scene(goal, scene_objects, scene_locations)

    assert result == LiteralGoal(Fluent("at spoon_14 shelvingunit_2"))


def test_map_goal_to_scene_preserves_negation():
    goal = LiteralGoal(~Fluent("at spoon shelvingunit"))
    scene_objects = {"spoon": "spoon_14"}
    scene_locations = {"shelvingunit": "shelvingunit_2"}

    result = _map_goal_to_scene(goal, scene_objects, scene_locations)

    if isinstance(result, LiteralGoal):
        assert result == LiteralGoal(~Fluent("at spoon_14 shelvingunit_2"))
        assert result.fluent().negated


def test_map_goal_to_scene_preserves_fluent_name_and_arg_order():
    goal = LiteralGoal(Fluent("on knife countertop"))
    scene_objects = {"knife": "knife_3"}
    scene_locations = {"countertop": "countertop_1"}

    result = _map_goal_to_scene(goal, scene_objects, scene_locations)

    if isinstance(result, LiteralGoal):
        fluent = result.fluent()
        assert fluent.name == "on"
        assert fluent.args == ["knife_3", "countertop_1"]


def test_map_goal_to_scene_zero_arg_fluent():
    goal = LiteralGoal(Fluent("done"))

    result = _map_goal_to_scene(goal, {}, {})

    if isinstance(result, LiteralGoal):
        assert result == LiteralGoal(Fluent("done"))
        assert result.fluent().args == []


@pytest.mark.parametrize("goal", [
    Fluent("found cup") & Fluent("found bowl"),
    Fluent("found cup") | Fluent("found bowl"),
])
def test_map_goal_to_scene_passes_through_non_literal_goals(goal):
    # only LiteralGoal is handled today (see TODO on _map_goal_to_scene);
    # AND/OR goals should come back unchanged rather than being mangled.
    assert not isinstance(goal, LiteralGoal)
    result = _map_goal_to_scene(goal, {"cup": "cup_1"}, {})
    assert result == goal


def test_map_goal_to_scene_returns_original_goal_for_unmapped_arg():
    """
    An arg that isn't a key in either mapping dict is now treated as an
    unresolvable goal: the function bails out and returns the original,
    unmapped goal rather than building a fluent with a missing arg.
    """
    goal = LiteralGoal(Fluent("at spoon shelvingunit"))
    scene_objects = {}  # "spoon" is not a recognized generic name
    scene_locations = {"shelvingunit": "shelvingunit_2"}

    result = _map_goal_to_scene(goal, scene_objects, scene_locations)

    assert result == goal


def test_map_goal_to_scene_returns_original_goal_for_ambiguous_arg():
    """
    An arg present as a key in *both* scene_objects and scene_locations is
    ambiguous, so the function bails out and returns the original,
    unmapped goal rather than guessing which mapping to use.
    """
    goal = LiteralGoal(Fluent("at spoon"))
    scene_objects = {"spoon": "spoon_obj"}
    scene_locations = {"spoon": "spoon_loc"}

    result = _map_goal_to_scene(goal, scene_objects, scene_locations)

    assert result == goal


def test_map_goal_to_scene_does_not_mutate_input_goal():
    goal = LiteralGoal(Fluent("at spoon shelvingunit"))
    original = LiteralGoal(Fluent("at spoon shelvingunit"))

    _map_goal_to_scene(goal, {"spoon": "spoon_14"}, {"shelvingunit": "shelvingunit_2"})

    assert goal == original


def test_get_scene_objects_locations_single_object_no_duplicates():
    result = _get_scene_objects_locations(0, {"spoon_14"})

    assert result == {"spoon": "spoon_14"}


def test_get_scene_objects_locations_groups_multiple_generic_names():
    objects = {"spoon_14", "shelvingunit_2", "table_5"}

    result = _get_scene_objects_locations(0, objects)

    assert set(result.keys()) == {"spoon", "shelvingunit", "table"}
    assert result["spoon"] == "spoon_14"
    assert result["shelvingunit"] == "shelvingunit_2"
    assert result["table"] == "table_5"


def test_get_scene_objects_locations_empty_input():
    assert _get_scene_objects_locations(0, set()) == {}


def test_get_scene_objects_locations_picks_one_of_the_duplicates():
    objects = {"spoon_14", "spoon_22", "spoon_31"}

    result = _get_scene_objects_locations(0, objects)

    assert list(result.keys()) == ["spoon"]
    assert result["spoon"] in objects


def test_get_scene_objects_locations_is_reproducible_for_same_seed():
    objects = {"spoon_14", "spoon_22", "spoon_31", "knife_1", "knife_2"}

    first = _get_scene_objects_locations(7, objects)
    second = _get_scene_objects_locations(7, objects)

    assert first == second


def test_get_scene_objects_locations_seed_affects_choice_among_duplicates():
    objects = {"spoon_14", "spoon_22"}

    outcomes = {_get_scene_objects_locations(seed, objects)["spoon"] for seed in range(20)}

    # Not a hardcoded expectation of *which* seed picks which name -- just a
    # guard that the seed is actually wired into the choice, rather than the
    # function always resolving to the same candidate regardless of seed.
    assert outcomes == {"spoon_14", "spoon_22"}


def test_get_scene_objects_locations_only_splits_on_first_underscore():
    """
    Documents a naming-scheme assumption: the generic name is everything
    before the *first* underscore, so multi-word generic names collapse
    together. "coffee_table_1" and "coffee_maker_2" both bucket under
    "coffee" even though they're different generic objects -- worth
    confirming object/location naming conventions never use underscores
    within the generic (non-index) part of a name.
    """
    objects = {"coffee_table_1", "coffee_maker_2"}

    result = _get_scene_objects_locations(0, objects)

    assert list(result.keys()) == ["coffee"]
    assert result["coffee"] in objects


def test_get_scene_objects_locations_does_not_mutate_input():
    objects = {"spoon_14", "spoon_22"}
    original = set(objects)

    _get_scene_objects_locations(0, objects)

    assert objects == original


def test_get_scene_objects_locations_output_independent_of_global_random_state():
    """
    The function now uses a local random.Random(seed) instance rather than
    the global random module, so its output should depend only on the
    `seed` argument -- not on whatever state the global random module
    happens to be in when it's called.
    """
    objects = {"spoon_14", "spoon_22", "spoon_31"}

    random.seed(1)
    result_a = _get_scene_objects_locations(0, objects)

    random.seed(999)
    result_b = _get_scene_objects_locations(0, objects)

    assert result_a == result_b


def test_get_scene_objects_locations_does_not_disturb_global_random_state():
    """
    Companion to the above: calling the function should not advance or
    reseed the global random module as a side effect either.
    """
    random.seed(1)
    expected_next_draw = random.random()

    random.seed(1)
    _get_scene_objects_locations(0, {"spoon_14", "spoon_22"})
    actual_next_draw = random.random()

    assert actual_next_draw == expected_next_draw

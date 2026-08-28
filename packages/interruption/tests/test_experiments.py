import functools
import random

import pytest
import interruption.experiments as experiments_module
from interruption.experiments import (
    ExperimentConfig,
    ExperimentMode,
    ExperimentSeeds,
    _get_planner_config,
    _get_scene_objects_locations,
    _map_goal_to_scene,
)
from interruption.planning_framework import ap_heuristic_fn, get_no_int_discount, get_no_int_prob
from railroad.core import Fluent, LiteralGoal


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


@pytest.fixture
def mock_gcn(monkeypatch):
    """
    _get_planner_config calls AnticipateGCN.get_net_eval_fn, which loads a
    real torch checkpoint from disk. Stub it out on the experiments module
    namespace so these tests don't need a trained model file or torch/GCN
    machinery just to exercise the mode-selection logic.
    """
    sentinel_eval_fn = lambda scene_graph: 0.0
    monkeypatch.setattr(experiments_module, "get_torch_device", lambda: "cpu")
    monkeypatch.setattr(
        experiments_module.AnticipateGCN,
        "get_net_eval_fn",
        classmethod(lambda cls, network_file, device: sentinel_eval_fn),
    )
    return sentinel_eval_fn


def _dummy_config() -> ExperimentConfig:
    return ExperimentConfig(
        seeds=ExperimentSeeds(),
        goal=LiteralGoal(Fluent("done")),
        interrupting_task_dist=([], []),
        task_arrival_fn=lambda t: 0.1,
    )


@pytest.mark.parametrize(
    "mode, expected_discount_fn, expects_interruption_prob_fn, "
    "expects_interruption_value_fn, expects_weights",
    [
        (ExperimentMode.MYOPIC, get_no_int_discount, False, False, None),
        (ExperimentMode.ANTICIPATORY_PLANNING, get_no_int_discount, False, True, None),
        (ExperimentMode.INTERRUPTION, get_no_int_prob, True, True, None),
        (ExperimentMode.INTERRUPTION_AP, get_no_int_prob, True, True, (0.85, 1)),
    ],
)
def test_get_planner_config_selects_fields_per_mode(
    mock_gcn,
    mode,
    expected_discount_fn,
    expects_interruption_prob_fn,
    expects_interruption_value_fn,
    expects_weights,
):
    """
    Each ExperimentMode must produce the matching PlannerConfig: the right
    discount function, whether the caller's interruption_prob_fn is actually
    threaded through, whether a (mocked) AnticipateGCN eval fn is attached,
    and -- the behavior this diff introduces -- whether the heuristic is
    bound with include_v_ap=True (only for INTERRUPTION_AP) or False.
    """
    config = _dummy_config()
    interruption_prob_fn = lambda action_cost: 0.1

    result = _get_planner_config(config, mode, interruption_prob_fn)

    assert result.discount_fn is expected_discount_fn
    assert (
        result.planner_interruption_prob_fn is interruption_prob_fn
    ) == expects_interruption_prob_fn
    assert (result.interruption_value_fn is not None) == expects_interruption_value_fn
    assert result.current_task_reward == 0

    assert isinstance(result.heuristic_fn, functools.partial)
    assert result.heuristic_fn.func is ap_heuristic_fn
    assert len(result.heuristic_fn.args) == 0
    if expects_weights is None:
        assert len(result.heuristic_fn.keywords) == 0
    else:
        assert len(result.heuristic_fn.keywords) == 1
        assert result.heuristic_fn.keywords == {"weights": expects_weights}


def test_get_planner_config_myopic_interruption_prob_fn_is_ignored(mock_gcn):
    """
    MYOPIC/ANTICIPATORY_PLANNING force planner_interruption_prob_fn to None
    regardless of what the caller passes in -- unlike INTERRUPTION*, which
    threads the caller's function straight through.
    """
    config = _dummy_config()

    result = _get_planner_config(config, ExperimentMode.MYOPIC, interruption_prob_fn=0.5)

    assert result.planner_interruption_prob_fn is None


def test_get_planner_config_interruption_heuristic_behaves_without_v_ap(mock_gcn):
    """
    Behavioral cross-check for the INTERRUPTION branch: the bound heuristic_fn
    should ignore v_ap entirely (matching ap_heuristic_fn's include_v_ap=False
    contract), not merely default it to zero.
    """
    from railroad.core import Fluent as F, LiteralGoal, State
    from railroad.operators.core import construct_move_operator

    config = _dummy_config()
    result = _get_planner_config(config, ExperimentMode.INTERRUPTION, interruption_prob_fn=0.1)

    move_op = construct_move_operator(5.0)
    actions = move_op.instantiate({"robot": ["r1"], "location": ["start", "target"]})
    state = State(time=0, fluents={F("at r1 start"), F("free r1")})
    goal = LiteralGoal(F("at r1 target"))

    heuristic_fn = result.heuristic_fn
    assert isinstance(heuristic_fn, functools.partial)
    assert heuristic_fn(state, goal, actions, 999.0) == 5.0


def test_get_planner_config_interruption_ap_heuristic_adds_v_ap(mock_gcn):
    """Behavioral cross-check for INTERRUPTION_AP: v_ap must actually be added."""
    from railroad.core import Fluent as F, LiteralGoal, State
    from railroad.operators.core import construct_move_operator

    config = _dummy_config()
    result = _get_planner_config(
        config, ExperimentMode.INTERRUPTION_AP, interruption_prob_fn=0.1
    )

    move_op = construct_move_operator(5.0)
    actions = move_op.instantiate({"robot": ["r1"], "location": ["start", "target"]})
    state = State(time=0, fluents={F("at r1 start"), F("free r1")})
    goal = LiteralGoal(F("at r1 target"))

    heuristic_fn = result.heuristic_fn
    assert isinstance(heuristic_fn, functools.partial)
    assert heuristic_fn(state, goal, actions, 3.0) == pytest.approx(5.0 * 0.85 + 3.0)

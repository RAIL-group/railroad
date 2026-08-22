"""Tests for ObjectSearchEnvironment: railroad's object-search conventions.

Generic SymbolicEnvironment behavior is covered in
test_symbolic_environment.py; everything here exercises the layered
object-search semantics: ground-truth search resolution, revelation,
robot intermediate locations, and the move/place/search action filters.
"""

import pytest

from railroad._bindings import Fluent as F, GroundedEffect, State
from railroad.core import Effect, Operator
from railroad.environment import ObjectSearchEnvironment

from env_helpers import env_with_operators


# =============================================================================
# Robot Intermediate Locations (robot_loc)
# =============================================================================


def test_interrupt_then_move_to_different_destination():
    """Test that after interruption, robot can move to a new destination with correct cost.

    Scenario:
    - Robot starts at kitchen (0,0) moving to bedroom (10,0)
    - Gets interrupted at 50% -> ends up at (5,0)
    - Then moves to living_room (10,5)
    - Expected cost: sqrt((10-5)^2 + (5-0)^2) = sqrt(50) ≈ 7.07

    Requires ObjectSearchEnvironment: replanning from the intermediate
    position needs the `robot1_loc` marker added to the grounding universe.
    """
    import math
    import numpy as np
    from railroad.environment import InterruptibleNavigationMoveSkill, LocationRegistry
    from railroad.core import get_action_by_name

    # Create registry with locations
    locations = {
        "kitchen": np.array([0.0, 0.0]),
        "bedroom": np.array([10.0, 0.0]),
        "living_room": np.array([10.0, 5.0]),
    }
    registry = LocationRegistry(locations)
    move_time = registry.move_time_fn(velocity=1.0)

    # Two robots: robot1 at kitchen (free), robot2 at bedroom (becomes free at t=5)
    initial_fluents = {
        F("at", "robot1", "kitchen"),
        F("at", "robot2", "bedroom"),
        F("free", "robot1"),
        # robot2 starts not free, becomes free at t=5 via initial effect
    }
    # Initial effect to make robot2 free at t=5, triggering interrupt
    initial_effects = [
        (5.0, GroundedEffect(5.0, {F("free", "robot2")})),
    ]

    # Move operator with dynamic time based on distance
    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(
                time=(move_time, ["?robot", "?from", "?to"]),
                resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to"), F("free", "?robot")},
            ),
        ]
    )

    # Wait operator for robot2 to advance time
    wait_op = Operator(
        name="wait",
        parameters=[("?robot", "robot")],
        preconditions=[F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=10.0, resulting_fluents={F("free", "?robot")}),
        ]
    )

    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, initial_fluents, initial_effects),
        objects_by_type={"robot": {"robot1", "robot2"}, "location": {"kitchen", "bedroom", "living_room"}},
        operators=[move_op, wait_op],
        skill_overrides={"move": InterruptibleNavigationMoveSkill},
        location_registry=registry,
    )

    # Robot1 starts moving from kitchen (0,0) to bedroom (10,0) - takes 10s
    # At t=5, robot2 becomes free (initial effect), interrupting robot1
    actions = env.get_actions()
    move_action = get_action_by_name(actions, "move robot1 kitchen bedroom")
    env.act(move_action)

    # At t=5, robot1 should be at intermediate location
    assert env.time == pytest.approx(5.0, abs=0.1)
    assert F("at", "robot1", "robot1_loc") in env.state.fluents
    assert F("free", "robot1") in env.state.fluents

    # Verify intermediate coordinates are correct (50% of way from kitchen to bedroom)
    intermediate_pos = registry.get("robot1_loc")
    assert intermediate_pos is not None
    assert np.allclose(intermediate_pos, np.array([5.0, 0.0]))

    # Now robot1 moves from intermediate location (5,0) to living_room (10,5)
    actions = env.get_actions()
    move_to_living = get_action_by_name(actions, "move robot1 robot1_loc living_room")
    assert move_to_living is not None, "Move action from intermediate location should be available"

    time_before_move = env.time
    env.act(move_to_living)

    # Robot2 does a long wait to advance time past robot1's move (~7.07s)
    actions = env.get_actions()
    wait_action = get_action_by_name(actions, "wait robot2")
    env.act(wait_action)

    # Verify robot1 reached living_room
    assert F("at", "robot1", "living_room") in env.state.fluents
    assert F("free", "robot1") in env.state.fluents

    # Verify that robot2's wait action is not interrupted, so it's not free
    assert F("free robot2") not in env.state.fluents

    # Verify the move took the expected time: sqrt((10-5)^2 + (5-0)^2) = sqrt(50)
    expected_move_time = math.sqrt(50)  # ~7.07
    actual_move_time = env.time - time_before_move
    assert actual_move_time == pytest.approx(expected_move_time, abs=0.1)


# =============================================================================
# Action Filtering (move/place/search conventions)
# =============================================================================


def test_object_search_environment_filters_zero_duration_moves():
    """Zero-duration move actions are filtered by the domain conventions."""
    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to")}),
        ],
    )
    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, {F("at", "robot1", "kitchen"), F("free", "robot1")}, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
        operators=[move_op],
    )
    assert env.get_actions() == []


# =============================================================================
# Revelation Tests (Object Discovery)
# =============================================================================


def test_object_search_environment_revelation():
    """Test that searching a location reveals objects at that location."""
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}

    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, fluents, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen"}},
        operators=[],
        true_object_locations={"kitchen": {"Knife", "Fork"}},
    )

    # Simulate a search completing (searched fluent added)
    effect = GroundedEffect(
        time=0.0,
        resulting_fluents={F("searched", "kitchen")},
    )
    env.apply_effect(effect)

    # Verify objects were revealed
    assert F("revealed", "kitchen") in env.fluents
    assert F("found", "Knife") in env.fluents
    assert F("found", "Fork") in env.fluents
    assert F("at", "Knife", "kitchen") in env.fluents
    assert F("at", "Fork", "kitchen") in env.fluents


# =============================================================================
# Object Location Tracking Tests
# =============================================================================


def test_object_search_environment_objects_at_locations():
    """Test internal objects_at_locations tracking."""
    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
        true_object_locations={"kitchen": {"Knife", "Fork"}},
    )

    # Access internal state for verification
    assert env._objects_at_locations["kitchen"] == {"Knife", "Fork"}
    assert env._objects_at_locations.get("bedroom", set()) == set()


def test_object_search_environment_object_location_from_fluents():
    """Test that object locations are derived from fluents."""
    # Initial ground truth: Knife and Fork at kitchen
    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
        true_object_locations={"kitchen": {"Knife", "Fork"}},
    )

    # Before any fluents, objects are at initial locations
    assert env._is_object_at_location("Knife", "kitchen")
    assert env._is_object_at_location("Fork", "kitchen")

    # After adding "holding" fluent, object is no longer at location
    env._fluents.add(F("holding", "robot1", "Knife"))
    assert not env._is_object_at_location("Knife", "kitchen")
    assert env._is_object_at_location("Fork", "kitchen")  # Fork still there

    # After adding "at" fluent at different location, object is there
    env._fluents.discard(F("holding", "robot1", "Knife"))
    env._fluents.add(F("at", "Knife", "bedroom"))
    assert not env._is_object_at_location("Knife", "kitchen")
    assert env._is_object_at_location("Knife", "bedroom")


def test_object_search_environment_fluent_overrides_ground_truth():
    """Test that fluents override initial ground truth for object locations."""
    # Initial ground truth: Knife at kitchen
    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
        true_object_locations={"kitchen": {"Knife"}},
    )

    # Initial: Knife is at kitchen (from ground truth)
    assert env._is_object_at_location("Knife", "kitchen")
    assert not env._is_object_at_location("Knife", "bedroom")

    # Add fluent saying Knife is at bedroom - this should override ground truth
    env._fluents.add(F("at", "Knife", "bedroom"))
    assert not env._is_object_at_location("Knife", "kitchen")  # Fluent takes priority
    assert env._is_object_at_location("Knife", "bedroom")


# =============================================================================
# Probabilistic Effect Resolution Tests
# =============================================================================


def test_object_search_environment_resolve_probabilistic_effect():
    """Test resolving probabilistic effects based on ground truth."""
    # Create an environment where "obj" IS at "loc"
    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
        true_object_locations={"loc": {"obj"}},  # obj is at loc
    )

    # Create a deterministic effect
    det_effect = GroundedEffect(
        time=1.0,
        resulting_fluents={F("done")},
    )

    # For non-probabilistic, should return unchanged
    effects, fluents = env.resolve_probabilistic_effect(det_effect, set())
    assert effects == [det_effect]

    # Create a probabilistic effect with proper search structure
    # Success branch has both "found obj" and "at obj loc"
    branch1_effect = GroundedEffect(
        time=0.0,
        resulting_fluents={F("found", "obj"), F("at", "obj", "loc")}
    )
    branch2_effect = GroundedEffect(time=0.0, resulting_fluents={F("searched", "loc")})
    prob_effect = GroundedEffect(
        time=2.0,
        resulting_fluents=set(),
        prob_effects=[
            (0.6, [branch1_effect]),  # success branch
            (0.4, [branch2_effect]),  # failure branch
        ],
    )

    # Since obj IS at loc in ground truth, should return success branch
    effects, fluents = env.resolve_probabilistic_effect(prob_effect, set())
    assert len(effects) == 1
    assert effects[0] == branch1_effect

    # Now test when object is NOT at location
    env2 = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
        true_object_locations={"other_loc": {"obj"}},  # obj is at other_loc, not loc
    )
    effects2, _ = env2.resolve_probabilistic_effect(prob_effect, set())
    assert len(effects2) == 1
    assert effects2[0] == branch2_effect  # failure branch


# =============================================================================
# Search Action Integration Tests
# =============================================================================


def test_search_skill_resolves_probabilistically():
    """Test that search skill resolves probabilistic effects via environment."""
    from railroad.core import get_action_by_name
    from railroad import operators

    # Object IS at kitchen - search should succeed
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}

    search_op = operators.construct_search_operator(
        object_find_prob=0.5,  # Probability doesn't matter - ground truth does
        search_time=3.0,
    )

    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, fluents, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen"}, "object": {"Knife"}},
        operators=[search_op],
        true_object_locations={"kitchen": {"Knife"}},
    )
    actions = env.get_actions()
    search_action = get_action_by_name(actions, "search robot1 kitchen Knife")

    env.act(search_action)

    # Since Knife IS at kitchen, search should succeed
    assert F("searched", "kitchen", "Knife") in env.state.fluents
    assert F("found", "Knife") in env.state.fluents
    assert F("at", "Knife", "kitchen") in env.state.fluents
    assert F("free", "robot1") in env.state.fluents


def test_search_skill_fails_when_object_not_at_location():
    """Test that search fails when object is NOT at the searched location."""
    from railroad.core import get_action_by_name
    from railroad import operators

    # Object is NOT at kitchen (it's at bedroom)
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}

    search_op = operators.construct_search_operator(
        object_find_prob=0.9,  # High probability, but ground truth says object NOT here
        search_time=3.0,
    )

    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, fluents, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}, "object": {"Knife"}},
        operators=[search_op],
        true_object_locations={"bedroom": {"Knife"}},  # Knife is NOT at kitchen
    )
    actions = env.get_actions()
    search_action = get_action_by_name(actions, "search robot1 kitchen Knife")

    env.act(search_action)

    # Search should complete but NOT find the object
    assert F("searched", "kitchen", "Knife") in env.state.fluents
    assert F("found", "Knife") not in env.state.fluents
    assert F("free", "robot1") in env.state.fluents


# =============================================================================
# Pick/Place Action Integration Tests
# =============================================================================


def test_pick_skill_updates_fluents():
    """Test that pick skill updates fluents correctly."""
    from railroad.core import get_action_by_name
    from railroad import operators

    fluents = {
        F("at", "robot1", "kitchen"), F("free", "robot1"),
        F("at", "Knife", "kitchen"), F("found", "Knife"),
    }

    pick_op = operators.construct_pick_operator_blocking(pick_time=2.0)

    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": {"robot1"},
            "location": {"kitchen"},
            "object": {"Knife"},
        },
        operators=[pick_op],
        true_object_locations={"kitchen": {"Knife"}},
    )

    actions = env.get_actions()
    pick_action = get_action_by_name(actions, "pick robot1 kitchen Knife")

    env.act(pick_action)

    # Verify fluents are correct
    assert F("holding", "robot1", "Knife") in env.state.fluents
    assert F("at", "Knife", "kitchen") not in env.state.fluents
    # Object location is derived from fluents - holding means not at location
    assert not env._is_object_at_location("Knife", "kitchen")


def test_place_skill_updates_fluents():
    """Test that place skill updates fluents correctly."""
    from railroad.core import get_action_by_name
    from railroad import operators

    fluents = {
        F("at", "robot1", "bedroom"), F("free", "robot1"),
        F("holding", "robot1", "Knife"), F("hand-full", "robot1"),
    }

    place_op = operators.construct_place_operator_blocking(place_time=2.0)

    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": {"robot1"},
            "location": {"bedroom"},
            "object": {"Knife"},
        },
        operators=[place_op],
        true_object_locations={"bedroom": set()},
    )

    actions = env.get_actions()
    place_action = get_action_by_name(actions, "place robot1 bedroom Knife")

    env.act(place_action)

    # Verify fluents are correct
    assert F("at", "Knife", "bedroom") in env.state.fluents
    assert F("holding", "robot1", "Knife") not in env.state.fluents
    # Object location is derived from fluents
    assert env._is_object_at_location("Knife", "bedroom")


# =============================================================================
# Nested Effects Timing Tests (Issue #6)
# =============================================================================


def test_nested_effects_with_timing_are_scheduled():
    """Test that nested effects inside prob_effects with time > 0 are scheduled correctly.

    This is a regression test for issue #6: nested effects inside prob_effects
    were being applied immediately instead of being scheduled at their proper time.
    """
    from railroad import operators
    from railroad.core import get_action_by_name

    move_time = 5.0
    pick_time = 3.0

    # construct_search_and_pick_operator has nested effects with timing:
    # - Effect at move_time with prob_effects containing:
    #   - Effect(time=0, ...) - immediate
    #   - Effect(time=pick_time, ...) - should be delayed
    search_pick_op = operators.construct_search_and_pick_operator(
        object_find_prob=1.0,  # Always find - takes success branch
        move_time=move_time,
        pick_time=pick_time,
    )

    initial_fluents = {
        F("at", "robot1", "living_room"),
        F("free", "robot1"),
    }

    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, initial_fluents, []),
        objects_by_type={
            "robot": {"robot1"},
            "location": {"living_room", "kitchen"},
            "object": {"Knife"},
        },
        operators=[search_pick_op],
        true_object_locations={"kitchen": {"Knife"}},  # Object is actually there
    )

    actions = env.get_actions()
    action = get_action_by_name(actions, "search robot1 living_room kitchen Knife")

    env.act(action)

    # Key assertions - these failed before the fix:
    # 1. Robot should be holding the knife (from nested Effect(time=pick_time))
    assert F("holding", "robot1", "Knife") in env.state.fluents, \
        "Nested effect with timing was not applied - holding fluent missing"

    # 2. Time should be move_time + pick_time (not just move_time)
    expected_time = move_time + pick_time
    assert env.time == pytest.approx(expected_time, abs=0.1), \
        f"Time should be {expected_time} but was {env.time} - nested timing not respected"

    # 3. Robot should be free after picking (from the same nested effect)
    assert F("free", "robot1") in env.state.fluents

    # 4. Robot should be at kitchen
    assert F("at", "robot1", "kitchen") in env.state.fluents


def test_search_with_certainty_probability_object_not_present():
    """Test search with object_find_prob=1.0 when object is NOT at location.

    This is a regression test: when ground truth says object isn't there,
    the environment should deterministically return the failure branch,
    even when the failure branch has probability 0.0 (from 1.0 - 1.0).
    """
    from railroad.core import get_action_by_name
    from railroad import operators

    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}

    # Probability of 1.0 means failure branch has probability 0.0
    search_op = operators.construct_search_operator(
        object_find_prob=1.0,  # 100% find probability
        search_time=3.0,
    )

    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, fluents, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}, "object": {"Knife"}},
        operators=[search_op],
        true_object_locations={"bedroom": {"Knife"}},  # Knife is NOT at kitchen
    )
    actions = env.get_actions()
    search_action = get_action_by_name(actions, "search robot1 kitchen Knife")

    # This should NOT raise "Total of weights must be greater than zero"
    env.act(search_action)

    # Search should complete but NOT find the object (ground truth overrides probability)
    assert F("searched", "kitchen", "Knife") in env.state.fluents
    assert F("found", "Knife") not in env.state.fluents
    assert F("free", "robot1") in env.state.fluents


def test_search_with_zero_probability_object_present():
    """Test search with object_find_prob=0.0 when object IS at location.

    When ground truth says object is there, the environment should
    deterministically return the success branch, even when the success
    branch has probability 0.0.
    """
    from railroad.core import get_action_by_name
    from railroad import operators

    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}

    # Probability of 0.0 means success branch has probability 0.0
    search_op = operators.construct_search_operator(
        object_find_prob=0.0,  # 0% find probability
        search_time=3.0,
    )

    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, fluents, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen"}, "object": {"Knife"}},
        operators=[search_op],
        true_object_locations={"kitchen": {"Knife"}},  # Knife IS at kitchen
    )
    actions = env.get_actions()
    search_action = get_action_by_name(actions, "search robot1 kitchen Knife")

    # This should NOT raise any errors - ground truth determines outcome
    env.act(search_action)

    # Search should find the object (ground truth overrides probability)
    assert F("searched", "kitchen", "Knife") in env.state.fluents
    assert F("found", "Knife") in env.state.fluents
    assert F("at", "Knife", "kitchen") in env.state.fluents
    assert F("free", "robot1") in env.state.fluents


def test_nested_effects_immediate_still_work():
    """Test that nested effects with time=0 are still applied immediately."""
    from railroad import operators
    from railroad.core import get_action_by_name

    # Use regular search operator which has nested effects with time=0
    search_op = operators.construct_search_operator(
        object_find_prob=1.0,  # Always find
        search_time=3.0,
    )

    initial_fluents = {
        F("at", "robot1", "kitchen"),
        F("free", "robot1"),
    }

    env = env_with_operators(ObjectSearchEnvironment,
        state=State(0.0, initial_fluents, []),
        objects_by_type={
            "robot": {"robot1"},
            "location": {"kitchen"},
            "object": {"Knife"},
        },
        operators=[search_op],
        true_object_locations={"kitchen": {"Knife"}},
    )

    actions = env.get_actions()
    action = get_action_by_name(actions, "search robot1 kitchen Knife")

    env.act(action)

    # Verify the search completed correctly
    assert F("found", "Knife") in env.state.fluents
    assert F("searched", "kitchen", "Knife") in env.state.fluents
    assert F("free", "robot1") in env.state.fluents
    assert env.time == pytest.approx(3.0, abs=0.1)

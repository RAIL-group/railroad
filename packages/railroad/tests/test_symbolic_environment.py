"""Tests for SymbolicEnvironment."""

import pytest
from railroad._bindings import Fluent as F, GroundedEffect, State
from railroad.core import Effect, Operator
from railroad.environment import SymbolicEnvironment


# =============================================================================
# Construction Tests
# =============================================================================


def test_symbolic_environment_construction():
    """Test basic construction of SymbolicEnvironment."""
    initial_fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}

    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=5.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to"), F("free", "?robot")}),
        ]
    )

    env = SymbolicEnvironment(
        state=State(0.0, initial_fluents, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
        operators=[move_op],
    )

    assert env.time == 0.0
    assert F("at", "robot1", "kitchen") in env.state.fluents


# =============================================================================
# Action Execution Tests
# =============================================================================


def test_symbolic_environment_act():
    """Test acting (advancing state) with an action."""
    from railroad.core import get_action_by_name

    initial_fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}

    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=5.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to"), F("free", "?robot")}),
        ]
    )

    env = SymbolicEnvironment(
        state=State(0.0, initial_fluents, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
        operators=[move_op],
    )
    actions = env.get_actions()
    move_action = get_action_by_name(actions, "move robot1 kitchen bedroom")

    env.act(move_action)

    assert env.time == pytest.approx(5.0, abs=0.1)
    assert F("at", "robot1", "bedroom") in env.state.fluents
    assert F("free", "robot1") in env.state.fluents


def test_symbolic_environment_multi_robot_interrupt():
    """Test that robot1's move is interrupted when robot2 becomes free."""
    import numpy as np
    from railroad.environment import InterruptableMoveSymbolicSkill, LocationRegistry
    from railroad.core import get_action_by_name
    from railroad.environment import LocationRegistry

    # Two robots: robot1 at kitchen, robot2 at bedroom
    initial_fluents = {
        F("at", "robot1", "kitchen"),
        F("at", "robot2", "bedroom"),
        F("free", "robot1"),
        F("free", "robot2"),
    }

    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=10.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to"), F("free", "?robot")}),
        ]
    )
    # Short action for robot2
    wait_op = Operator(
        name="wait",
        parameters=[("?robot", "robot")],
        preconditions=[F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=2.0, resulting_fluents={F("free", "?robot")}),
        ]
    )

    env = SymbolicEnvironment(
        state=State(0.0, initial_fluents, []),
        objects_by_type={"robot": {"robot1", "robot2"}, "location": {"kitchen", "bedroom", "living_room"}},
        operators=[move_op, wait_op],
        skill_overrides={"move": InterruptableMoveSymbolicSkill},
        location_registry=LocationRegistry({}),  # Optional, but avoids warning
    )
    actions = env.get_actions()

    # Robot1 starts long move (10s)
    move_action = get_action_by_name(actions, "move robot1 kitchen living_room")
    env.act(move_action)

    # Now robot1 is busy, robot2 is still free
    assert F("free", "robot2") in env.state.fluents
    assert F("free", "robot1") not in env.state.fluents

    # Robot2 starts short wait (2s), with interrupt enabled
    actions = env.get_actions()
    wait_action = get_action_by_name(actions, "wait robot2")
    env.act(wait_action)

    # At t=2, robot2 becomes free, robot1's move should be interrupted
    assert env.time == pytest.approx(2.0, abs=0.1)
    assert F("free", "robot2") in env.state.fluents

    # Robot1 should now be at intermediate location and free
    assert F("at", "robot1", "robot1_loc") in env.state.fluents
    assert F("free", "robot1") in env.state.fluents
    assert F("at", "robot1", "living_room") not in env.state.fluents  # Did NOT reach destination


def test_interrupt_then_move_to_different_destination():
    """Test that after interruption, robot can move to a new destination with correct cost.

    Scenario:
    - Robot starts at kitchen (0,0) moving to bedroom (10,0)
    - Gets interrupted at 50% -> ends up at (5,0)
    - Then moves to living_room (10,5)
    - Expected cost: sqrt((10-5)^2 + (5-0)^2) = sqrt(50) ≈ 7.07
    """
    import math
    import numpy as np
    from railroad.environment import InterruptableMoveSymbolicSkill, LocationRegistry
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

    env = SymbolicEnvironment(
        state=State(0.0, initial_fluents, initial_effects),
        objects_by_type={"robot": {"robot1", "robot2"}, "location": {"kitchen", "bedroom", "living_room"}},
        operators=[move_op, wait_op],
        skill_overrides={"move": InterruptableMoveSymbolicSkill},
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
# Effect Application Tests
# =============================================================================


def test_symbolic_environment_apply_effect():
    """Test applying effects modifies fluents."""
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    env = SymbolicEnvironment(
        state=State(0.0, fluents, []),
        objects_by_type={},
        operators=[],
    )

    # Create a grounded effect that removes free
    effect = GroundedEffect(
        time=0.0,
        resulting_fluents={~F("free", "robot1")},
    )

    env.apply_effect(effect)

    assert F("free", "robot1") not in env.fluents


def test_symbolic_environment_apply_effect_add():
    """Test applying effects that add fluents."""
    fluents = {F("at", "robot1", "kitchen")}
    env = SymbolicEnvironment(
        state=State(0.0, fluents, []),
        objects_by_type={},
        operators=[],
    )

    # Create a grounded effect that adds a fluent
    effect = GroundedEffect(
        time=0.0,
        resulting_fluents={F("free", "robot1")},
    )

    env.apply_effect(effect)

    assert F("free", "robot1") in env.fluents


# =============================================================================
# Skill Creation Tests
# =============================================================================


def test_symbolic_environment_create_skill():
    """Test skill creation via factory method."""
    from railroad.environment import SymbolicSkill

    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
    )

    op = Operator(
        name="test",
        parameters=[("?robot", "robot")],
        preconditions=[],
        effects=[Effect(time=1.0, resulting_fluents={F("done", "?robot")})]
    )
    action = op.instantiate({"robot": ["r1"]})[0]

    skill = env.create_skill(action, time=0.0)

    assert isinstance(skill, SymbolicSkill)


def test_symbolic_environment_create_move_skill():
    """Test move skill creation via factory method."""
    from railroad.environment import SymbolicSkill, InterruptableMoveSymbolicSkill

    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
    )

    op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=10.0, resulting_fluents={
                ~F("at", "?robot", "?from"),
                F("at", "?robot", "?to"),
                F("free", "?robot")
            }),
        ]
    )
    actions = op.instantiate({"robot": ["r1"], "location": ["kitchen", "bedroom"]})
    action = [a for a in actions if "kitchen" in a.name and "bedroom" in a.name][0]

    skill = env.create_skill(action, time=0.0)

    # Move skills use SymbolicSkill by default (not interruptible)
    assert isinstance(skill, SymbolicSkill)
    assert not isinstance(skill, InterruptableMoveSymbolicSkill)
    assert not skill.is_interruptible

    # Can use skill_overrides to make moves interruptible
    env_with_override = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
        skill_overrides={"move": InterruptableMoveSymbolicSkill},
    )
    skill_interruptible = env_with_override.create_skill(action, time=0.0)
    assert isinstance(skill_interruptible, InterruptableMoveSymbolicSkill)
    assert skill_interruptible.is_interruptible


def test_symbolic_environment_create_search_skill():
    """Test search skill creation via factory method."""
    from railroad.environment import SymbolicSkill
    from railroad import operators

    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={"robot": {"r1"}, "location": {"kitchen"}, "object": {"Knife"}},
        operators=[],
    )

    search_op = operators.construct_search_operator(
        object_find_prob=0.5,
        search_time=3.0,
    )
    actions = search_op.instantiate(env.objects_by_type)
    search_action = [a for a in actions if "r1" in a.name and "kitchen" in a.name and "Knife" in a.name][0]

    skill = env.create_skill(search_action, time=0.0)

    assert isinstance(skill, SymbolicSkill)
    assert not skill.is_interruptible  # Search skills are not interruptible


def test_symbolic_environment_create_pick_skill():
    """Test pick skill creation via factory method."""
    from railroad.environment import SymbolicSkill
    from railroad import operators

    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={"robot": {"r1"}, "location": {"kitchen"}, "object": {"Knife"}},
        operators=[],
        true_object_locations={"kitchen": {"Knife"}},
    )

    pick_op = operators.construct_pick_operator_blocking(pick_time=2.0)
    actions = pick_op.instantiate(env.objects_by_type)
    pick_action = [a for a in actions if "r1" in a.name and "kitchen" in a.name and "Knife" in a.name][0]

    skill = env.create_skill(pick_action, time=0.0)

    assert isinstance(skill, SymbolicSkill)
    assert not skill.is_interruptible  # Pick skills are not interruptible


def test_symbolic_environment_create_place_skill():
    """Test place skill creation via factory method."""
    from railroad.environment import SymbolicSkill
    from railroad import operators

    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={"robot": {"r1"}, "location": {"bedroom"}, "object": {"Knife"}},
        operators=[],
    )

    place_op = operators.construct_place_operator_blocking(place_time=2.0)
    actions = place_op.instantiate(env.objects_by_type)
    place_action = [a for a in actions if "r1" in a.name and "bedroom" in a.name and "Knife" in a.name][0]

    skill = env.create_skill(place_action, time=0.0)

    assert isinstance(skill, SymbolicSkill)
    assert not skill.is_interruptible  # Place skills are not interruptible


# =============================================================================
# Revelation Tests (Object Discovery)
# =============================================================================


def test_symbolic_environment_revelation():
    """Test that searching a location reveals objects at that location."""
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}

    env = SymbolicEnvironment(
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


def test_symbolic_environment_objects_at_locations():
    """Test internal objects_at_locations tracking."""
    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
        true_object_locations={"kitchen": {"Knife", "Fork"}},
    )

    # Access internal state for verification
    assert env._objects_at_locations["kitchen"] == {"Knife", "Fork"}
    assert env._objects_at_locations.get("bedroom", set()) == set()


def test_symbolic_environment_object_location_from_fluents():
    """Test that object locations are derived from fluents."""
    # Initial ground truth: Knife and Fork at kitchen
    env = SymbolicEnvironment(
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


def test_symbolic_environment_fluent_overrides_ground_truth():
    """Test that fluents override initial ground truth for object locations."""
    # Initial ground truth: Knife at kitchen
    env = SymbolicEnvironment(
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


def test_symbolic_environment_resolve_probabilistic_effect():
    """Test resolving probabilistic effects based on ground truth."""
    # Create an environment where "obj" IS at "loc"
    env = SymbolicEnvironment(
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
    env2 = SymbolicEnvironment(
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

    env = SymbolicEnvironment(
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

    env = SymbolicEnvironment(
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

    env = SymbolicEnvironment(
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

    env = SymbolicEnvironment(
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

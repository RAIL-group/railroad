"""Tests for ActiveSkill protocol."""
import pytest


def test_symbolic_skill_base_class():
    """Test SymbolicSkill base implementation."""
    from railroad.environment import SymbolicSkill
    from railroad._bindings import Fluent as F
    from railroad.core import Effect, Operator

    # Create a simple action
    op = Operator(
        name="test_action",
        parameters=[("?robot", "robot")],
        preconditions=[F("free", "?robot")],
        effects=[Effect(time=5.0, resulting_fluents={F("done", "?robot")})]
    )
    actions = op.instantiate({"robot": ["r1"]})
    action = actions[0]

    skill = SymbolicSkill(action=action, start_time=0.0)

    assert skill.is_done is False
    assert skill.is_interruptible is False  # Default
    assert len(skill.upcoming_effects) == 1
    assert skill.time_to_next_event == 5.0


def test_symbolic_skill_move_not_interruptible_by_default():
    """Test that SymbolicSkill is NOT interruptible for move actions by default."""
    from railroad.environment import SymbolicSkill
    from railroad._bindings import Fluent as F
    from railroad.core import Effect, Operator

    # Create a move action
    op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=10.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to"), F("free", "?robot")}),
        ]
    )
    actions = op.instantiate({"robot": ["r1"], "location": ["kitchen", "bedroom"]})
    action = [a for a in actions if "kitchen" in a.name and "bedroom" in a.name][0]

    skill = SymbolicSkill(action=action, start_time=0.0)

    # Move actions are NOT interruptible by default
    assert skill.is_interruptible is False


def test_interruptible_navigation_move_skill_interrupt_behavior():
    """Test that InterruptibleNavigationMoveSkill.interrupt() rewrites fluents correctly."""
    import numpy as np
    from railroad.environment import InterruptibleNavigationMoveSkill, LocationRegistry, SymbolicEnvironment
    from railroad._bindings import Fluent as F, State
    from railroad.core import Effect, Operator

    # Create move operator
    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=10.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to"), F("free", "?robot")}),
        ]
    )

    # Create environment with registry for trajectory construction
    initial_state = State(0.0, {F("at", "r1", "kitchen"), F("free", "r1")})
    registry = LocationRegistry({
        "kitchen": np.array([0.0, 0.0]),
        "bedroom": np.array([10.0, 0.0]),
    })
    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type={"robot": {"r1"}, "location": {"kitchen", "bedroom"}},
        operators=[move_op],
        location_registry=registry,
    )

    actions = move_op.instantiate({"robot": ["r1"], "location": ["kitchen", "bedroom"]})
    action = [a for a in actions if "kitchen" in a.name and "bedroom" in a.name][0]

    skill = InterruptibleNavigationMoveSkill(action=action, start_time=0.0, env=env)

    # InterruptibleNavigationMoveSkill is interruptible
    assert skill.is_interruptible is True

    # Advance partway (apply first effect at t=0)
    skill.advance(0.0, env)
    assert F("free", "r1") not in env.fluents  # First effect applied

    # Advance time but not to completion
    skill.advance(5.0, env)  # Updates _current_time internally

    # Interrupt
    skill.interrupt(env)

    # Verify interrupt behavior
    assert skill.is_done  # upcoming_effects should be cleared
    assert F("at", "r1", "r1_loc") in env.fluents  # Destination rewritten to robot_loc
    assert F("free", "r1") in env.fluents  # Free fluent added
    assert F("at", "r1", "bedroom") not in env.fluents  # Original destination NOT used


def test_location_registry_basic():
    """Test LocationRegistry basic functionality."""
    import numpy as np
    from railroad.environment import LocationRegistry

    locations = {
        "kitchen": np.array([0.0, 0.0]),
        "bedroom": np.array([10.0, 0.0]),
    }
    registry = LocationRegistry(locations)

    # Test get
    kitchen_pos = registry.get("kitchen")
    assert kitchen_pos is not None
    assert np.array_equal(kitchen_pos, np.array([0.0, 0.0]))
    assert registry.get("unknown") is None

    # Test contains
    assert "kitchen" in registry
    assert "unknown" not in registry

    # Test register
    registry.register("living_room", np.array([5.0, 5.0]))
    assert "living_room" in registry
    living_room_pos = registry.get("living_room")
    assert living_room_pos is not None
    assert np.array_equal(living_room_pos, np.array([5.0, 5.0]))


def test_location_registry_move_time_fn():
    """Test LocationRegistry.move_time_fn helper."""
    import numpy as np
    from railroad.environment import LocationRegistry

    locations = {
        "kitchen": np.array([0.0, 0.0]),
        "bedroom": np.array([10.0, 0.0]),  # 10 units away
    }
    registry = LocationRegistry(locations)

    move_time = registry.move_time_fn(velocity=2.0)

    # Distance is 10, velocity is 2, so time should be 5
    assert move_time("robot1", "kitchen", "bedroom") == pytest.approx(5.0)
    assert move_time("robot1", "bedroom", "kitchen") == pytest.approx(5.0)

    # Unknown location returns infinity
    assert move_time("robot1", "kitchen", "unknown") == float("inf")
    assert move_time("robot1", "unknown", "bedroom") == float("inf")


def test_location_registry_interrupt_registers_coords():
    """Test that interrupt registers intermediate location coordinates."""
    import numpy as np
    from railroad.environment import InterruptibleNavigationMoveSkill, LocationRegistry, SymbolicEnvironment
    from railroad._bindings import Fluent as F, State
    from railroad.core import Effect, Operator

    # Create registry with locations
    locations = {
        "kitchen": np.array([0.0, 0.0]),
        "bedroom": np.array([10.0, 0.0]),
    }
    registry = LocationRegistry(locations)

    # Create move operator
    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=10.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to"), F("free", "?robot")}),
        ]
    )

    # Create environment with registry
    initial_state = State(0.0, {F("at", "r1", "kitchen"), F("free", "r1")})
    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type={"robot": {"r1"}, "location": {"kitchen", "bedroom"}},
        operators=[move_op],
        skill_overrides={"move": InterruptibleNavigationMoveSkill},
        location_registry=registry,
    )

    actions = move_op.instantiate({"robot": ["r1"], "location": ["kitchen", "bedroom"]})
    action = [a for a in actions if "kitchen" in a.name and "bedroom" in a.name][0]

    skill = InterruptibleNavigationMoveSkill(action=action, start_time=0.0, env=env)

    # Advance to 50% progress (5s out of 10s)
    skill.advance(0.0, env)
    skill.advance(5.0, env)

    # Interrupt - should register r1_loc at 50% between kitchen and bedroom
    skill.interrupt(env)

    # Verify intermediate location was registered
    assert "r1_loc" in registry
    intermediate_coords = registry.get("r1_loc")
    assert intermediate_coords is not None
    # 50% of the way from [0,0] to [10,0] is [5,0]
    assert np.allclose(intermediate_coords, np.array([5.0, 0.0]))

    # Now move_time should work with the intermediate location
    move_time = registry.move_time_fn(velocity=1.0)
    # Distance from r1_loc [5,0] to bedroom [10,0] is 5
    assert move_time("r1", "r1_loc", "bedroom") == pytest.approx(5.0)


def test_skill_overrides_mapping():
    """Test that SymbolicEnvironment respects skill_overrides."""
    from railroad.environment import (
        InterruptibleNavigationMoveSkill,
        LocationRegistry,
        SymbolicEnvironment,
        SymbolicSkill,
    )
    from railroad._bindings import Fluent as F, State
    from railroad.core import Effect, Operator

    import numpy as np
    registry = LocationRegistry({"kitchen": np.array([0, 0]), "bedroom": np.array([10, 0])})

    # Create environment with skill override for move actions
    env = SymbolicEnvironment(
        state=State(0.0, {F("at", "r1", "kitchen"), F("free", "r1")}, []),
        objects_by_type={"robot": {"r1"}, "location": {"kitchen", "bedroom"}},
        operators=[],
        skill_overrides={"move": InterruptibleNavigationMoveSkill},
        location_registry=registry,
    )

    # Create move and non-move actions
    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[Effect(time=5.0, resulting_fluents={F("at", "?robot", "?to")})],
    )
    wait_op = Operator(
        name="wait",
        parameters=[("?robot", "robot")],
        preconditions=[F("free", "?robot")],
        effects=[Effect(time=1.0, resulting_fluents={F("waited", "?robot")})],
    )

    move_actions = move_op.instantiate({"robot": ["r1"], "location": ["kitchen", "bedroom"]})
    wait_actions = wait_op.instantiate({"robot": ["r1"]})

    move_action = [a for a in move_actions if "kitchen" in a.name and "bedroom" in a.name][0]
    wait_action = wait_actions[0]

    # Move should get InterruptibleNavigationMoveSkill
    move_skill = env.create_skill(move_action, 0.0)
    assert isinstance(move_skill, InterruptibleNavigationMoveSkill)
    assert move_skill.is_interruptible is True

    # Wait should get default SymbolicSkill
    wait_skill = env.create_skill(wait_action, 0.0)
    assert isinstance(wait_skill, SymbolicSkill)
    assert not isinstance(wait_skill, InterruptibleNavigationMoveSkill)
    assert wait_skill.is_interruptible is False

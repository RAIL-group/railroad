"""Tests for ActiveSkill protocol."""
import pytest


def test_symbolic_skill_base_class():
    """Test SymbolicSkill base implementation."""
    from railroad.environment.symbolic import SymbolicSkill
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
    from railroad.environment.symbolic import SymbolicSkill
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


def test_interruptable_move_skill_interrupt_behavior():
    """Test that InterruptableMoveSymbolicSkill.interrupt() rewrites fluents correctly."""
    from railroad.environment.symbolic import InterruptableMoveSymbolicSkill
    from railroad.environment import SymbolicEnvironment
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

    # Create environment
    initial_state = State(0.0, {F("at", "r1", "kitchen"), F("free", "r1")})
    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type={"robot": {"r1"}, "location": {"kitchen", "bedroom"}},
        operators=[move_op],
    )

    actions = move_op.instantiate({"robot": ["r1"], "location": ["kitchen", "bedroom"]})
    action = [a for a in actions if "kitchen" in a.name and "bedroom" in a.name][0]

    skill = InterruptableMoveSymbolicSkill(action=action, start_time=0.0)

    # InterruptableMoveSymbolicSkill IS interruptible
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


def test_skill_overrides_mapping():
    """Test that SymbolicEnvironment respects skill_overrides."""
    from railroad.environment.symbolic import SymbolicSkill, InterruptableMoveSymbolicSkill
    from railroad.environment.symbolic import SymbolicEnvironment
    from railroad._bindings import Fluent as F, State
    from railroad.core import Effect, Operator

    # Create environment with skill override for move actions
    env = SymbolicEnvironment(
        state=State(0.0, {F("at", "r1", "kitchen"), F("free", "r1")}, []),
        objects_by_type={"robot": {"r1"}, "location": {"kitchen", "bedroom"}},
        operators=[],
        skill_overrides={"move": InterruptableMoveSymbolicSkill},
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

    # Move should get InterruptableMoveSymbolicSkill
    move_skill = env.create_skill(move_action, 0.0)
    assert isinstance(move_skill, InterruptableMoveSymbolicSkill)
    assert move_skill.is_interruptible is True

    # Wait should get default SymbolicSkill
    wait_skill = env.create_skill(wait_action, 0.0)
    assert isinstance(wait_skill, SymbolicSkill)
    assert not isinstance(wait_skill, InterruptableMoveSymbolicSkill)
    assert wait_skill.is_interruptible is False

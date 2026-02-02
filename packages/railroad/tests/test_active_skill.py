"""Tests for ActiveSkill protocol."""
import pytest
from typing import Protocol, runtime_checkable


def test_active_skill_protocol_exists():
    """Test that ActiveSkill protocol can be imported."""
    from railroad.environment.skill import ActiveSkill

    assert hasattr(ActiveSkill, 'robot')
    assert hasattr(ActiveSkill, 'is_done')
    assert hasattr(ActiveSkill, 'is_interruptible')
    assert hasattr(ActiveSkill, 'upcoming_effects')
    assert hasattr(ActiveSkill, 'time_to_next_event')
    assert hasattr(ActiveSkill, 'advance')
    assert hasattr(ActiveSkill, 'interrupt')


def test_environment_protocol_exists():
    """Test that Environment protocol can be imported."""
    from railroad.environment.skill import Environment

    assert hasattr(Environment, 'fluents')
    assert hasattr(Environment, 'objects_by_type')
    assert hasattr(Environment, 'create_skill')
    assert hasattr(Environment, 'apply_effect')
    assert hasattr(Environment, 'resolve_probabilistic_effect')
    assert hasattr(Environment, 'get_objects_at_location')
    assert hasattr(Environment, 'remove_object_from_location')
    assert hasattr(Environment, 'add_object_at_location')


def test_symbolic_skill_base_class():
    """Test SymbolicSkill base implementation."""
    from railroad.environment.skill import SymbolicSkill
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

    skill = SymbolicSkill(action=action, start_time=0.0, robot="r1")

    assert skill.robot == "r1"
    assert skill.is_done is False
    assert skill.is_interruptible is False  # Default
    assert len(skill.upcoming_effects) == 1
    assert skill.time_to_next_event == 5.0

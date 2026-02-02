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

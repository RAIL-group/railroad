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

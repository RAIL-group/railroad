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


def test_symbolic_skill_move_not_interruptible_by_default():
    """Test that SymbolicSkill is NOT interruptible for move actions by default."""
    from railroad.environment.skill import SymbolicSkill
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

    skill = SymbolicSkill(action=action, start_time=0.0, robot="r1")

    # Move actions are NOT interruptible by default
    assert skill.is_interruptible is False


def test_interruptable_move_skill_interrupt_behavior():
    """Test that InterruptableMoveSymbolicSkill.interrupt() rewrites fluents correctly."""
    from railroad.environment.skill import InterruptableMoveSymbolicSkill
    from railroad._bindings import Fluent as F
    from railroad.core import Effect, Operator

    # Create a simple mock environment that implements Environment protocol
    class MockEnvironment:
        def __init__(self):
            self._fluents = {F("at", "r1", "kitchen"), F("free", "r1")}
            self._objects_by_type = {"robot": {"r1"}, "location": {"kitchen", "bedroom"}}

        @property
        def fluents(self):
            return self._fluents

        @property
        def objects_by_type(self):
            return self._objects_by_type

        def create_skill(self, action, time):
            from railroad.environment.skill import SymbolicSkill
            robot = action.name.split()[1] if action.name else "r1"
            return SymbolicSkill(action=action, start_time=time, robot=robot)

        def apply_effect(self, effect):
            for fluent in effect.resulting_fluents:
                if fluent.negated:
                    self._fluents.discard(~fluent)
                else:
                    self._fluents.add(fluent)

        def resolve_probabilistic_effect(self, effect, current_fluents):
            return [effect], current_fluents

        def get_objects_at_location(self, location):
            return {}

        def remove_object_from_location(self, obj, location):
            pass

        def add_object_at_location(self, obj, location):
            pass

    # Create move operator and action
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

    env = MockEnvironment()
    skill = InterruptableMoveSymbolicSkill(action=action, start_time=0.0, robot="r1")

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
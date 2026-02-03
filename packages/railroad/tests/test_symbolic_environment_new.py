"""Tests for SymbolicEnvironment."""
import pytest
from railroad._bindings import Fluent as F, State
from railroad.core import Effect, Operator


def test_symbolic_environment_construction():
    """Test SymbolicEnvironment can be constructed with new API."""
    from railroad.environment.symbolic import SymbolicEnvironment

    initial_fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    initial_state = State(0.0, initial_fluents, [])

    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
        operators=[],
        true_object_locations={"kitchen": {"Knife"}},
    )

    assert F("at", "robot1", "kitchen") in env.fluents
    assert env.time == 0.0


def test_symbolic_environment_act():
    """Test SymbolicEnvironment executes actions correctly."""
    from railroad.environment.symbolic import SymbolicEnvironment

    initial_fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    initial_state = State(0.0, initial_fluents, [])

    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?robot ?from"), F("free ?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents=[~F("free ?robot")]),
            Effect(time=5.0, resulting_fluents=[~F("at ?robot ?from"), F("at ?robot ?to"), F("free ?robot")]),
        ]
    )

    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
        operators=[move_op],
    )

    actions = env.get_actions()
    move_action = next(a for a in actions if a.name == "move robot1 kitchen bedroom")

    env.act(move_action, do_interrupt=False)

    assert env.time == pytest.approx(5.0, abs=0.1)
    assert F("at", "robot1", "bedroom") in env.state.fluents

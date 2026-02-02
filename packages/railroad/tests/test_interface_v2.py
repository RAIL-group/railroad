"""Tests for new EnvironmentInterface (v2)."""
import pytest
from railroad._bindings import Fluent as F, State
from railroad.core import Effect, Operator


def test_interface_v2_construction():
    """Test basic construction of new EnvironmentInterface."""
    from railroad.environment.interface_v2 import EnvironmentInterfaceV2
    from railroad.environment.symbolic import SimpleSymbolicEnvironment

    initial_fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    env = SimpleSymbolicEnvironment(
        initial_state=State(0.0, initial_fluents, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
        objects_at_locations={},
    )

    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=5.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to"), F("free", "?robot")}),
        ]
    )

    interface = EnvironmentInterfaceV2(
        environment=env,
        operators=[move_op],
    )

    assert interface.time == 0.0
    assert F("at", "robot1", "kitchen") in interface.state.fluents


def test_interface_v2_advance():
    """Test advancing state with an action."""
    from railroad.environment.interface_v2 import EnvironmentInterfaceV2
    from railroad.environment.symbolic import SimpleSymbolicEnvironment
    from railroad.core import get_action_by_name

    initial_fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    env = SimpleSymbolicEnvironment(
        initial_state=State(0.0, initial_fluents, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
        objects_at_locations={},
    )

    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=5.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to"), F("free", "?robot")}),
        ]
    )

    interface = EnvironmentInterfaceV2(environment=env, operators=[move_op])
    actions = interface.get_actions()
    move_action = get_action_by_name(actions, "move robot1 kitchen bedroom")

    interface.advance(move_action, do_interrupt=False)

    assert interface.time == pytest.approx(5.0, abs=0.1)
    assert F("at", "robot1", "bedroom") in interface.state.fluents
    assert F("free", "robot1") in interface.state.fluents

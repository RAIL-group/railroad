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


def test_interface_v2_multi_robot_interrupt():
    """Test that robot1's move is interrupted when robot2 becomes free."""
    from railroad.environment.interface_v2 import EnvironmentInterfaceV2
    from railroad.environment.symbolic import SimpleSymbolicEnvironment
    from railroad.environment.skill import InterruptableMoveSymbolicSkill
    from railroad.core import get_action_by_name

    # Two robots: robot1 at kitchen, robot2 at bedroom
    initial_fluents = {
        F("at", "robot1", "kitchen"),
        F("at", "robot2", "bedroom"),
        F("free", "robot1"),
        F("free", "robot2"),
    }
    env = SimpleSymbolicEnvironment(
        initial_state=State(0.0, initial_fluents, []),
        objects_by_type={"robot": {"robot1", "robot2"}, "location": {"kitchen", "bedroom", "living_room"}},
        objects_at_locations={},
        skill_overrides={"move": InterruptableMoveSymbolicSkill},
    )

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

    interface = EnvironmentInterfaceV2(environment=env, operators=[move_op, wait_op])
    actions = interface.get_actions()

    # Robot1 starts long move (10s)
    move_action = get_action_by_name(actions, "move robot1 kitchen living_room")
    interface.advance(move_action, do_interrupt=False)

    # Now robot1 is busy, robot2 is still free
    assert F("free", "robot2") in interface.state.fluents
    assert F("free", "robot1") not in interface.state.fluents

    # Robot2 starts short wait (2s), with interrupt enabled
    actions = interface.get_actions()
    wait_action = get_action_by_name(actions, "wait robot2")
    interface.advance(wait_action, do_interrupt=True)

    # At t=2, robot2 becomes free, robot1's move should be interrupted
    assert interface.time == pytest.approx(2.0, abs=0.1)
    assert F("free", "robot2") in interface.state.fluents

    # Robot1 should now be at intermediate location and free
    assert F("at", "robot1", "robot1_loc") in interface.state.fluents
    assert F("free", "robot1") in interface.state.fluents
    assert F("at", "robot1", "living_room") not in interface.state.fluents  # Did NOT reach destination

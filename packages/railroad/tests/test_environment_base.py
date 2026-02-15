"""Tests for base Environment class."""
import pytest
from typing import Dict, Set, List
from railroad._bindings import Fluent as F, State
from railroad.core import Effect, Operator
from railroad.environment.environment import Environment


class MinimalEnvironment(Environment):
    """Minimal concrete implementation for testing base class."""

    def __init__(self, state: State, operators: List[Operator], fluents: Set[F]):
        self._fluents_set = fluents
        self._objects = {"robot": {"robot1"}, "location": {"kitchen", "bedroom"}}
        super().__init__(state=state, operators=operators)

    @property
    def fluents(self) -> Set[F]:
        return self._fluents_set

    @property
    def objects_by_type(self) -> Dict[str, Set[str]]:
        return self._objects

    def create_skill(self, action, time):
        from railroad.environment import SymbolicSkill
        return SymbolicSkill(action=action, start_time=time)

    def _create_initial_effects_skill(self, start_time, upcoming_effects):
        from railroad.environment import SymbolicSkill
        from railroad._bindings import Action, GroundedEffect
        relative_effects = [
            GroundedEffect(abs_time - start_time, effect.resulting_fluents)
            for abs_time, effect in upcoming_effects
        ]
        action = Action(set(), relative_effects, name="_initial_effects")
        return SymbolicSkill(action=action, start_time=start_time)

    def get_objects_at_location(self, location: str) -> Dict[str, Set[str]]:
        return {"object": set()}


def test_environment_state_assembly():
    """Test that state property assembles fluents + upcoming effects."""
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    state = State(0.0, fluents, [])

    env = MinimalEnvironment(state=state, operators=[], fluents=fluents)

    assert env.time == 0.0
    assert F("at", "robot1", "kitchen") in env.state.fluents
    assert F("free", "robot1") in env.state.fluents


def test_environment_get_actions():
    """Test that get_actions instantiates from operators."""
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    state = State(0.0, fluents, [])

    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?robot ?from"), F("free ?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free ?robot")}),
            Effect(time=5.0, resulting_fluents={~F("at ?robot ?from"), F("at ?robot ?to"), F("free ?robot")}),
        ]
    )

    env = MinimalEnvironment(state=state, operators=[move_op], fluents=fluents)
    actions = env.get_actions()

    action_names = [a.name for a in actions]
    assert "move robot1 kitchen bedroom" in action_names


def test_environment_act_executes_action():
    """Test that act() executes an action and returns new state."""
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    state = State(0.0, fluents, [])

    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?robot ?from"), F("free ?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free ?robot")}),
            Effect(time=5.0, resulting_fluents={~F("at ?robot ?from"), F("at ?robot ?to"), F("free ?robot")}),
        ]
    )

    env = MinimalEnvironment(state=state, operators=[move_op], fluents=fluents)
    actions = env.get_actions()
    move_action = next(a for a in actions if a.name == "move robot1 kitchen bedroom")

    result_state = env.act(move_action)

    assert env.time == pytest.approx(5.0, abs=0.1)
    assert F("at", "robot1", "bedroom") in result_state.fluents
    assert F("free", "robot1") in result_state.fluents


def test_environment_act_rejects_invalid_preconditions():
    """Test that act() raises ValueError for invalid preconditions."""
    fluents = {F("at", "robot1", "kitchen")}  # Missing "free robot1"
    state = State(0.0, fluents, [])

    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?robot ?from"), F("free ?robot")],
        effects=[Effect(time=5.0, resulting_fluents={F("at ?robot ?to")})]
    )

    env = MinimalEnvironment(state=state, operators=[move_op], fluents=fluents)
    actions = env.get_actions()
    move_action = next(a for a in actions if a.name == "move robot1 kitchen bedroom")

    with pytest.raises(ValueError, match="preconditions not satisfied"):
        env.act(move_action)

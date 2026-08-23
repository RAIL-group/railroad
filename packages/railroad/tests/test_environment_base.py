"""Tests for base Environment class."""
import sys
import warnings

import pytest

from env_helpers import make_move_op
from typing import Dict, Set, List
from railroad._bindings import Fluent as F, State
from railroad.core import Effect, Operator
from railroad.environment.environment import Environment


class _MinimalBody:
    """Abstract-method bodies shared by the two minimal environments below."""

    # Set by each subclass's __init__ before super().__init__ runs.
    _fluents_set: Set[F]
    _objects: Dict[str, Set[str]]

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

    def apply_effect(self, effect):
        for fluent in effect.resulting_fluents:
            if fluent.negated:
                self._fluents_set.discard(~fluent)
            else:
                self._fluents_set.add(fluent)
        return []  # No delayed effects in this minimal implementation

    def resolve_probabilistic_effect(self, effect, current_fluents):
        return [effect], current_fluents


class MinimalEnvironment(_MinimalBody, Environment):
    """Minimal concrete implementation for testing base class."""

    def __init__(self, state: State, operators: List[Operator], fluents: Set[F]):
        self._fluents_set = fluents
        self._objects = {"robot": {"robot1"}, "location": {"kitchen", "bedroom"}}
        self._operators_to_define = list(operators)
        super().__init__(state=state)

    def define_operators(self) -> List[Operator]:
        return self._operators_to_define


class LegacyKwargEnvironment(_MinimalBody, Environment):
    """Resolves operators through the deprecated ``operators=`` kwarg.

    Kept deliberately: the kwarg is still supported, so something has to pin
    that it works and that it warns. Everything else in this file goes through
    ``define_operators()``.
    """

    def __init__(self, state: State, operators: List[Operator], fluents: Set[F]):
        self._fluents_set = fluents
        self._objects = {"robot": {"robot1"}, "location": {"kitchen", "bedroom"}}
        super().__init__(state=state, operators=operators)


def test_deprecated_operators_kwarg_still_resolves_and_warns():
    """The deprecated path works, warns, and blames the caller -- not the library."""
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?robot ?from"), F("free ?robot")],
        effects=[Effect(time=5.0, resulting_fluents={F("at ?robot ?to")})],
    )

    with pytest.warns(DeprecationWarning, match="Prefer overriding") as recorded:
        env = LegacyKwargEnvironment(
            state=State(0.0, fluents, []), operators=[move_op], fluents=fluents
        )

    # The operators still take effect...
    assert "move robot1 kitchen bedroom" in [a.name for a in env.get_actions()]
    # ...and the warning points at this file, not at environment.py. A fixed
    # stacklevel used to name the library for every caller.
    assert recorded[0].filename == __file__


def test_deprecation_blames_a_caller_that_builds_from_its_own_constructor():
    """The frame walk stops at the caller's ``__init__``, not one frame past it.

    Building an environment inside a constructor is the common wrapper shape.
    Walking out to "the first frame that is not an ``__init__``" consumes that
    frame too, blaming whatever called ``Experiment()`` -- so this pins the
    line number, not just the file: the test above passes either way.
    """
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    move_op = make_move_op()

    class Experiment:
        def __init__(self):
            self.expected_lineno = sys._getframe().f_lineno + 1
            self.env = LegacyKwargEnvironment(
                state=State(0.0, fluents, []), operators=[move_op], fluents=fluents
            )

    with pytest.warns(DeprecationWarning, match="Prefer overriding") as recorded:
        experiment = Experiment()

    assert recorded[0].filename == __file__
    assert recorded[0].lineno == experiment.expected_lineno


def test_resolve_operators_warns_for_direct_callers_but_not_for_init():
    """The helper is deprecated for downstream callers; ``__init__``'s call is not.

    Kept deliberately: ``define_operators()`` is the intended API, and this
    warning is how an out-of-tree subclass calling the old helper finds that
    out. ``_from_init=True`` marks the one call that must stay silent -- no
    other test here would notice if it started warning.
    """
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    move_op = make_move_op()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = MinimalEnvironment(
            state=State(0.0, fluents, []), operators=[move_op], fluents=fluents
        )
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []

    with pytest.warns(DeprecationWarning, match="_resolve_operators"):
        assert env._resolve_operators(None) == [move_op]


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

    move_op = make_move_op()

    env = MinimalEnvironment(state=state, operators=[move_op], fluents=fluents)
    actions = env.get_actions()

    action_names = [a.name for a in actions]
    assert "move robot1 kitchen bedroom" in action_names


def test_environment_act_executes_action():
    """Test that act() executes an action and returns new state."""
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    state = State(0.0, fluents, [])

    move_op = make_move_op()

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

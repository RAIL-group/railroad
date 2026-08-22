"""Tests for Goal pickling support (required for multiprocessing in benchmarks)."""

import pickle

import pytest

from railroad.core import Fluent as F
from railroad._bindings import TrueGoal, FalseGoal


@pytest.mark.parametrize(
    "goal",
    [
        TrueGoal(),
        FalseGoal(),
        F("at robot1 kitchen"),
        ~F("at robot1 kitchen"),
        F("at robot1 kitchen") & F("at robot2 bedroom") & F("found Knife"),
        F("at robot1 kitchen") | F("at robot1 bedroom"),
        # AND containing OR children.
        (F("at robot1 kitchen") | F("at robot1 bedroom")) & F("found Knife"),
        # Both levels branching.
        ((F("a") & F("b")) | (F("c") & F("d"))) & (F("e") | F("f")),
    ],
    ids=["true", "false", "literal", "negated_literal", "and", "or", "nested",
         "deeply_nested"],
)
def test_goal_survives_a_pickle_round_trip(goal):
    """Every Goal type reconstructs equal to -- and hashing as -- the original.

    The hash check matters as much as equality: benchmark workers put goals in
    sets and dict keys across the process boundary, so a goal that compares
    equal but hashes differently would silently duplicate. Only half of the
    merged tests used to assert it.
    """
    restored = pickle.loads(pickle.dumps(goal))
    assert goal == restored
    assert hash(goal) == hash(restored)


def test_goal_evaluate_after_pickle():
    """Verify restored goals still evaluate correctly."""
    goal = F("at robot1 kitchen") & ~F("holding robot1 obj")
    restored = pickle.loads(pickle.dumps(goal))

    state_satisfied = {F("at robot1 kitchen")}
    state_not_satisfied = {F("at robot1 kitchen"), F("holding robot1 obj")}

    assert restored.evaluate(state_satisfied) is True
    assert restored.evaluate(state_not_satisfied) is False

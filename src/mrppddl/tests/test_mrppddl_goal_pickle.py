"""Tests for Goal pickling support (required for multiprocessing in benchmarks)."""

import pickle
import pytest

from mrppddl.core import Fluent as F
from mrppddl._bindings import TrueGoal, FalseGoal, LiteralGoal, AndGoal, OrGoal


class TestGoalPickle:
    """Test that all Goal types can be pickled and unpickled correctly."""

    def test_true_goal_pickle(self):
        goal = TrueGoal()
        restored = pickle.loads(pickle.dumps(goal))
        assert goal == restored

    def test_false_goal_pickle(self):
        goal = FalseGoal()
        restored = pickle.loads(pickle.dumps(goal))
        assert goal == restored

    def test_literal_goal_pickle(self):
        goal = F("at robot1 kitchen")
        restored = pickle.loads(pickle.dumps(goal))
        assert goal == restored
        assert hash(goal) == hash(restored)

    def test_negated_literal_pickle(self):
        goal = ~F("at robot1 kitchen")
        restored = pickle.loads(pickle.dumps(goal))
        assert goal == restored

    def test_and_goal_pickle(self):
        goal = F("at robot1 kitchen") & F("at robot2 bedroom") & F("found Knife")
        restored = pickle.loads(pickle.dumps(goal))
        assert goal == restored
        assert hash(goal) == hash(restored)

    def test_or_goal_pickle(self):
        goal = F("at robot1 kitchen") | F("at robot1 bedroom")
        restored = pickle.loads(pickle.dumps(goal))
        assert goal == restored
        assert hash(goal) == hash(restored)

    def test_nested_goal_pickle(self):
        """Test AND containing OR children."""
        goal = (F("at robot1 kitchen") | F("at robot1 bedroom")) & F("found Knife")
        restored = pickle.loads(pickle.dumps(goal))
        assert goal == restored

    def test_deeply_nested_goal_pickle(self):
        """Test complex nested structure."""
        goal = (
            (F("a") & F("b")) | (F("c") & F("d"))
        ) & (
            F("e") | F("f")
        )
        restored = pickle.loads(pickle.dumps(goal))
        assert goal == restored
        assert hash(goal) == hash(restored)

    def test_goal_evaluate_after_pickle(self):
        """Verify restored goals still evaluate correctly."""
        goal = F("at robot1 kitchen") & ~F("holding robot1 obj")
        restored = pickle.loads(pickle.dumps(goal))

        state_satisfied = {F("at robot1 kitchen")}
        state_not_satisfied = {F("at robot1 kitchen"), F("holding robot1 obj")}

        assert restored.evaluate(state_satisfied) is True
        assert restored.evaluate(state_not_satisfied) is False

"""Tests for base Environment class."""
import pytest
from abc import ABC


def test_environment_is_abstract():
    """Test that Environment cannot be instantiated directly."""
    from railroad.environment.environment import Environment

    assert issubclass(Environment, ABC)

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Environment(state=None, operators=[])

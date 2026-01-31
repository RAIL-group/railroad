"""
Tests for benchmark registry and parameter access.
"""

import pytest
from railroad.bench.registry import BenchmarkCase, _DotAccessor


class TestBenchmarkCaseDotNotation:
    """Test dot notation access to benchmark case parameters."""

    def test_simple_parameter_access(self):
        """Test accessing simple parameters via dot notation."""
        case = BenchmarkCase(
            benchmark_name="test",
            case_idx=0,
            repeat_idx=0,
            params={"num_robots": 2, "seed": 42}
        )

        assert case.num_robots == 2
        assert case.seed == 42

    def test_nested_parameter_access(self):
        """Test accessing nested parameters with dots."""
        case = BenchmarkCase(
            benchmark_name="test",
            case_idx=0,
            repeat_idx=0,
            params={
                "mcts.iterations": 100,
                "mcts.exploration": 1.4,
                "num_robots": 2,
            }
        )

        assert case.mcts.iterations == 100
        assert case.mcts.exploration == 1.4
        assert case.num_robots == 2

    def test_deeply_nested_parameters(self):
        """Test accessing deeply nested parameters."""
        case = BenchmarkCase(
            benchmark_name="test",
            case_idx=0,
            repeat_idx=0,
            params={
                "planner.heuristic.type": "ff",
                "planner.heuristic.weight": 1.5,
                "planner.timeout": 30,
            }
        )

        assert case.planner.heuristic.type == "ff"
        assert case.planner.heuristic.weight == 1.5
        assert case.planner.timeout == 30

    def test_dict_access_still_works(self):
        """Test that dictionary access still works alongside dot notation."""
        case = BenchmarkCase(
            benchmark_name="test",
            case_idx=0,
            repeat_idx=0,
            params={
                "mcts.iterations": 100,
                "num_robots": 2,
            }
        )

        # Both should work
        assert case.params["mcts.iterations"] == 100
        assert case.mcts.iterations == 100
        assert case.params["num_robots"] == 2
        assert case.num_robots == 2

    def test_missing_parameter_raises_error(self):
        """Test that accessing non-existent parameter raises AttributeError."""
        case = BenchmarkCase(
            benchmark_name="test",
            case_idx=0,
            repeat_idx=0,
            params={"num_robots": 2}
        )

        with pytest.raises(AttributeError, match="No parameter 'nonexistent'"):
            _ = case.nonexistent

    def test_missing_nested_parameter_raises_error(self):
        """Test that accessing non-existent nested parameter raises AttributeError."""
        case = BenchmarkCase(
            benchmark_name="test",
            case_idx=0,
            repeat_idx=0,
            params={"mcts.iterations": 100}
        )

        with pytest.raises(AttributeError, match="No parameter 'mcts.nonexistent'"):
            _ = case.mcts.nonexistent

    def test_mixed_parameter_names(self):
        """Test handling of mixed parameter naming styles."""
        case = BenchmarkCase(
            benchmark_name="test",
            case_idx=0,
            repeat_idx=0,
            params={
                "simple": 1,
                "dotted.param": 2,
                "deeply.nested.param": 3,
                "another.deeply.nested.one": 4,
            }
        )

        assert case.simple == 1
        assert case.dotted.param == 2
        assert case.deeply.nested.param == 3
        assert case.another.deeply.nested.one == 4

    def test_parameter_with_various_types(self):
        """Test that different parameter types work correctly."""
        case = BenchmarkCase(
            benchmark_name="test",
            case_idx=0,
            repeat_idx=0,
            params={
                "int_param": 42,
                "float_param": 3.14,
                "str_param": "hello",
                "bool_param": True,
                "list_param": [1, 2, 3],
                "dict_param": {"key": "value"},
            }
        )

        assert case.int_param == 42
        assert case.float_param == 3.14
        assert case.str_param == "hello"
        assert case.bool_param is True
        assert case.list_param == [1, 2, 3]
        assert case.dict_param == {"key": "value"}


class TestDotAccessor:
    """Test the _DotAccessor helper class."""

    def test_dot_accessor_creation(self):
        """Test creating a DotAccessor."""
        params = {"mcts.iterations": 100, "mcts.exploration": 1.4}
        accessor = _DotAccessor(params, "mcts")

        assert accessor.iterations == 100
        assert accessor.exploration == 1.4

    def test_dot_accessor_nested(self):
        """Test nested DotAccessor."""
        params = {
            "planner.heuristic.type": "ff",
            "planner.heuristic.weight": 1.5,
        }
        accessor = _DotAccessor(params, "planner")
        heuristic = accessor.heuristic

        assert heuristic.type == "ff"
        assert heuristic.weight == 1.5

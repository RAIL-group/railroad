"""
Benchmark harness for PDDL planning system.

This package provides a pytest-inspired benchmark framework with MLflow tracking,
dry-run capabilities, and parallel execution support.

Requires the 'bench' extra: pip install railroad[bench]
"""

try:
    import mlflow  # noqa: F401
    import pandas  # noqa: F401
    import dash  # noqa: F401
    import plotly  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The 'bench' extra is required for benchmarking. "
        "Install with: pip install railroad[bench]"
    ) from e

from .registry import benchmark, BenchmarkCase, Benchmark, get_all_benchmarks
from .runner import BenchmarkRunner
from .analysis import BenchmarkAnalyzer
from .plan import ExecutionPlan, Task, TaskStatus

__all__ = [
    # Registration
    "benchmark",
    "BenchmarkCase",
    "Benchmark",
    "get_all_benchmarks",
    # Execution
    "BenchmarkRunner",
    # Analysis
    "BenchmarkAnalyzer",
    # Data structures
    "ExecutionPlan",
    "Task",
    "TaskStatus",
]

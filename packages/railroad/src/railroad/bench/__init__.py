"""
Benchmark harness for PDDL planning system.

This package provides a pytest-inspired benchmark framework with MLflow tracking,
dry-run capabilities, and parallel execution support.

Requires the 'bench' extra: pip install railroad[bench]
"""

import os

# Disable MLflow's anonymous SDK/UI usage telemetry before mlflow is imported
# (it reads these at import time). setdefault so an explicit env override still
# wins; this removes the need for every user/shell to set it themselves.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")

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

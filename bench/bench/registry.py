"""
Benchmark registration and parametrization system.

Provides decorator-based API for registering benchmarks with cases.
Benchmarks are automatically registered when decorated.
"""

from dataclasses import dataclass
from typing import Dict, Any, Callable, List, Optional


# Global registry of all benchmarks (populated by @benchmark decorator)
_BENCHMARKS: List['Benchmark'] = []


def get_all_benchmarks() -> List['Benchmark']:
    """Get all registered benchmarks."""
    return _BENCHMARKS.copy()


def clear_registry():
    """Clear the benchmark registry (mainly for testing)."""
    _BENCHMARKS.clear()


@dataclass
class BenchmarkCase:
    """
    Encapsulates parameters for a single benchmark case.

    Passed to benchmark functions during execution.
    """
    benchmark_name: str
    case_idx: int
    repeat_idx: int
    params: Dict[str, Any]

    def __str__(self):
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.benchmark_name}[{self.case_idx}]({param_str}) repeat={self.repeat_idx}"


class Benchmark:
    """
    Internal representation of a registered benchmark.

    Holds the benchmark function, its metadata, and parameter cases.
    """

    def __init__(
        self,
        fn: Callable,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        timeout: float = 300.0,  # 5 minutes default
    ):
        self.fn = fn
        self.name = name
        self.description = description
        self.tags = tags or []
        self.timeout = timeout
        self.cases: List[Dict[str, Any]] = []

    def add_cases(self, cases: List[Dict[str, Any]]):
        """
        Add parameter combinations for this benchmark.

        Args:
            cases: List of parameter dictionaries. Each dict represents one case.
                   Example: [{"num_robots": 1, "seed": 42}, {"num_robots": 2, "seed": 43}]
        """
        self.cases.extend(cases)

    def __repr__(self):
        return f"Benchmark(name={self.name}, cases={len(self.cases)}, timeout={self.timeout})"


def benchmark(
    name: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    timeout: float = 300.0,
):
    """
    Decorator to register a function as a benchmark.

    Usage:
        @benchmark(
            name="my_benchmark",
            description="Description of what this benchmarks",
            tags=["planning", "multi-agent"],
            timeout=120.0,
        )
        def bench_my_test(case: BenchmarkCase):
            # Extract params
            num_robots = case.params["num_robots"]
            # ... run benchmark ...
            return {
                "success": True,
                "wall_time": 10.5,
                "cost": 42.0,
            }

        # Register parameter cases
        bench_my_test.add_cases([
            {"num_robots": 1, "seed": 42},
            {"num_robots": 2, "seed": 43},
        ])

    Args:
        name: Unique benchmark name
        description: Human-readable description
        tags: List of tags for filtering
        timeout: Timeout in seconds for each case execution

    Returns:
        Decorated function with benchmark metadata attached
    """

    def decorator(fn: Callable) -> Benchmark:
        # Create Benchmark object
        bench = Benchmark(
            fn=fn,
            name=name,
            description=description,
            tags=tags,
            timeout=timeout,
        )

        # Attach metadata to function for easy access
        fn._benchmark = bench

        # Auto-register in global registry
        _BENCHMARKS.append(bench)

        return bench

    return decorator

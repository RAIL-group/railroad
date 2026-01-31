"""
Benchmark registration and parametrization system.

Provides decorator-based API for registering benchmarks with cases.
Benchmarks are automatically registered when decorated.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Optional
import inspect
from pathlib import Path


# Global registry of all benchmarks (populated by @benchmark decorator)
_BENCHMARKS: List['Benchmark'] = []


class _DotAccessor:
    """
    Helper class for accessing nested parameters via dot notation.

    Wraps a subset of parameters that share a common prefix.
    """

    def __init__(self, params: Dict[str, Any], prefix: str = ""):
        self._params = params
        self._prefix = prefix

    def __getattr__(self, name: str):
        # Construct the full key with prefix
        if self._prefix:
            full_key = f"{self._prefix}.{name}"
        else:
            full_key = name

        # Check for direct match
        if full_key in self._params:
            return self._params[full_key]

        # Check for nested parameters (keys starting with "full_key.")
        nested_params = {}
        prefix_search = f"{full_key}."
        for key, value in self._params.items():
            if key.startswith(prefix_search):
                nested_params[key] = value

        if nested_params:
            # Return a new accessor for the nested level
            return _DotAccessor(self._params, full_key)

        raise AttributeError(f"No parameter '{full_key}'")

    def __repr__(self):
        if self._prefix:
            # Show only params with this prefix
            relevant = {k: v for k, v in self._params.items() if k.startswith(f"{self._prefix}.")}
            return f"_DotAccessor({relevant})"
        return f"_DotAccessor({self._params})"


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

    Parameters can be accessed in two ways:
    1. Dictionary style: case.params["mcts.iterations"]
    2. Dot notation: case.mcts.iterations

    Dynamic attributes can be set and retrieved via dot notation:
        case.goal = some_goal  # stored in extra, not logged
        goal = case.goal       # retrieved from extra
    """
    benchmark_name: str
    case_idx: int
    repeat_idx: int
    params: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Class-level set of dataclass field names (not stored in extra)
    _FIELDS = {'benchmark_name', 'case_idx', 'repeat_idx', 'params', 'extra'}

    def __setattr__(self, name: str, value: Any):
        """Store unknown attributes in extra dict (not logged)."""
        # Allow setting dataclass fields normally
        if name in BenchmarkCase._FIELDS:
            object.__setattr__(self, name, value)
        else:
            # Store in extra for dynamic attributes
            self.extra[name] = value

    def __getattr__(self, name: str):
        """
        Enable dot notation access to extra and params.

        Priority: extra > params > nested params
        Example: case.mcts.iterations accesses params["mcts.iterations"]
        """
        # Avoid infinite recursion by checking if we're looking for private attrs
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check extra first (dynamic attributes)
        if name in self.extra:
            return self.extra[name]

        # Try direct parameter access
        if name in self.params:
            return self.params[name]

        # Check for nested parameters (keys starting with "name.")
        prefix_search = f"{name}."
        nested_params = {}
        for key, value in self.params.items():
            if key.startswith(prefix_search):
                nested_params[key] = value

        if nested_params:
            # Return a dot accessor for nested access
            return _DotAccessor(self.params, name)

        raise AttributeError(f"No parameter '{name}' in benchmark case")

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
        repeat: int = 32,  # 32 repeats default
    ):
        self.fn = fn
        self.name = name
        self.description = description
        self.tags = tags or []
        self.timeout = timeout
        self.repeat = repeat
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
        return f"Benchmark(name={self.name}, cases={len(self.cases)}, timeout={self.timeout}, repeat={self.repeat})"


def benchmark(
    name: str,
    description: str = "",
    tags: List[str] | None = None,
    timeout: float = 300.0,
    repeat: int = 32,
) -> Callable[[Callable], Benchmark]:
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
        repeat: Number of times to repeat each case (default: 3)

    Returns:
        Decorated function with benchmark metadata attached
    """

    def decorator(fn: Callable) -> Benchmark:
        # Get the file name where the benchmark is defined
        # Use the function's code object to get its file location
        if hasattr(fn, '__code__') and hasattr(fn.__code__, 'co_filename'):
            caller_file: str = fn.__code__.co_filename  # type: ignore[assignment]
            file_name = Path(caller_file).stem
            # Format name as "file_name::benchmark_name" (pytest style)
            full_name = f"{file_name}::{name}"
        else:
            # Fallback if we can't get the file name
            full_name = name

        # Create Benchmark object
        bench = Benchmark(
            fn=fn,
            name=full_name,
            description=description,
            tags=tags,
            timeout=timeout,
            repeat=repeat,
        )

        # Attach metadata to function for easy access
        setattr(fn, '_benchmark', bench)

        # Auto-register in global registry
        _BENCHMARKS.append(bench)

        return bench

    return decorator

# Benchmark Harness

Benchmark harness for evaluating the PDDL planning system with MLflow tracking and interactive visualization.

## Quick Start

### Running Benchmarks

```bash
# Run all benchmarks
uv run benchmarks-run

# Dry run to see what will execute
uv run benchmarks-run --dry-run

# Filter by file name (matches any benchmark in that file)
uv run benchmarks-run -k movie_night --repeat-max 3 --parallel 4

# Filter by specific benchmark name
uv run benchmarks-run -k single_robot_at_location

# Filter by parameter values
uv run benchmarks-run -k "mcts.iterations=400"
uv run benchmarks-run -k "num_robots=1 or num_robots=2"

# Filter by tags
uv run benchmarks-run --tags multi-agent
```

### Viewing Results

```bash
# Launch interactive dashboard
uv run benchmarks-dashboard
```

Open http://127.0.0.1:8050/ in your browser to view benchmark results with interactive visualizations.

## Architecture

- **`bench/`**: Core benchmark harness
  - `registry.py`: Benchmark registration via `@benchmark` decorator
  - `runner.py`: Parallel execution with MLflow tracking
  - `cli.py`: Command-line interface
  - `dashboard/`: Interactive Plotly Dash visualization

- **`benchmarks/`**: Benchmark definitions
  - `basic_planning.py`: Simple planning scenarios
  - `movie_night.py`: Multi-agent coordination tasks
  - `multi_object_search.py`: Object search benchmarks

## Creating Benchmarks

Benchmarks are automatically registered with a name in the format `{file_name}::{benchmark_name}` (similar to pytest). This allows filtering by either the file name or the specific benchmark name.

```python
from bench.registry import benchmark, BenchmarkCase

@benchmark(
    name="my_benchmark",  # Will be registered as "my_file::my_benchmark"
    description="Description of what this benchmarks",
    tags=["planning", "multi-agent"],
    timeout=120.0,
    repeat=32,
)
def bench_my_test(case: BenchmarkCase):
    # Extract parameters
    num_robots = case.params["num_robots"]

    # Run benchmark
    # ...

    # Return results
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
```

## TODO

Visualization improvements:
- [ ] Case filter should look at variable names
- [ ] Axis labels should spill over into the plot to avoid line wrapping
- [ ] Display failed runs separately with red 'x' on rightmost part of plot
- [ ] Make xmax benchmark-specific instead of global

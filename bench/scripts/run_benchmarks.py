#!/usr/bin/env python3
"""
CLI entry point for running benchmarks.

Usage:
    python scripts/run_benchmarks.py --repeat-max 5 --parallel 4
    python scripts/run_benchmarks.py --dry-run
    python scripts/run_benchmarks.py -k robot --repeat-max 3
    python scripts/run_benchmarks.py --tags multi-agent
"""

import argparse
import sys
from pathlib import Path

# Add bench package to path
bench_path = Path(__file__).parent.parent
# sys.path.insert(0, str(bench_path))

from bench.runner import BenchmarkRunner
from bench.registry import get_all_benchmarks


def main():
    parser = argparse.ArgumentParser(
        description="Run PDDL planning benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks with maximum 3 repeats per case
  python scripts/run_benchmarks.py --repeat-max 3

  # Dry run to see what will execute
  python scripts/run_benchmarks.py --dry-run

  # Run with 4 parallel workers
  python scripts/run_benchmarks.py --repeat-max 5 --parallel 4

  # Filter by benchmark name
  python scripts/run_benchmarks.py -k movie_night

  # Filter by parameter values (supports and/or/not)
  python scripts/run_benchmarks.py -k "mcts_iterations=400"
  python scripts/run_benchmarks.py -k "num_robots=1 or num_robots=2"
  python scripts/run_benchmarks.py -k "movie_night and not mcts_iterations=10000"

  # Filter by tags
  python scripts/run_benchmarks.py --tags multi-agent
        """,
    )

    parser.add_argument(
        "-k",
        "--filter",
        help="Filter cases using pytest-style expressions with 'and', 'or', 'not' (e.g., 'movie_night and mcts_iterations=400', 'num_robots=1 or num_robots=2')",
    )
    parser.add_argument(
        "--repeat-max",
        type=int,
        default=None,
        help="Maximum number of repeats per case (default: None, uses benchmark's repeat setting)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Filter benchmarks by tags",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing",
    )
    parser.add_argument(
        "--mlflow-uri",
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )

    args = parser.parse_args()

    # Import all benchmarks (triggers @benchmark decorators which auto-register)
    try:
        # Add benchmarks directory to path
        benchmarks_path = bench_path / "benchmarks"
        sys.path.insert(0, str(benchmarks_path))

        # Import the benchmarks module to trigger decorator registration
        import benchmarks

    except ImportError as e:
        print(f"Error importing benchmarks: {e}")
        print("\nMake sure benchmarks/__init__.py exists.")
        sys.exit(1)

    # Get all auto-registered benchmarks
    all_benchmarks = get_all_benchmarks()

    if not all_benchmarks:
        print("Error: No benchmarks found. Make sure your benchmarks use the @benchmark decorator.")
        sys.exit(1)

    # Create runner (filtering now happens at case level in runner)
    runner = BenchmarkRunner(
        benchmarks=all_benchmarks,
        repeat_max=args.repeat_max,
        parallel=args.parallel,
        mlflow_tracking_uri=args.mlflow_uri,
        tags=args.tags,
        case_filter=args.filter,
    )

    # Create plan
    plan = runner.create_plan()

    if not plan.tasks:
        print("No tasks to run. Check your filters.")
        sys.exit(0)

    if args.dry_run:
        # Dry run
        runner.dry_run(plan)
    else:
        # Execute
        runner.run(plan)


if __name__ == "__main__":
    main()

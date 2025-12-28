"""
CLI entry point for running benchmarks.

Usage:
    uv run benchmarks-run --repeat-max 5 --parallel 4
    uv run benchmarks-run --dry-run
    uv run benchmarks-run -k robot --repeat-max 3
    uv run benchmarks-run --tags multi-agent
"""

import argparse
import sys


def main():
    """Entry point for benchmarks-run command."""
    # Import here to avoid circular imports and speed up --help
    import benchmarks  # noqa: F401 - registers benchmarks

    from bench.runner import BenchmarkRunner
    from bench.registry import get_all_benchmarks

    parser = argparse.ArgumentParser(
        description="Run PDDL planning benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks with maximum 3 repeats per case
  uv run benchmarks-run --repeat-max 3

  # Dry run to see what will execute
  uv run benchmarks-run --dry-run

  # Run with 4 parallel workers
  uv run benchmarks-run --repeat-max 5 --parallel 4

  # Filter by benchmark name
  uv run benchmarks-run -k movie_night

  # Filter by parameter values (supports and/or/not)
  uv run benchmarks-run -k "mcts_iterations=400"
  uv run benchmarks-run -k "num_robots=1 or num_robots=2"
  uv run benchmarks-run -k "movie_night and not mcts_iterations=10000"

  # Filter by tags
  uv run benchmarks-run --tags multi-agent
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
        default=None,
        help="Number of parallel workers (default: auto-detect CPU count)",
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

#!/usr/bin/env python3
"""
CLI entry point for analyzing benchmark results.

Usage:
    python scripts/analyze_results.py <experiment_name> --output ./results
    python scripts/analyze_results.py latest --output ./results
"""

import argparse
import sys
from pathlib import Path

# Add bench package to path
bench_path = Path(__file__).parent.parent
sys.path.insert(0, str(bench_path))

from bench.analysis import BenchmarkAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results from MLflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest experiment
  python scripts/analyze_results.py latest

  # Analyze specific experiment
  python scripts/analyze_results.py mrppddl_bench_1234567890

  # Export to CSV
  python scripts/analyze_results.py latest --output ./results

  # List all experiments
  python scripts/analyze_results.py --list
        """,
    )

    parser.add_argument(
        "experiment_name",
        nargs="?",
        help="MLflow experiment name (use 'latest' for most recent)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for CSV exports",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all benchmark experiments",
    )
    parser.add_argument(
        "--mlflow-uri",
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = BenchmarkAnalyzer(tracking_uri=args.mlflow_uri)

    # List experiments if requested
    if args.list:
        experiments = analyzer.list_experiments()
        if experiments.empty:
            print("No benchmark experiments found.")
        else:
            print("\nBenchmark Experiments:")
            print(experiments.to_string(index=False))
        return

    # Require experiment name if not listing
    if not args.experiment_name:
        print("Error: experiment_name required (or use --list)")
        parser.print_help()
        sys.exit(1)

    # Handle 'latest'
    if args.experiment_name == "latest":
        experiments = analyzer.list_experiments()
        if experiments.empty:
            print("Error: No experiments found.")
            sys.exit(1)

        experiment_name = experiments.iloc[0]["name"]
        print(f"Using latest experiment: {experiment_name}\n")
    else:
        experiment_name = args.experiment_name

    # Print summary to console
    try:
        analyzer.print_summary(experiment_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Export if output directory specified
    if args.output:
        try:
            analyzer.export_summary(experiment_name, args.output)
        except Exception as e:
            print(f"Error exporting summary: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

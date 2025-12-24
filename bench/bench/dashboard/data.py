"""
Data loading functions for MLflow experiments.
"""

import pandas as pd
from bench.analysis import BenchmarkAnalyzer


# Global cache for downloaded artifact paths
_artifact_cache: dict[str, str] = {}


def load_latest_experiment() -> tuple[pd.DataFrame, str]:
    """Load the most recent benchmark experiment."""
    analyzer = BenchmarkAnalyzer()
    experiments = analyzer.list_experiments()

    if experiments.empty:
        raise ValueError("No benchmark experiments found in MLflow database")

    latest_exp = experiments.iloc[0]
    print(f"Loading experiment: {latest_exp['name']}")
    print(f"Created at: {latest_exp['creation_time']}")

    df = analyzer.load_experiment(latest_exp["name"])

    if df.empty:
        raise ValueError(f"No runs found in experiment {latest_exp['name']}")

    print(f"Loaded {len(df)} runs")
    return df, latest_exp["name"]


def load_experiment_by_name(experiment_name: str) -> tuple[pd.DataFrame, str, dict]:
    """
    Load a specific experiment by name.

    Returns:
        Tuple of (dataframe, experiment_name, metadata)
    """
    analyzer = BenchmarkAnalyzer()
    df = analyzer.load_experiment(experiment_name)
    metadata = analyzer.get_experiment_metadata(experiment_name)

    print(f"Loading experiment: {experiment_name}")
    print(f"Loaded {len(df)} runs")

    return df, experiment_name, metadata


def load_all_experiments_with_summaries(limit: int = 10) -> list[dict]:
    """
    Load recent experiments with their summary statistics.

    Args:
        limit: Maximum number of experiments to load (default: 10)

    Returns:
        List of dicts with experiment info and summaries
    """
    print("Loading experiments list...")
    analyzer = BenchmarkAnalyzer()
    experiments = analyzer.list_experiments()

    if experiments.empty:
        print("No experiments found")
        return []

    # Limit to most recent experiments
    total_experiments = len(experiments)
    experiments = experiments.head(limit)

    print(f"Found {total_experiments} total experiments, loading summaries for {len(experiments)} most recent...")
    results = []
    for i, (_, exp_row) in enumerate(experiments.iterrows()):
        exp_name = exp_row["name"]
        print(f"  [{i+1}/{len(experiments)}] Loading {exp_name}...")
        try:
            summary = analyzer.get_experiment_summary(exp_name)
            metadata = analyzer.get_experiment_metadata(exp_name)

            results.append({
                "name": exp_name,
                "creation_time": exp_row["creation_time"],
                "summary": summary,
                "metadata": metadata,
            })
            print(f"    ✓ Loaded {summary['total_runs']} runs")
        except Exception as e:
            print(f"    ✗ Failed to load: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"Successfully loaded {len(results)} experiments")
    return results


def compute_per_benchmark_limits(bench_df: pd.DataFrame) -> tuple[float, float]:
    """
    Compute x-axis limits for a specific benchmark.
    Returns (xmin, xmax) based on plan_cost values in the benchmark data.
    """
    if "metrics.plan_cost" not in bench_df.columns:
        return 0.0, 1.0

    all_cost = bench_df["metrics.plan_cost"].dropna()
    if all_cost.empty:
        return 0.0, 1.0

    xmin, xmax = float(all_cost.min()), float(all_cost.max())
    return xmin, xmax

"""
Data loading functions for MLflow experiments.
"""

import pandas as pd
from railroad.bench.analysis import BenchmarkAnalyzer
from railroad.bench import compact


# Global cache for downloaded artifact paths
_artifact_cache: dict[str, str] = {}


def load_experiment_by_name(
    experiment_name: str,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, str, dict]:
    """
    Load a specific experiment by name.

    Args:
        experiment_name: Name of the experiment
        use_cache: If True, consult the compaction cache first and refresh it

    Returns:
        Tuple of (dataframe, experiment_name, metadata)
    """
    if use_cache:
        cached = compact.load(experiment_name)
        if cached is not None:
            df, metadata, _summary = cached
            print(f"Loaded experiment '{experiment_name}' from cache ({len(df)} runs)")
            return df, experiment_name, metadata

    print(f"Loading experiment: {experiment_name}")
    analyzer = BenchmarkAnalyzer()
    df = analyzer.load_experiment(experiment_name)
    metadata = analyzer.get_experiment_metadata(experiment_name)
    print(f"Loaded {len(df)} runs")

    if use_cache:
        try:
            summary = analyzer.get_experiment_summary(experiment_name, df=df)
            if compact.save(experiment_name, df, metadata, summary):
                print(f"Cached experiment '{experiment_name}'")
        except Exception as e:
            print(f"  ✗ Failed to cache: {e}")

    return df, experiment_name, metadata


def load_experiment_summary(
    experiment_name: str,
    df: pd.DataFrame | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Load just the summary for an experiment, preferring the cache.

    Falls back to computing from ``df`` (or loading runs) on a cache miss.
    """
    if use_cache:
        cached = compact.load(experiment_name)
        if cached is not None:
            _df, _meta, summary = cached
            return summary

    analyzer = BenchmarkAnalyzer()
    return analyzer.get_experiment_summary(experiment_name, df=df)


def load_all_experiments_with_summaries(
    limit: int = 10,
    use_cache: bool = True,
) -> list[dict]:
    """
    Load recent experiments with their summary statistics.

    Args:
        limit: Maximum number of experiments to load (default: 10)
        use_cache: If True, consult the compaction cache for each experiment

    Returns:
        List of dicts with experiment info and summaries
    """
    print("Loading experiments list...")
    analyzer = BenchmarkAnalyzer()
    experiments = analyzer.list_experiments()

    if experiments.empty:
        print("No experiments found")
        return []

    total_experiments = len(experiments)
    experiments = experiments.head(limit)

    print(f"Found {total_experiments} total experiments, loading summaries for {len(experiments)} most recent...")
    results = []
    for i, (_, exp_row) in enumerate(experiments.iterrows()):
        exp_name = exp_row["name"]
        print(f"  [{i+1}/{len(experiments)}] Loading {exp_name}...")

        # Cache fast-path
        if use_cache:
            cached = compact.load(exp_name)
            if cached is not None:
                _df, metadata, summary = cached
                results.append({
                    "name": exp_name,
                    "creation_time": exp_row["creation_time"],
                    "summary": summary,
                    "metadata": metadata,
                })
                print(f"    ✓ {summary['total_runs']} runs (cached)")
                continue

        try:
            metadata = analyzer.get_experiment_metadata(exp_name)
            df = analyzer.load_experiment(exp_name)
            summary = analyzer.get_experiment_summary(exp_name, df=df)

            results.append({
                "name": exp_name,
                "creation_time": exp_row["creation_time"],
                "summary": summary,
                "metadata": metadata,
            })
            if use_cache:
                compact.save(exp_name, df, metadata, summary)
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

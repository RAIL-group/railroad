"""
Analysis tools for querying and aggregating benchmark results.

Provides utilities to read MLflow experiments and compute statistics.
"""

import mlflow
import pandas as pd
from typing import Optional
from datetime import datetime


class BenchmarkAnalyzer:
    """
    Query MLflow and compute aggregate statistics.

    Provides methods to load experiments, compute aggregates, and export results.
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize analyzer.

        Args:
            tracking_uri: MLflow tracking URI (default: sqlite:///mlflow.db)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")

    def list_experiments(self) -> pd.DataFrame:
        """
        List all benchmark experiments.

        Returns:
            DataFrame with experiment names, IDs, and creation times
        """
        experiments = mlflow.search_experiments()  # type: ignore[possibly-missing-attribute]

        df = pd.DataFrame([
            {
                "name": exp.name,
                "experiment_id": exp.experiment_id,
                "creation_time": datetime.fromtimestamp(int(exp.creation_time) / 1000),
                "tags": exp.tags if exp.tags else {},
            }
            for exp in experiments
            if exp.name.startswith("mrppddl_bench_")
        ])

        if not df.empty:
            df = df.sort_values("creation_time", ascending=False)

        return df

    def get_experiment_metadata(self, experiment_name: str) -> dict:
        """
        Get experiment metadata including tags and benchmark descriptions.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary with metadata and benchmark descriptions

        Raises:
            ValueError: If experiment not found
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)  # type: ignore[possibly-missing-attribute]
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        metadata = dict(experiment.tags) if experiment.tags else {}

        # Extract benchmark descriptions from tags
        benchmark_descriptions = {}
        for key, value in list(metadata.items()):
            if key.startswith("benchmark_desc_"):
                bench_name = key.replace("benchmark_desc_", "")
                benchmark_descriptions[bench_name] = value

        metadata["benchmark_descriptions"] = benchmark_descriptions

        return metadata

    def get_experiment_summary(self, experiment_name: str) -> dict:
        """
        Get summary statistics for an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary with summary statistics including:
            - total_runs: Total number of runs
            - benchmarks: List of benchmark names
            - success_rate: Overall success rate
            - success_by_benchmark: Dict of success rates per benchmark
            - timeout_rate: Overall timeout rate

        Raises:
            ValueError: If experiment not found
        """
        df = self.load_experiment(experiment_name)

        summary = {
            "total_runs": len(df),
            "benchmarks": [],
            "success_rate": 0.0,
            "success_by_benchmark": {},
            "timeout_rate": 0.0,
        }

        if df.empty:
            return summary

        # Get unique benchmarks
        if "params.benchmark_name" in df.columns:
            summary["benchmarks"] = sorted(df["params.benchmark_name"].unique().tolist())

        # Overall success rate
        if "metrics.success" in df.columns:
            summary["success_rate"] = float(df["metrics.success"].mean())

        # Success rate by benchmark
        if "params.benchmark_name" in df.columns and "metrics.success" in df.columns:
            success_by_bench = df.groupby("params.benchmark_name")["metrics.success"].agg(
                ["mean", "count"]
            )
            summary["success_by_benchmark"] = {
                bench: {
                    "success_rate": float(row["mean"]),
                    "total_runs": int(row["count"]),
                }
                for bench, row in success_by_bench.iterrows()
            }

        # Overall timeout rate
        if "metrics.timeout" in df.columns:
            summary["timeout_rate"] = float(df["metrics.timeout"].mean())

        return summary

    def load_experiment(self, experiment_name: str) -> pd.DataFrame:
        """
        Load all runs from an experiment.

        Args:
            experiment_name: Name of the experiment to load

        Returns:
            DataFrame with parameters, metrics, and metadata

        Raises:
            ValueError: If experiment not found
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)  # type: ignore[possibly-missing-attribute]
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Search all runs except the summary run
        runs = mlflow.search_runs(  # type: ignore[possibly-missing-attribute]
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.run_name != '__summary__'",
            output_format="pandas",
        )

        assert isinstance(runs, pd.DataFrame)
        return runs

    def aggregate_by_benchmark(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate metrics by benchmark and case.

        Computes mean, std, min, max across repeats.

        Args:
            df: DataFrame from load_experiment()

        Returns:
            Aggregated DataFrame with statistics
        """
        # Group by benchmark + case
        group_cols = ["params.benchmark_name", "params.case_idx"]

        # Filter to only existing columns
        if not all(col in df.columns for col in group_cols):
            # Return empty DataFrame if required columns missing
            return pd.DataFrame()

        # Identify metric columns
        metric_cols = [c for c in df.columns if c.startswith("metrics.")]

        if not metric_cols:
            # No metrics to aggregate
            return pd.DataFrame()

        # Aggregate
        agg_funcs = ["mean", "std", "min", "max", "count"]
        aggregated = df.groupby(group_cols)[metric_cols].agg(agg_funcs)

        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]

        return aggregated.reset_index()

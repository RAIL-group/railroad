"""
Analysis tools for querying and aggregating benchmark results.

Provides utilities to read MLflow experiments and compute statistics.
"""

import mlflow
import pandas as pd
from typing import Optional
from pathlib import Path
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
        experiments = mlflow.search_experiments()

        df = pd.DataFrame([
            {
                "name": exp.name,
                "experiment_id": exp.experiment_id,
                "creation_time": datetime.fromtimestamp(int(exp.creation_time) / 1000),
            }
            for exp in experiments
            if exp.name.startswith("mrppddl_bench_")
        ])

        if not df.empty:
            df = df.sort_values("creation_time", ascending=False)

        return df

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
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Search all runs except the summary run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.run_name != '__summary__'",
        )

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

    def compute_success_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute success rate by benchmark.

        Args:
            df: DataFrame from load_experiment()

        Returns:
            DataFrame with success rates
        """
        if "params.benchmark_name" not in df.columns or "metrics.success" not in df.columns:
            return pd.DataFrame()

        success_rate = df.groupby("params.benchmark_name").agg({
            "metrics.success": ["mean", "count"]
        })

        success_rate.columns = ["success_rate", "total_runs"]

        return success_rate.reset_index()

    def export_summary(self, experiment_name: str, output_path: Path):
        """
        Export comprehensive summary to CSV files.

        Creates:
        - aggregated_metrics.csv: Mean/std/min/max by benchmark and case
        - success_rates.csv: Success rates by benchmark

        Args:
            experiment_name: Experiment to analyze
            output_path: Output directory path
        """
        # Load experiment
        df = self.load_experiment(experiment_name)

        if df.empty:
            print(f"No runs found in experiment '{experiment_name}'")
            return

        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Aggregate metrics
        aggregated = self.aggregate_by_benchmark(df)
        if not aggregated.empty:
            agg_file = output_path / "aggregated_metrics.csv"
            aggregated.to_csv(agg_file, index=False)
            print(f"Saved aggregated metrics to {agg_file}")

        # Success rates
        success_rates = self.compute_success_rate(df)
        if not success_rates.empty:
            success_file = output_path / "success_rates.csv"
            success_rates.to_csv(success_file, index=False)
            print(f"Saved success rates to {success_file}")

        print(f"\nSummary exported to {output_path}/")

    def print_summary(self, experiment_name: str):
        """
        Print summary statistics to console.

        Args:
            experiment_name: Experiment to analyze
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Load experiment
        df = self.load_experiment(experiment_name)

        if df.empty:
            console.print(f"[red]No runs found in experiment '{experiment_name}'[/red]")
            return

        console.print(f"\n[bold cyan]Experiment: {experiment_name}[/bold cyan]")
        console.print(f"Total runs: {len(df)}\n")

        # Success rates
        success_rates = self.compute_success_rate(df)
        if not success_rates.empty:
            console.print("[bold]Success Rates by Benchmark[/bold]")
            table = Table()
            table.add_column("Benchmark", style="cyan")
            table.add_column("Success Rate", justify="right")
            table.add_column("Total Runs", justify="right")

            for _, row in success_rates.iterrows():
                table.add_row(
                    str(row["params.benchmark_name"]),
                    f"{row['success_rate']:.2%}",
                    str(int(row['total_runs'])),
                )

            console.print(table)
            console.print()

        # Aggregated metrics (show first few rows)
        aggregated = self.aggregate_by_benchmark(df)
        if not aggregated.empty:
            console.print("[bold]Aggregated Metrics (first 10 rows)[/bold]")
            console.print(aggregated.head(10).to_string())
            console.print()

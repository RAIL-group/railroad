"""
Summarise, per benchmark case, how many experiments required at least one
search retry (the solver failed and fell back to a smaller sub-task).

Reads the ``required_search_retry`` metric that ``run_experiment`` logs for each
run of the interruption benchmarks. Runs that never produced a result (timeouts,
errors) have no such metric, so the count is reported against the runs that did.

Usage:
    uv run python scripts/summarize_search_retries.py
    uv run python scripts/summarize_search_retries.py railroad_bench_1790693927
    uv run python scripts/summarize_search_retries.py --tracking-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import argparse

import pandas as pd

from railroad.bench.analysis import BenchmarkAnalyzer

RETRY_METRIC = "metrics.required_search_retry"
# parameter that labels a case, newest name first
CASE_LABEL_PARAMS = ["params.arrival_exceedance", "params.time_between_arrivals"]


def summarize_search_retries(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (benchmark, case): runs, runs with a result, runs with a retry."""
    if RETRY_METRIC not in df.columns:
        df = df.assign(**{RETRY_METRIC: float("nan")})
    label = next((col for col in CASE_LABEL_PARAMS if col in df.columns), None)

    rows = []
    for _, case in df.groupby(["params.benchmark_name", "params.case_idx"]):
        reported = case[RETRY_METRIC].dropna()
        row = {
            "benchmark": case["params.benchmark_name"].iloc[0],
            "case": int(case["params.case_idx"].iloc[0]),
        }
        if label is not None:
            row[label.removeprefix("params.")] = case[label].iloc[0]
        row.update({
            "runs": len(case),
            "reported": len(reported),
            "with_retry": int((reported > 0).sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["benchmark", "case"], ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", nargs="?", default=None,
                   help="MLflow experiment name (default: the most recent benchmark experiment)")
    p.add_argument("--tracking-uri", default=None,
                   help="MLflow tracking URI (default: sqlite:///mlflow.db)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = BenchmarkAnalyzer(args.tracking_uri)

    experiment = args.experiment
    if experiment is None:
        experiments = analyzer.list_experiments()
        if experiments.empty:
            raise SystemExit("no benchmark experiments found")
        experiment = experiments.iloc[0]["name"]

    summary = summarize_search_retries(analyzer.load_experiment(experiment))
    print(f"experiment: {experiment}\n")
    print(summary.to_string(index=False))
    print("\nwith_retry: runs that required at least one search retry, out of the "
          "'reported' runs that logged a result")
    if summary["reported"].sum() == 0:
        print("no run in this experiment logged required_search_retry "
              "(it predates the metric, or isn't an interruption benchmark)")


if __name__ == "__main__":
    main()

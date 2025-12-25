"""
Helper functions for data transformation and formatting.
"""

import pandas as pd
from dash import html, dcc

from .styles import (
    COLOR_SUCCESS,
    COLOR_FAILURE,
    COLOR_ERROR,
    COLOR_TIMEOUT,
    SYMBOL_SUCCESS,
    SYMBOL_FAILURE,
    SYMBOL_ERROR,
    SYMBOL_TIMEOUT,
    CATPPUCCIN_SAPPHIRE,
    CATPPUCCIN_YELLOW,
)


def _bool_mask(series: pd.Series) -> pd.Series:
    """
    Robustly interpret a metric series as a boolean mask.
    Handles bool, 0/1, floats, and missing values.
    """
    if series is None:
        return pd.Series(False)
    s = series.fillna(0)
    try:
        s = s.astype(float)
    except Exception:
        s = s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
        return s
    return s > 0.5


def _success_mask(series: pd.Series) -> pd.Series:
    """Interpret metrics.success as a boolean mask."""
    return _bool_mask(series)


def _timeout_mask(series: pd.Series) -> pd.Series:
    """Interpret metrics.timeout as a boolean mask."""
    return _bool_mask(series)


def format_status_count(n_success: int, n_total: int, n_error: int = 0, n_timeout: int = 0) -> str:
    """
    Format success/failure counts with emoji symbols matching progress.py.
    Returns HTML-formatted string like: "<span style='color:green'>✓5</span>/10"
    """
    n_failure = n_total - n_success - n_error - n_timeout
    parts = []

    if n_success > 0:
        parts.append(f"<span style='color:{COLOR_SUCCESS}; font-weight:bold;'>{SYMBOL_SUCCESS}{n_success}</span>")
    if n_error > 0:
        parts.append(f"<span style='color:{COLOR_ERROR}; font-weight:bold;'>{SYMBOL_ERROR}{n_error}</span>")
    if n_failure > 0:
        parts.append(f"<span style='color:{COLOR_FAILURE}; font-weight:bold;'>{SYMBOL_FAILURE}{n_failure}</span>")
    if n_timeout > 0:
        parts.append(f"<span style='color:{COLOR_TIMEOUT}; font-weight:bold;'>{SYMBOL_TIMEOUT}{n_timeout}</span>")

    if not parts:
        return f"0/{n_total}"

    return "/".join(parts) + f"/{n_total}"


def compute_case_summary_stats(case_data: pd.DataFrame) -> dict:
    """
    Compute summary statistics for a case.
    Returns dict with avg_plan_cost, success_rate, avg_wall_time.
    """
    stats = {
        "avg_plan_cost": None,
        "success_rate": None,
        "avg_wall_time": None,
    }

    if "metrics.plan_cost" in case_data.columns:
        plan_costs = case_data["metrics.plan_cost"].dropna()
        if len(plan_costs) > 0:
            stats["avg_plan_cost"] = float(plan_costs.mean())

    if "metrics.success" in case_data.columns:
        success_mask = _success_mask(case_data["metrics.success"])
        stats["success_rate"] = float(success_mask.mean())

    if "metrics.wall_time" in case_data.columns:
        wall_times = case_data["metrics.wall_time"].dropna()
        if len(wall_times) > 0:
            stats["avg_wall_time"] = float(wall_times.mean())

    return stats


def format_case_params(params: dict) -> str:
    """Format case parameters with colored HTML (matching progress.py style)."""
    parts = [
        f"<span style='color:{CATPPUCCIN_SAPPHIRE}'>{k}</span>=<span style='color:{CATPPUCCIN_YELLOW}'>{v}</span>"
        for k, v in params.items() if v is not None
    ]
    return ", ".join(parts)


def build_benchmark_summary_line(bench_name: str, bench_stats: dict, description: str = "") -> html.Pre:
    """
    Build a single benchmark summary line with status symbols.

    Args:
        bench_name: Name of the benchmark
        bench_stats: Dict with success_rate and total_runs
        description: Optional description of the benchmark

    Returns:
        html.Pre element with formatted benchmark line
    """
    success_rate = bench_stats.get("success_rate", 0.0)
    total_runs = bench_stats.get("total_runs", 0)

    # Format success count with colored symbols
    n_success = int(success_rate * total_runs)
    n_total = total_runs
    n_failure = n_total - n_success

    # Build status string with colored symbols
    status_parts = []
    if n_success > 0:
        status_parts.append(html.Span(f"{SYMBOL_SUCCESS}{n_success}", className="status-success"))
        status_parts.append("/")
    if n_failure > 0:
        status_parts.append(html.Span(f"{SYMBOL_FAILURE}{n_failure}", className="status-failure"))
        status_parts.append("/")

    # Benchmark line
    return html.Pre([
        f"  {bench_name}: ",
        *status_parts,
        f"{n_total} ",
        html.Span(f"({success_rate:.1%})", className="text-dimmed"),
    ], className="pre-text text-base")


def build_experiment_summary_block(exp_name: str, summary: dict, metadata: dict, creation_time=None, include_link: bool = False) -> list:
    """
    Build a complete experiment summary block with benchmarks.

    Args:
        exp_name: Experiment name
        summary: Summary dict with total_runs, success_rate, benchmarks, success_by_benchmark
        metadata: Metadata dict with benchmark_descriptions
        creation_time: Optional creation time
        include_link: If True, make experiment name a clickable link

    Returns:
        List of Dash components forming the summary block
    """
    benchmark_descriptions = metadata.get("benchmark_descriptions", {})
    children = []

    # Experiment name
    if include_link:
        children.append(html.Div([
            html.Span("# ", className="pre-text text-base"),
            dcc.Link(exp_name, href=f"/experiment/{exp_name}", className="link"),
        ], className="text-base"))
    else:
        children.append(html.Pre(
            f"# {exp_name}",
            className="pre-text text-base text-bold"
        ))

    # Creation time (if provided)
    if creation_time:
        creation_time_str = creation_time.strftime("%Y-%m-%d %H:%M:%S")
        children.append(html.Pre(
            f"  Created: {creation_time_str}",
            className="pre-text text-base text-dimmed"
        ))

    # Total runs and success rate
    success_rate_class = "status-success" if summary['success_rate'] > 0.8 else ("status-error" if summary['success_rate'] > 0.5 else "status-failure")
    children.append(html.Pre([
        "  Total runs: ",
        html.Span(f"{summary['total_runs']}", style={"color": CATPPUCCIN_SAPPHIRE}),
        " | Success rate: ",
        html.Span(f"{summary['success_rate']:.1%}", className=success_rate_class),
    ], className="pre-text text-base"))

    # Benchmarks
    if summary.get("benchmarks"):
        children.append(html.Pre(
            "  Benchmarks:",
            className="pre-text text-base"
        ))

        for bench_name in summary["benchmarks"]:
            bench_stats = summary["success_by_benchmark"].get(bench_name, {})
            description = benchmark_descriptions.get(bench_name, "")

            # Benchmark line
            children.append(build_benchmark_summary_line(bench_name, bench_stats, description))

            # Description if available
            if description:
                children.append(html.Pre(
                    f"      {description}",
                    className="pre-text text-base text-secondary"
                ))

    return children

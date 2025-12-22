#!/usr/bin/env python3
"""
Interactive Plotly Dash dashboard for benchmark results visualization.

Usage:
    uv run bench/scripts/dashboard.py
"""

import dash
from dash import dcc, html, Input, Output, State, ALL, callback_context
import dash_bootstrap_components as dbc
from flask import send_file
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import time
import json

from bench.analysis import BenchmarkAnalyzer


# Global cache for downloaded artifact paths
_artifact_cache: dict[str, str] = {}


# Needed to opt-in to new behavior and suppress warnings
pd.set_option("future.no_silent_downcasting", True)


# =============================================================================
# STYLING CONSTANTS - Catppuccin Mocha Theme
# =============================================================================

# Font Configuration
FONT_FAMILY = "Iosevka, monospace"
FONT_SIZE = 12
FONT_SIZE_TITLE = 16

# Catppuccin Mocha Palette
CATPPUCCIN_BASE = "#1e1e2e"        # Main background
CATPPUCCIN_MANTLE = "#181825"      # Darker background
CATPPUCCIN_CRUST = "#11111b"       # Darkest background
CATPPUCCIN_TEXT = "#cdd6f4"        # Main text
CATPPUCCIN_SUBTEXT1 = "#bac2de"    # Secondary text
CATPPUCCIN_SUBTEXT0 = "#a6adc8"    # Tertiary text
CATPPUCCIN_OVERLAY0 = "#6c7086"    # Overlay/dimmed
CATPPUCCIN_SURFACE0 = "#313244"    # Surface
CATPPUCCIN_SURFACE1 = "#45475a"    # Surface raised

# Catppuccin Accent Colors
CATPPUCCIN_GREEN = "#a6e3a1"       # Success
CATPPUCCIN_RED = "#f38ba8"         # Failure
CATPPUCCIN_PEACH = "#fab387"       # Error/Warning
CATPPUCCIN_YELLOW = "#f9e2af"      # Timeout
CATPPUCCIN_BLUE = "#89b4fa"        # Links/Info
CATPPUCCIN_SAPPHIRE = "#74c7ec"    # Accents
CATPPUCCIN_MAUVE = "#cba6f7"       # Highlights

# Status Colors
COLOR_SUCCESS = CATPPUCCIN_GREEN
COLOR_FAILURE = CATPPUCCIN_RED
COLOR_ERROR = CATPPUCCIN_PEACH
COLOR_TIMEOUT = CATPPUCCIN_YELLOW

# Status Symbols (matching progress.py)
SYMBOL_SUCCESS = "\u2713"      # checkmark
SYMBOL_FAILURE = "\u2717"      # x
SYMBOL_ERROR = "\u26a0"        # warning
SYMBOL_TIMEOUT = "\u23f1"      # stopwatch

# Violin Plot Styling
VIOLIN_FILL = f"rgba(137, 180, 250, 0.1)"  # Blue with transparency
VIOLIN_LINE = f"rgba(137, 180, 250, 0.5)"  # Blue
VIOLIN_LINE_WIDTH = 0.35
VIOLIN_WIDTH = 0.25

# Data Point Markers
POINT_SIZE = 5
POINT_COLOR = f"rgba(205, 214, 244, 0.4)"  # Text color with transparency
POINT_OUTLINE = f"rgba(49, 50, 68, 0.8)"   # Surface0
POINT_OUTLINE_WIDTH = 0.5

# Mean Marker
MEAN_MARKER_SIZE = 7
MEAN_MARKER_COLOR = CATPPUCCIN_SAPPHIRE
MEAN_MARKER_WIDTH = 2

# Failed Run Markers
FAILED_MARKER_SYMBOL = "x-thin"
FAILED_MARKER_SIZE = 5
FAILED_MARKER_WIDTH = 1
FAILED_MARKER_COLOR = f"rgba(243, 139, 168, 0.8)"  # Red with transparency

# Layout & Spacing
MARGIN = dict(l=0, r=10, t=0, b=0)
HEIGHT_PER_CASE = 40
MIN_PLOT_HEIGHT = 80
ANNOTATION_PADDING = 0.5
ANNOTATION_X_OFFSET = 0.3  # Fraction of dx to offset annotations

# Grid Styling
GRID_COLOR = f"rgba(108, 112, 134, 0.2)"  # Overlay0 with transparency
GRID_WIDTH = 0.5
PLOT_BGCOLOR = CATPPUCCIN_BASE
PAPER_BGCOLOR = CATPPUCCIN_BASE
# ANNOTATION_BGCOLOR = f"rgba(24, 24, 37, 0.80)"  # Surface0 with slight transparency
ANNOTATION_BGCOLOR = PLOT_BGCOLOR

# Text Colors
TEXT_COLOR = CATPPUCCIN_TEXT
TEXT_DIMMED = CATPPUCCIN_SUBTEXT0
TEXT_SECONDARY = CATPPUCCIN_OVERLAY0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _success_mask(series: pd.Series) -> pd.Series:
    """
    Robustly interpret metrics.success as a boolean mask.
    Handles bool, 0/1, floats, and missing.
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


def _timeout_mask(series: pd.Series) -> pd.Series:
    """
    Robustly interpret metrics.timeout as a boolean mask.
    Handles bool, 0/1, floats, and missing.
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
        status_parts.append(html.Span(f"{SYMBOL_SUCCESS}{n_success}", style={"color": COLOR_SUCCESS}))
        status_parts.append("/")
    if n_failure > 0:
        status_parts.append(html.Span(f"{SYMBOL_FAILURE}{n_failure}", style={"color": COLOR_FAILURE}))
        status_parts.append("/")

    # Benchmark line
    return html.Pre([
        f"  {bench_name}: ",
        *status_parts,
        f"{n_total} ",
        html.Span(f"({success_rate:.1%})", style={"color": TEXT_DIMMED}),
    ], style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_COLOR})


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
            html.Span("# ", style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "color": TEXT_COLOR}),
            dcc.Link(
                exp_name,
                href=f"/experiment/{exp_name}",
                style={
                    "fontFamily": FONT_FAMILY,
                    "fontSize": f"{FONT_SIZE}px",
                    "textDecoration": "underline",
                    "color": CATPPUCCIN_BLUE,
                },
            ),
        ], style={"margin": "0", "padding": "0"}))
    else:
        children.append(html.Pre(
            f"# {exp_name}",
            style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_COLOR, "fontWeight": "bold"}
        ))

    # Creation time (if provided)
    if creation_time:
        creation_time_str = creation_time.strftime("%Y-%m-%d %H:%M:%S")
        children.append(html.Pre(
            f"  Created: {creation_time_str}",
            style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_DIMMED}
        ))

    # Total runs and success rate
    success_rate_color = COLOR_SUCCESS if summary['success_rate'] > 0.8 else (COLOR_ERROR if summary['success_rate'] > 0.5 else COLOR_FAILURE)
    children.append(html.Pre([
        "  Total runs: ",
        html.Span(f"{summary['total_runs']}", style={"color": CATPPUCCIN_SAPPHIRE}),
        " | Success rate: ",
        html.Span(f"{summary['success_rate']:.1%}", style={"color": success_rate_color}),
    ], style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_COLOR}))

    # Benchmarks
    if summary.get("benchmarks"):
        children.append(html.Pre(
            "  Benchmarks:",
            style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_COLOR}
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
                    style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_SECONDARY, "fontStyle": "italic"}
                ))

    return children


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

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


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_violin_trace(
    plan_costs: pd.Series,
    case_label: str,
    run_ids: pd.Series,
    dx: float = 0.0,
    trace_type: str = "success",
) -> go.Violin:
    """
    Create a violin trace with customizable appearance for different run types.

    Args:
        plan_costs: Series of plan_cost values
        case_label: Y-axis label (e.g., "Case 5")
        run_ids: Series of run_id values for click handling
        dx: Range of x values for jitter scaling (used for failed/timeout traces)
        trace_type: Type of trace - "success", "failed", or "timeout"

    Returns:
        Configured go.Violin trace
    """
    # Default parameters for success traces
    config = {
        "fillcolor": VIOLIN_FILL,
        "line_color": VIOLIN_LINE,
        "line_width": VIOLIN_LINE_WIDTH,
        "violin_width": VIOLIN_WIDTH,
        "meanline_visible": True,
        "marker_symbol": "circle",
        "marker_size": POINT_SIZE,
        "marker_color": POINT_COLOR,
        "marker_line_color": POINT_OUTLINE,
        "marker_line_width": POINT_OUTLINE_WIDTH,
        "hovertemplate": "Plan cost: %{x}<br>Run ID: %{customdata}<br>Click for log<extra></extra>",
        "name_suffix": "",
        "apply_jitter": False,
    }

    # Customize based on trace type
    if trace_type == "failed":
        config.update({
            "fillcolor": "rgba(0,0,0,0)",
            "line_color": "rgba(0,0,0,0)",
            "line_width": 0,
            "violin_width": VIOLIN_WIDTH,
            "meanline_visible": False,
            "marker_symbol": FAILED_MARKER_SYMBOL,
            "marker_size": FAILED_MARKER_SIZE,
            "marker_color": None,  # No fill, line only
            "marker_line_color": FAILED_MARKER_COLOR,
            "marker_line_width": FAILED_MARKER_WIDTH,
            "hovertemplate": "FAILED<br>Plan cost: %{x}<br>Run ID: %{customdata}<br>Click for log<extra></extra>",
            "name_suffix": " failed",
            "apply_jitter": True,
        })
    elif trace_type == "timeout":
        config.update({
            "fillcolor": "rgba(0,0,0,0)",
            "line_color": "rgba(0,0,0,0)",
            "line_width": 0,
            "violin_width": VIOLIN_WIDTH,
            "meanline_visible": False,
            "marker_symbol": "diamond",
            "marker_size": POINT_SIZE + 1,
            "marker_color": "rgba(255, 215, 0, 0.8)",
            "marker_line_color": COLOR_TIMEOUT,
            "marker_line_width": 1,
            "hovertemplate": "TIMEOUT ⏱<br>Plan cost: %{x}<br>Run ID: %{customdata}<br>Click for log<extra></extra>",
            "name_suffix": " timeout",
            "apply_jitter": True,
        })

    # Apply jitter if needed
    x_values = plan_costs.copy()
    if config["apply_jitter"] and dx > 0:
        x_values = plan_costs + np.random.uniform(-0.01 * dx, 0.01 * dx, size=len(plan_costs))

    # Build marker dict
    marker_dict = {
        "symbol": config["marker_symbol"],
        "size": config["marker_size"],
        "line": dict(color=config["marker_line_color"], width=config["marker_line_width"]),
    }
    if config["marker_color"] is not None:
        marker_dict["color"] = config["marker_color"]

    return go.Violin(
        y=[case_label] * len(plan_costs),
        x=x_values,
        customdata=run_ids.tolist(),
        name=case_label + config["name_suffix"],
        orientation="h",
        fillcolor=config["fillcolor"],
        line=dict(color=config["line_color"], width=config["line_width"]),
        width=config["violin_width"],
        box_visible=False,
        meanline_visible=config["meanline_visible"],
        points="all",
        pointpos=0,
        jitter=0.5,
        marker=marker_dict,
        hoveron="points",
        hovertemplate=config["hovertemplate"],
        showlegend=False,
    )


def create_mean_marker(mean_val: float, case_label: str) -> go.Scatter:
    """
    Create the mean marker trace.

    Args:
        mean_val: Mean value to mark
        case_label: Y-axis label

    Returns:
        Configured go.Scatter trace for mean marker
    """
    return go.Scatter(
        x=[mean_val],
        y=[case_label],
        mode="markers",
        marker=dict(
            symbol="line-ns",
            size=MEAN_MARKER_SIZE,
            color="rgba(255, 255, 255, 0)",
            line=dict(color=MEAN_MARKER_COLOR, width=MEAN_MARKER_WIDTH),
        ),
        showlegend=False,
        hoverinfo="skip",
    )


def create_case_annotation(
    case_idx: int,
    params: dict,
    n_success: int,
    n_total: int,
    summary_stats: dict,
    x_position: float,
    case_label: str,
    case_width: int = 2,
    n_error: int = 0,
    n_timeout: int = 0,
) -> dict:
    """
    Create annotation with hover showing summary statistics.

    Args:
        case_idx: Case index number
        params: Dictionary of case parameters
        n_success: Number of successful runs
        n_total: Total number of runs
        summary_stats: Dict with avg_plan_cost, success_rate, avg_wall_time
        x_position: X coordinate for annotation
        case_label: Y coordinate reference
        case_width: Width for case index formatting
        n_error: Number of error runs
        n_timeout: Number of timeout runs

    Returns:
        Plotly annotation dict
    """
    # Format status count
    status_html = format_status_count(n_success, n_total, n_error, n_timeout)

    # Format parameters with colors
    param_str_colored = format_case_params(params)
    param_str = f"   {status_html} Case {case_idx:{case_width}d}: " + param_str_colored

    # Build hover text with summary statistics
    hover_parts = ["<b>Summary Statistics</b>"]
    if summary_stats["avg_plan_cost"] is not None:
        hover_parts.append(f"Avg Plan Cost: {summary_stats['avg_plan_cost']:.2f}")
    if summary_stats["success_rate"] is not None:
        hover_parts.append(f"Success Rate: {summary_stats['success_rate']:.1%}")
    if summary_stats["avg_wall_time"] is not None:
        hover_parts.append(f"Avg Wall Time: {summary_stats['avg_wall_time']:.2f}s")

    hover_text = "<br>".join(hover_parts)

    return dict(
        x=x_position,
        y=case_label,
        xref="x",
        yref="y",
        text=param_str,
        hovertext=hover_text,
        hoverlabel=dict(
            font=dict(family=FONT_FAMILY, size=FONT_SIZE, color=TEXT_COLOR),
            bgcolor=CATPPUCCIN_SURFACE0,
        ),
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        yshift=4,
        font=dict(size=FONT_SIZE, family=FONT_FAMILY, color=TEXT_COLOR),
        bgcolor=ANNOTATION_BGCOLOR,
        borderpad=ANNOTATION_PADDING,
    )


def create_benchmark_figure(benchmark: str, bench_df: pd.DataFrame) -> go.Figure:
    """
    Create a complete figure for one benchmark.

    Args:
        benchmark: Benchmark name
        bench_df: DataFrame filtered for this benchmark

    Returns:
        Configured go.Figure
    """
    bench_df = bench_df.copy()

    if "case_id" not in bench_df.columns:
        bench_df["case_id"] = (
            bench_df["params.benchmark_name"] + "_case_" + bench_df["params.case_idx"].astype(str)
        )

    # Convert case_idx to int for proper sorting
    bench_df["case_idx_int"] = bench_df["params.case_idx"].astype(int)

    # Get unique cases sorted by integer case index (reversed for low-to-high display)
    unique_cases = bench_df.groupby("case_idx_int").first().sort_index()
    case_order = unique_cases.index.tolist()[::-1]
    case_labels = [f"Case {i}" for i in case_order]

    # Compute per-benchmark limits
    xmin, xmax = compute_per_benchmark_limits(bench_df)
    dx = max(xmax - xmin, 1.0)

    # Place annotation block to the left of xmin
    annote_xloc = 0.95 * xmin - ANNOTATION_X_OFFSET * dx

    # Calculate width for case index formatting
    max_case_idx = max(case_order) if case_order else 0
    case_width = len(str(max_case_idx))

    fig = go.Figure()
    annotations = []

    for case_idx in case_order:
        case_label = f"Case {case_idx}"
        case_data = bench_df[bench_df["case_idx_int"] == case_idx].copy()

        # Determine success, timeout, and failure masks
        if "metrics.success" in case_data.columns:
            success_mask = _success_mask(case_data["metrics.success"])
        else:
            success_mask = pd.Series([True] * len(case_data), index=case_data.index)

        if "metrics.timeout" in case_data.columns:
            timeout_mask = _timeout_mask(case_data["metrics.timeout"])
        else:
            timeout_mask = pd.Series([False] * len(case_data), index=case_data.index)

        # Separate data: success, timeout, and other failures
        success_data = case_data[success_mask].copy()
        timeout_data = case_data[~success_mask & timeout_mask].copy()
        failed_data = case_data[~success_mask & ~timeout_mask].copy()

        # Fill NaN plan_cost with xmax for display
        if "metrics.plan_cost" in success_data.columns:
            success_data["metrics.plan_cost"] = success_data["metrics.plan_cost"].fillna(xmax)
        if "metrics.plan_cost" in timeout_data.columns:
            timeout_data["metrics.plan_cost"] = timeout_data["metrics.plan_cost"].fillna(xmax)
        if "metrics.plan_cost" in failed_data.columns:
            failed_data["metrics.plan_cost"] = failed_data["metrics.plan_cost"].fillna(xmax)

        # Violin trace for successful runs
        if not success_data.empty and "metrics.plan_cost" in success_data.columns:
            run_ids = success_data["run_id"] if "run_id" in success_data.columns else pd.Series([""] * len(success_data))
            fig.add_trace(create_violin_trace(
                success_data["metrics.plan_cost"],
                case_label,
                run_ids,
                dx=dx,
                trace_type="success",
            ))

            # Mean marker
            case_mean = float(success_data["metrics.plan_cost"].mean())
            fig.add_trace(create_mean_marker(case_mean, case_label))

        # Timeout runs as diamond markers
        if not timeout_data.empty and "metrics.plan_cost" in timeout_data.columns:
            run_ids = timeout_data["run_id"] if "run_id" in timeout_data.columns else pd.Series([""] * len(timeout_data))
            fig.add_trace(create_violin_trace(
                timeout_data["metrics.plan_cost"],
                case_label,
                run_ids,
                dx=dx,
                trace_type="timeout",
            ))

        # Failed runs as X markers
        if not failed_data.empty and "metrics.plan_cost" in failed_data.columns:
            run_ids = failed_data["run_id"] if "run_id" in failed_data.columns else pd.Series([""] * len(failed_data))
            fig.add_trace(create_violin_trace(
                failed_data["metrics.plan_cost"],
                case_label,
                run_ids,
                dx=dx,
                trace_type="failed",
            ))

        # Get case parameters (from first row)
        case_row = case_data.iloc[0]
        params = {}
        for col in bench_df.columns:
            if col.startswith("params.") and col not in [
                "params.benchmark_name",
                "params.case_idx",
                "params.repeat_idx",
            ]:
                param_name = col.replace("params.", "")
                params[param_name] = case_row[col]

        # Compute summary stats
        summary_stats = compute_case_summary_stats(case_data)

        # Create annotation
        n_total = len(case_data)
        n_success = int(success_mask.sum())
        n_timeout = int(timeout_mask.sum())
        n_error = 0  # Could be computed from error field if available

        annotations.append(create_case_annotation(
            case_idx=case_idx,
            params=params,
            n_success=n_success,
            n_total=n_total,
            n_error=n_error,
            n_timeout=n_timeout,
            summary_stats=summary_stats,
            x_position=annote_xloc,
            case_label=case_label,
            case_width=case_width,
        ))

    # Calculate height
    height = max(MIN_PLOT_HEIGHT, len(case_order) * HEIGHT_PER_CASE)

    # Update layout
    fig.update_layout(
        xaxis_title="",
        height=height,
        margin=MARGIN,
        annotations=annotations,
        font=dict(size=FONT_SIZE, family=FONT_FAMILY, color=TEXT_COLOR),
        plot_bgcolor=PLOT_BGCOLOR,
        paper_bgcolor=PAPER_BGCOLOR,
    )

    # Force the y categories so cases with only failed points still show up
    fig.update_yaxes(
        range=[-0.3, len(case_order)-0.3],
        categoryorder="array",
        categoryarray=case_labels,
        ticks="",
        showticklabels=False,
        color=TEXT_COLOR,
    )

    # Set x-axis range
    fig.update_xaxes(
        range=[annote_xloc, (xmax * 1.05) if xmax != 0 else 1],
        gridcolor=GRID_COLOR,
        gridwidth=GRID_WIDTH,
        color=TEXT_COLOR,
    )

    return fig


def create_violin_plots_by_benchmark(df: pd.DataFrame) -> list[dict]:
    """
    Create figures for all benchmarks.

    Args:
        df: DataFrame with all benchmark data

    Returns:
        List of dicts with 'benchmark' and 'figure' keys
    """
    if "metrics.plan_cost" not in df.columns:
        return []

    benchmarks = sorted(df["params.benchmark_name"].unique())
    figures = []

    for benchmark in benchmarks:
        bench_df = df[df["params.benchmark_name"] == benchmark].copy()
        fig = create_benchmark_figure(benchmark, bench_df)
        figures.append({"benchmark": benchmark, "figure": fig})

    return figures


# =============================================================================
# LAYOUT FUNCTIONS
# =============================================================================

def create_log_modal() -> dbc.Modal:
    """Create the modal component for displaying log.html artifacts."""
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle(id="modal-title", style={"fontFamily": FONT_FAMILY}),
            ),
            dbc.ModalBody(
                html.Div(id="modal-body-content"),
                style={"padding": "0"},
            ),
        ],
        id="log-modal",
        size="xl",
        is_open=False,
    )


def create_main_layout() -> html.Div:
    """Create the main app layout with refresh capability."""
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="data-store"),
            dcc.Loading(
                id="loading",
                type="default",
                children=html.Div(id="main-content"),
            ),
            create_log_modal(),
        ],
        style={
            "padding": "20px",
            "fontFamily": FONT_FAMILY,
            "backgroundColor": CATPPUCCIN_BASE,
            "color": TEXT_COLOR,
            "minHeight": "100vh",
        },
    )


def build_content_layout(experiment_name: str, figures: list[dict], df: pd.DataFrame, metadata: dict = None, summary: dict = None) -> list:
    """
    Build the content layout with all graphs and data sample.

    Args:
        experiment_name: Name of the experiment
        figures: List of figure dicts from create_violin_plots_by_benchmark
        df: DataFrame for raw data sample
        metadata: Optional experiment metadata with benchmark descriptions
        summary: Optional experiment summary with stats

    Returns:
        List of Dash components
    """
    children = [
        dcc.Link("← Back to Experiment List", href="/", style={
            "fontFamily": FONT_FAMILY,
            "fontSize": f"{FONT_SIZE}px",
            "marginBottom": "10px",
            "display": "block",
            "color": CATPPUCCIN_BLUE,
        }),
        html.Br(),
    ]

    # Add experiment summary if available
    if summary and metadata:
        # Build summary with clickable benchmark links
        summary_children = []
        benchmark_descriptions = metadata.get("benchmark_descriptions", {})

        # Experiment name
        summary_children.append(html.Pre(
            f"# Benchmark Results: {experiment_name}",
            style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_COLOR, "fontWeight": "bold",                         "textDecoration": "underline",
}
        ))

        # Total runs and success rate
        success_rate_color = COLOR_SUCCESS if summary['success_rate'] > 0.8 else (COLOR_ERROR if summary['success_rate'] > 0.5 else COLOR_FAILURE)
        summary_children.append(html.Pre([
            "  Total runs: ",
            html.Span(f"{summary['total_runs']}", style={"color": CATPPUCCIN_SAPPHIRE}),
            " | Success rate: ",
            html.Span(f"{summary['success_rate']:.1%}", style={"color": success_rate_color}),
        ], style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_COLOR}))

        # Benchmarks with clickable links
        if summary.get("benchmarks"):
            summary_children.append(html.Pre(
                "  Benchmarks:",
                style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_COLOR}
            ))

            for bench_name in summary["benchmarks"]:
                bench_stats = summary["success_by_benchmark"].get(bench_name, {})
                description = benchmark_descriptions.get(bench_name, "")
                success_rate = bench_stats.get("success_rate", 0.0)
                total_runs = bench_stats.get("total_runs", 0)

                # Format success count with colored symbols
                n_success = int(success_rate * total_runs)
                n_total = total_runs
                n_failure = n_total - n_success

                # Build status string with colored symbols
                status_parts = []
                if n_success > 0:
                    status_parts.append(html.Span(f"{SYMBOL_SUCCESS}{n_success}", style={"color": COLOR_SUCCESS}))
                    status_parts.append("/")
                if n_failure > 0:
                    status_parts.append(html.Span(f"{SYMBOL_FAILURE}{n_failure}", style={"color": COLOR_FAILURE}))
                    status_parts.append("/")

                # Benchmark line with clickable link
                summary_children.append(html.Pre([
                    "    ",
                    html.A(bench_name, href=f"#benchmark-{bench_name}", style={
                        "fontFamily": FONT_FAMILY,
                        "fontSize": f"{FONT_SIZE}px",
                        "color": CATPPUCCIN_BLUE,
                        "textDecoration": "underline",
                    }),
                    " ",
                    *status_parts,
                    f"{n_total} ",
                    html.Span(f"({success_rate:.1%})", style={"color": TEXT_DIMMED}),
                ], style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_COLOR}))

                # Description if available
                if description:
                    summary_children.append(html.Pre(
                        f"      {description}",
                        style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_SECONDARY, "fontStyle": "italic"}
                    ))

        children.extend(summary_children)
    else:
        # Fallback if no summary available
        children.append(html.Pre(
            f"# Benchmark Results: {experiment_name}",
            style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_COLOR, "fontWeight": "bold"}
        ))

    children.append(html.Br())

    # Add violin plot for each benchmark
    for i, plot_info in enumerate(figures):
        bench_name = plot_info["benchmark"]

        # Add benchmark-specific summary with anchor
        if summary and metadata:
            bench_stats = summary["success_by_benchmark"].get(bench_name, {})
            benchmark_descriptions = metadata.get("benchmark_descriptions", {})
            description = benchmark_descriptions.get(bench_name, "")

            success_rate = bench_stats.get("success_rate", 0.0)
            total_runs = bench_stats.get("total_runs", 0)
            n_success = int(success_rate * total_runs)
            n_total = total_runs
            n_failure = n_total - n_success

            # Build status string
            status_parts = []
            if n_success > 0:
                status_parts.append(html.Span(f"{SYMBOL_SUCCESS}{n_success}", style={"color": COLOR_SUCCESS}))
                status_parts.append("/")
            if n_failure > 0:
                status_parts.append(html.Span(f"{SYMBOL_FAILURE}{n_failure}", style={"color": COLOR_FAILURE}))
                status_parts.append("/")

            # Benchmark header with anchor
            children.append(html.Div(id=f"benchmark-{bench_name}"))
            children.append(html.Pre([
                html.Span(f"## {bench_name}", style={"fontWeight": "bold", "textDecoration": "underline"}),
                " ",
                *status_parts,
                f"{n_total} ",
                html.Span(f"({success_rate:.1%})", style={"color": TEXT_DIMMED}),
            ], style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_COLOR}))

            if description:
                children.append(html.Pre(
                    f"   {description}",
                    style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_SECONDARY, "fontStyle": "italic"}
                ))

        children.append(html.Br())
        children.append(html.Pre(
            "Plan Cost",
            style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_COLOR}
        ))
        children.extend([
            html.Div([
                dcc.Graph(
                    id={"type": "graph", "index": i},
                    figure=plot_info["figure"],
                )
            ]),
            html.Br(),
        ])

    return children


def build_experiment_list_layout(experiments: list[dict]) -> list:
    """
    Build the experiment list view layout with CLI-style formatting.

    Args:
        experiments: List of experiment dicts with summaries

    Returns:
        List of Dash components
    """
    if not experiments:
        return [
            html.Pre(
                "No experiments found in MLflow database.",
                style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "color": COLOR_FAILURE},
            )
        ]

    # Build formatted output with Rich-style colors
    children = []

    # Header
    children.append(html.Pre(
        f"Benchmark Experiments (showing {len(experiments)} most recent)",
        style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px", "margin": "0", "padding": "0", "color": TEXT_COLOR}
    ))
    children.append(html.Br())

    for exp_idx, exp in enumerate(experiments):
        summary = exp["summary"]
        metadata = exp["metadata"]
        exp_name = exp["name"]
        creation_time = exp["creation_time"]

        # Use reusable function to build experiment summary block
        summary_block = build_experiment_summary_block(
            exp_name=exp_name,
            summary=summary,
            metadata=metadata,
            creation_time=creation_time,
            include_link=True,
        )
        children.extend(summary_block)

        # Add spacing between experiments
        children.append(html.Br())

    return children


# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.layout = create_main_layout()


# Flask route to serve artifact files
@app.server.route("/artifact/<run_id>/<path:filename>")
def serve_artifact(run_id: str, filename: str):
    """Serve an artifact file for a given run."""
    cache_key = f"{run_id}/{filename}"

    if cache_key not in _artifact_cache:
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=filename,
            )
            _artifact_cache[cache_key] = local_path
        except Exception as e:
            print(f"Could not download artifact {filename} for {run_id}: {e}")
            return f"Artifact not found: {e}", 404

    return send_file(_artifact_cache[cache_key])


# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    [Output("main-content", "children"), Output("data-store", "data")],
    [Input("url", "pathname")],
)
def refresh_data(pathname):
    """Reload data from MLflow and regenerate plots on page load/refresh."""
    try:
        # Route to experiment list view or detail view
        if pathname == "/" or pathname is None:
            # Show experiment list (limited to 10 most recent)
            experiments = load_all_experiments_with_summaries(limit=10)
            content = build_experiment_list_layout(experiments)
            return content, {}

        elif pathname.startswith("/experiment/"):
            # Show specific experiment
            experiment_name = pathname.replace("/experiment/", "")

            try:
                df, experiment_name, metadata = load_experiment_by_name(experiment_name)

                # Get experiment summary
                analyzer = BenchmarkAnalyzer()
                summary = analyzer.get_experiment_summary(experiment_name)

                figures = create_violin_plots_by_benchmark(df)
                content = build_content_layout(experiment_name, figures, df, metadata, summary)

                # Store minimal data for click handling
                # Just store run_id to artifact_uri mapping
                store_data = {}
                if "run_id" in df.columns and "artifact_uri" in df.columns:
                    for _, row in df.iterrows():
                        store_data[row["run_id"]] = row.get("artifact_uri", "")

                return content, store_data
            except ValueError as e:
                return [html.Div(
                    f"Experiment not found: {e}",
                    style={"color": "red", "fontFamily": FONT_FAMILY}
                )], {}

        else:
            # Unknown path
            return [html.Div(
                f"Page not found: {pathname}",
                style={"color": "red", "fontFamily": FONT_FAMILY}
            )], {}

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return [html.Div([
            html.H3("Error loading data:", style={"color": "red", "fontFamily": FONT_FAMILY}),
            html.Pre(str(e), style={"fontFamily": FONT_FAMILY}),
            html.Details([
                html.Summary("Full traceback", style={"fontFamily": FONT_FAMILY}),
                html.Pre(error_detail, style={"fontFamily": FONT_FAMILY, "fontSize": "10px"}),
            ]),
        ])], {}


@app.callback(
    [
        Output("log-modal", "is_open"),
        Output("modal-body-content", "children"),
        Output("modal-title", "children"),
    ],
    [Input({"type": "graph", "index": ALL}, "clickData")],
    [State("data-store", "data"), State("log-modal", "is_open")],
    prevent_initial_call=True,
)
def show_log_modal(click_data_list, stored_data, is_open):
    """Handle click on data point to show log.html in modal."""
    ctx = callback_context
    if not ctx.triggered:
        return False, None, ""

    # Get the specific input that triggered this callback
    trigger = ctx.triggered[0]
    trigger_id = trigger["prop_id"]

    # Extract the index from the trigger ID
    # Format is: '{"index":0,"type":"graph"}.clickData'
    if ".clickData" in trigger_id:
        # Parse the trigger ID to get which graph index was clicked
        id_part = trigger_id.split(".")[0]
        try:
            trigger_dict = json.loads(id_part)
            graph_index = trigger_dict.get("index")

            # Get the clickData for this specific graph
            if graph_index is not None and graph_index < len(click_data_list):
                click_data = click_data_list[graph_index]

                if click_data is not None:
                    points = click_data.get("points", [])
                    if points:
                        point = points[0]
                        run_id = point.get("customdata")

                        if run_id:
                            # Use the Flask route to serve the artifact
                            # Add timestamp to force iframe reload
                            timestamp = int(time.time() * 1000)
                            artifact_url = f"/artifact/{run_id}/log.html?t={timestamp}"
                            title = f"Run: {run_id} | Timestamp: {timestamp}"

                            # Create a new iframe component with unique key to force reload
                            iframe = html.Iframe(
                                src=artifact_url,
                                style={
                                    "width": "100%",
                                    "height": "80vh",
                                    "border": "none",
                                    "zoom": "0.75",
                                },
                                key=f"iframe-{run_id}-{timestamp}",  # Unique key forces recreation
                            )

                            return True, iframe, title
        except Exception as e:
            print(f"Error parsing trigger ID: {e}")

    return False, None, ""


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\nStarting Dash server...")
    print("Open http://127.0.0.1:8050/ in your browser")
    app.run(debug=True)

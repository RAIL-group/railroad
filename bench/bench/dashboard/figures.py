"""
Plotly figure creation functions for violin plots and visualizations.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from .styles import (
    FONT_FAMILY,
    FONT_SIZE,
    CATPPUCCIN_SURFACE0,
    COLOR_TIMEOUT,
    VIOLIN_FILL,
    VIOLIN_LINE,
    VIOLIN_LINE_WIDTH,
    VIOLIN_WIDTH,
    POINT_SIZE,
    POINT_COLOR,
    POINT_OUTLINE,
    POINT_OUTLINE_WIDTH,
    MEAN_MARKER_SIZE,
    MEAN_MARKER_COLOR,
    MEAN_MARKER_WIDTH,
    FAILED_MARKER_SYMBOL,
    FAILED_MARKER_SIZE,
    FAILED_MARKER_WIDTH,
    FAILED_MARKER_COLOR,
    MARGIN,
    HEIGHT_PER_CASE,
    MIN_PLOT_HEIGHT,
    ANNOTATION_PADDING,
    ANNOTATION_X_OFFSET,
    GRID_COLOR,
    GRID_WIDTH,
    PLOT_BGCOLOR,
    PAPER_BGCOLOR,
    ANNOTATION_BGCOLOR,
    TEXT_COLOR,
)
from .helpers import (
    _success_mask,
    _timeout_mask,
    format_status_count,
    format_case_params,
    compute_case_summary_stats,
)
from .data import compute_per_benchmark_limits


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

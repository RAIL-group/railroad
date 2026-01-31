"""
Parameter sweep analysis and visualization.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from .styles import (
    FONT_FAMILY,
    FONT_SIZE,
    CATPPUCCIN_SURFACE0,
    PLOT_BGCOLOR,
    PAPER_BGCOLOR,
    TEXT_COLOR,
    GRID_COLOR,
    GRID_WIDTH,
)
from .helpers import _success_mask, _timeout_mask


@dataclass
class SweepGroup:
    """A group of runs where only one parameter varies."""
    sweep_param: str           # The parameter being swept (e.g., "params.mcts.iterations")
    fixed_params: dict         # Other parameters that are constant in this group
    param_values: list         # Values of the sweep parameter (numeric, sorted)
    plan_costs: list           # Corresponding plan_cost values
    run_ids: list              # Run IDs for each point
    success_mask: list         # Whether each run succeeded
    timeout_mask: list         # Whether each run timed out
    wall_times: list           # Wall time for each run


@dataclass
class SweepAnalysis:
    """Analysis results for a parameter sweep."""
    param_name: str
    groups: list               # list[SweepGroup]
    correlation: Optional[float]  # Pearson correlation across all points
    slope: Optional[float]     # Linear fit slope
    intercept: Optional[float] # Linear fit intercept


def identify_sweep_parameters(df: pd.DataFrame, include_categorical: bool = True) -> list[str]:
    """
    Identify which params.* columns have >1 unique value.

    Args:
        df: DataFrame with params.* columns
        include_categorical: Whether to include non-numeric (categorical) parameters

    Returns:
        List of parameter names suitable for sweeps (excluding benchmark_name,
        case_idx, repeat_idx, timeout, and any with only 1 unique value)
    """
    excluded = {"params.benchmark_name", "params.case_idx", "params.repeat_idx", "params.timeout"}
    sweep_params = []

    for col in df.columns:
        if not col.startswith("params."):
            continue
        if col in excluded:
            continue

        # Check unique values
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 1:
            continue

        # Try to convert to numeric
        try:
            pd.to_numeric(df[col], errors='raise')
            sweep_params.append(col)
        except (ValueError, TypeError):
            # Include categorical parameters if requested
            if include_categorical:
                sweep_params.append(col)

    return sweep_params


def is_numeric_parameter(df: pd.DataFrame, param: str) -> bool:
    """Check if a parameter column is numeric."""
    try:
        pd.to_numeric(df[param], errors='raise')
        return True
    except (ValueError, TypeError):
        return False


def find_sweep_groups(
    df: pd.DataFrame,
    sweep_param: str,
    min_group_size: int = 2
) -> list[SweepGroup]:
    """
    Find groups where all parameters except sweep_param are constant.

    Algorithm:
    1. Get all params.* columns except sweep_param and meta-columns
    2. Group by those columns
    3. For each group with >=min_group_size rows, create a SweepGroup

    Args:
        df: DataFrame with params.* and metrics.* columns
        sweep_param: The parameter to sweep (e.g., "params.mcts.iterations")
        min_group_size: Minimum points required for a valid sweep group

    Returns:
        List of SweepGroup objects
    """
    # Identify fixed parameters (all params except sweep_param and meta-columns)
    meta_cols = {"params.benchmark_name", "params.case_idx", "params.repeat_idx"}
    fixed_param_cols = [
        col for col in df.columns
        if col.startswith("params.")
        and col not in meta_cols
        and col != sweep_param
    ]

    groups = []

    # Check if parameter is numeric
    is_numeric = is_numeric_parameter(df, sweep_param)

    # Group by fixed parameters
    if fixed_param_cols:
        grouped = df.groupby(fixed_param_cols, dropna=False)
    else:
        # No fixed params, entire df is one group
        grouped = [(None, df)]

    for group_key, group_df in grouped:
        if len(group_df) < min_group_size:
            continue

        if is_numeric:
            # Convert sweep_param to numeric and sort
            sweep_values = pd.to_numeric(group_df[sweep_param], errors='coerce')
            valid_mask = ~sweep_values.isna()

            if valid_mask.sum() < min_group_size:
                continue

            group_df_valid = group_df[valid_mask].copy()
            sweep_values_valid = sweep_values[valid_mask]

            # Sort by sweep parameter value
            sort_idx = sweep_values_valid.argsort()
            param_values = sweep_values_valid.iloc[sort_idx].tolist()
        else:
            # Categorical parameter - use string representation and sort alphabetically
            group_df_valid = group_df.copy()
            sweep_values_str = group_df_valid[sweep_param].astype(str)

            # Sort by string value for consistent ordering
            sort_idx = sweep_values_str.argsort()
            param_values = group_df_valid[sweep_param].iloc[sort_idx].tolist()

        # Extract fixed params dict
        if fixed_param_cols:
            if isinstance(group_key, tuple):
                fixed_params = dict(zip(fixed_param_cols, group_key))
            else:
                fixed_params = {fixed_param_cols[0]: group_key}
        else:
            fixed_params = {}

        # Get plan_cost, success, timeout, and wall_time
        plan_costs = group_df_valid["metrics.plan_cost"].iloc[sort_idx].tolist() if "metrics.plan_cost" in group_df_valid.columns else []
        success = _success_mask(group_df_valid["metrics.success"]).iloc[sort_idx].tolist() if "metrics.success" in group_df_valid.columns else [True] * len(sort_idx)
        timeout = _timeout_mask(group_df_valid["metrics.timeout"]).iloc[sort_idx].tolist() if "metrics.timeout" in group_df_valid.columns else [False] * len(sort_idx)
        wall_times = group_df_valid["metrics.wall_time"].iloc[sort_idx].tolist() if "metrics.wall_time" in group_df_valid.columns else []
        run_ids = group_df_valid["run_id"].iloc[sort_idx].tolist() if "run_id" in group_df_valid.columns else []

        groups.append(SweepGroup(
            sweep_param=sweep_param,
            fixed_params=fixed_params,
            param_values=param_values,
            plan_costs=plan_costs,
            run_ids=run_ids,
            success_mask=success,
            timeout_mask=timeout,
            wall_times=wall_times,
        ))

    return groups


def compute_sweep_correlation(groups: list[SweepGroup]) -> tuple:
    """
    Compute linear fit and correlation across all points in sweep groups.

    Args:
        groups: List of SweepGroup objects

    Returns:
        (correlation, slope, intercept) - None values if insufficient data or non-numeric params
    """
    all_x = []
    all_y = []

    for group in groups:
        for i, (x, y, success) in enumerate(zip(
            group.param_values,
            group.plan_costs,
            group.success_mask
        )):
            if success and y is not None and not np.isnan(y):
                all_x.append(x)
                all_y.append(y)

    if len(all_x) < 2:
        return None, None, None

    # Check if x values are numeric
    try:
        x_arr = np.array(all_x, dtype=float)
    except (ValueError, TypeError):
        # Non-numeric parameter - can't compute correlation
        return None, None, None

    y_arr = np.array(all_y)

    # Compute Pearson correlation
    correlation = np.corrcoef(x_arr, y_arr)[0, 1]

    # Linear fit
    slope, intercept = np.polyfit(x_arr, y_arr, 1)

    return correlation, slope, intercept


def should_use_log_scale(values: list) -> bool:
    """
    Determine if log scale should be used based on value range.

    Returns True if max/min > 10 (spans >1 order of magnitude).

    Args:
        values: List of numeric values

    Returns:
        True if log scale should be used
    """
    if not values or len(values) < 2:
        return False

    values_array = np.array(values)
    values_array = values_array[values_array > 0]  # Only positive values for log

    if len(values_array) < 2:
        return False

    ratio = values_array.max() / values_array.min()
    return ratio > 10


def analyze_parameter_sweep(
    df: pd.DataFrame,
    sweep_param: str
) -> SweepAnalysis:
    """
    Complete analysis for one parameter sweep.

    Args:
        df: Full experiment DataFrame
        sweep_param: Parameter to analyze

    Returns:
        SweepAnalysis with groups and statistics
    """
    groups = find_sweep_groups(df, sweep_param)
    correlation, slope, intercept = compute_sweep_correlation(groups)

    return SweepAnalysis(
        param_name=sweep_param,
        groups=groups,
        correlation=correlation,
        slope=slope,
        intercept=intercept,
    )


def create_sweep_figure(
    analysis: SweepAnalysis,
    show_fit_line: bool = True,
    show_group_lines: bool = True,
) -> go.Figure:
    """
    Create a parameter sweep plot with vertical violin plots.

    Design:
    - X-axis: Parameter value (e.g., mcts.iterations) with auto-detected log scale
    - Y-axis: plan_cost
    - Vertical violin plot for each parameter value (matching per-case styling)
    - Mean markers for each parameter value
    - Line connecting the means
    - Dim data points matching the violin plot styling

    Args:
        analysis: SweepAnalysis object
        show_fit_line: Whether to show linear regression line
        show_group_lines: Whether to connect points in same group (unused, kept for compatibility)

    Returns:
        Configured go.Figure
    """
    from .styles import (
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
        COLOR_TIMEOUT,
        FAILED_MARKER_SYMBOL,
        FAILED_MARKER_SIZE,
        FAILED_MARKER_WIDTH,
        FAILED_MARKER_COLOR,
    )

    fig = go.Figure()

    # Collect all data organized by parameter value, separating success/failure/timeout
    # First pass: collect all costs to determine xmax for filling NaN timeouts
    all_costs = []
    for group in analysis.groups:
        for cost in group.plan_costs:
            if cost is not None and not np.isnan(cost):
                all_costs.append(cost)

    xmax = max(all_costs) if all_costs else 1.0

    # Second pass: organize by parameter value
    data_by_param = {}  # param_val -> dict of success/failed/timeout/plan_costs/wall_times lists
    for group in analysis.groups:
        # Handle case where wall_times might be empty list
        wall_times_iter = group.wall_times if group.wall_times else [None] * len(group.param_values)

        for param_val, plan_cost, success, timeout, wall_time in zip(
            group.param_values,
            group.plan_costs,
            group.success_mask,
            group.timeout_mask,
            wall_times_iter
        ):
            if param_val not in data_by_param:
                data_by_param[param_val] = {
                    "success": [],
                    "failed": [],
                    "timeout": [],
                    "plan_costs_all": [],
                    "wall_times_all": [],
                }

            # Fill NaN plan_cost with xmax for timeout/failed runs (like per-case plots)
            display_cost = plan_cost
            if display_cost is None or np.isnan(display_cost):
                display_cost = xmax

            # Store original plan_cost and wall_time for statistics (not filled)
            data_by_param[param_val]["plan_costs_all"].append(plan_cost)
            data_by_param[param_val]["wall_times_all"].append(wall_time)

            if success:
                data_by_param[param_val]["success"].append(display_cost)
            elif timeout:
                data_by_param[param_val]["timeout"].append(display_cost)
            else:
                data_by_param[param_val]["failed"].append(display_cost)

    # Determine if parameter values are numeric
    param_values_list = list(data_by_param.keys())
    is_numeric_param = all(isinstance(v, (int, float)) and not np.isnan(v) for v in param_values_list)

    if is_numeric_param:
        # Sort numerically
        sorted_params = sorted(data_by_param.keys())
        # Check if log scale should be used
        use_log = should_use_log_scale(sorted_params)
        # Use actual values for x-axis
        param_to_x = {p: p for p in sorted_params}
        # Compute dx for jitter on failed/timeout runs
        if len(sorted_params) > 1:
            if use_log:
                dx = np.log10(sorted_params[-1]) - np.log10(sorted_params[0])
            else:
                dx = sorted_params[-1] - sorted_params[0]
        else:
            dx = 1.0
    else:
        # Categorical parameter - sort by string representation
        sorted_params = sorted(data_by_param.keys(), key=lambda x: str(x))
        use_log = False
        # Use index-based positioning for x-axis
        param_to_x = {p: i for i, p in enumerate(sorted_params)}
        dx = 1.0  # Fixed spacing for categorical

    # Create violin plot for each parameter value
    mean_x = []
    mean_y = []

    for param_val in sorted_params:
        param_data = data_by_param[param_val]
        success_costs = param_data["success"]
        failed_costs = param_data["failed"]
        timeout_costs = param_data["timeout"]
        x_pos = param_to_x[param_val]

        # Create vertical violin plot for successful runs
        if success_costs:
            fig.add_trace(go.Violin(
                x=[x_pos] * len(success_costs),
                y=success_costs,
                name=str(param_val),
                fillcolor=VIOLIN_FILL,
                line=dict(color=VIOLIN_LINE, width=VIOLIN_LINE_WIDTH),
                width=VIOLIN_WIDTH,
                box_visible=False,
                meanline_visible=False,
                points="all",
                pointpos=0,
                jitter=0.2,
                marker=dict(
                    size=POINT_SIZE,
                    color=POINT_COLOR,
                    line=dict(color=POINT_OUTLINE, width=POINT_OUTLINE_WIDTH),
                ),
                hoveron="points",
                hovertemplate=f"Plan cost: %{{y}}<br>{analysis.param_name.replace('params.', '')}: {param_val}<extra></extra>",
                showlegend=False,
            ))

            # Calculate mean from successful runs only
            mean_cost = np.mean(success_costs)
            mean_x.append(x_pos)
            mean_y.append(mean_cost)

            # Add mean marker
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[mean_cost],
                mode="markers",
                marker=dict(
                    symbol="line-ew",
                    size=MEAN_MARKER_SIZE,
                    color="rgba(255, 255, 255, 0)",
                    line=dict(color=MEAN_MARKER_COLOR, width=MEAN_MARKER_WIDTH),
                ),
                showlegend=False,
                hoverinfo="skip",
            ))

        # Add failed runs as X markers
        if failed_costs:
            # Apply jitter to x position
            x_jittered = [x_pos + np.random.uniform(-0.01 * dx, 0.01 * dx) for _ in failed_costs]
            fig.add_trace(go.Violin(
                x=x_jittered,
                y=failed_costs,
                name=f"{param_val} failed",
                fillcolor="rgba(0,0,0,0)",
                line=dict(color="rgba(0,0,0,0)", width=0),
                width=VIOLIN_WIDTH,
                box_visible=False,
                meanline_visible=False,
                points="all",
                pointpos=0,
                jitter=0.2,
                marker=dict(
                    symbol=FAILED_MARKER_SYMBOL,
                    size=FAILED_MARKER_SIZE,
                    color=None,
                    line=dict(color=FAILED_MARKER_COLOR, width=FAILED_MARKER_WIDTH),
                ),
                hoveron="points",
                hovertemplate=f"FAILED<br>Plan cost: %{{y}}<br>{analysis.param_name.replace('params.', '')}: {param_val}<extra></extra>",
                showlegend=False,
            ))

        # Add timeout runs as diamond markers
        if timeout_costs:
            x_jittered = [x_pos + np.random.uniform(-0.01 * dx, 0.01 * dx) for _ in timeout_costs]
            fig.add_trace(go.Violin(
                x=x_jittered,
                y=timeout_costs,
                name=f"{param_val} timeout",
                fillcolor="rgba(0,0,0,0)",
                line=dict(color="rgba(0,0,0,0)", width=0),
                width=VIOLIN_WIDTH,
                box_visible=False,
                meanline_visible=False,
                points="all",
                pointpos=0,
                jitter=0.2,
                marker=dict(
                    symbol="diamond",
                    size=POINT_SIZE + 1,
                    color="rgba(255, 215, 0, 0.8)",
                    line=dict(color=COLOR_TIMEOUT, width=1),
                ),
                hoveron="points",
                hovertemplate=f"TIMEOUT ⏱<br>Plan cost: %{{y}}<br>{analysis.param_name.replace('params.', '')}: {param_val}<extra></extra>",
                showlegend=False,
            ))

    # Add line connecting means (only for numeric parameters where order is meaningful)
    if len(mean_x) > 1 and is_numeric_param:
        fig.add_trace(go.Scatter(
            x=mean_x,  # Use actual numeric values
            y=mean_y,
            mode="lines",
            line=dict(color=MEAN_MARKER_COLOR, width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Add invisible scatter points for hover with summary statistics
    hover_x = []
    hover_y = []
    hover_text = []

    # Get y-axis range to position hover points at bottom
    all_y_values = []
    for param_data in data_by_param.values():
        all_y_values.extend(param_data["success"])
        all_y_values.extend(param_data["failed"])
        all_y_values.extend(param_data["timeout"])

    y_min = min(all_y_values) if all_y_values else 0
    hover_y_pos = y_min * 0.95  # Position slightly below minimum y value

    for param_val in sorted_params:
        param_data = data_by_param[param_val]

        # Compute summary statistics
        plan_costs = [c for c in param_data["plan_costs_all"] if c is not None and not np.isnan(c)]
        wall_times = [w for w in param_data["wall_times_all"] if w is not None and not np.isnan(w)]

        n_success = len(param_data["success"])
        n_total = n_success + len(param_data["failed"]) + len(param_data["timeout"])

        avg_plan_cost = np.mean(plan_costs) if plan_costs else None
        success_rate = n_success / n_total if n_total > 0 else None
        avg_wall_time = np.mean(wall_times) if wall_times else None

        # Build hover text
        hover_parts = ["<b>Summary Statistics</b>", f"<b>{param_val}</b>"]
        if avg_plan_cost is not None:
            hover_parts.append(f"Avg Plan Cost: {avg_plan_cost:.2f}")
        if success_rate is not None:
            hover_parts.append(f"Success Rate: {success_rate:.1%}")
        if avg_wall_time is not None:
            hover_parts.append(f"Avg Wall Time: {avg_wall_time:.2f}s")

        hover_x.append(param_to_x[param_val])
        hover_y.append(hover_y_pos)
        hover_text.append("<br>".join(hover_parts))

    # Add invisible scatter trace for hover
    if hover_x:
        fig.add_trace(go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(
                size=15,
                color="rgba(0,0,0,0)",  # Fully transparent
            ),
            hovertext=hover_text,
            hoverinfo="text",
            hoverlabel=dict(
                font=dict(family=FONT_FAMILY, size=FONT_SIZE, color=TEXT_COLOR),
                bgcolor=CATPPUCCIN_SURFACE0,
            ),
            showlegend=False,
        ))

    # Prepare tick labels with success rate annotations
    tickvals = []
    ticktext = []
    for param_val in sorted_params:
        param_data = data_by_param[param_val]
        n_success = len(param_data["success"])
        n_total = n_success + len(param_data["failed"]) + len(param_data["timeout"])
        success_pct = (n_success / n_total * 100) if n_total > 0 else 0
        # For categorical params, truncate long labels
        label = str(param_val)
        if not is_numeric_param and len(label) > 40:
            label = label[:37] + "..."
        tickvals.append(param_to_x[param_val])
        ticktext.append(f"{label}<br><sub>{n_success}/{n_total}<br>({success_pct:.0f}%)</sub>")

    # Layout
    param_display = analysis.param_name.replace("params.", "")

    # Create annotations list for parameter name and r-value
    annotations = []

    # Add left-aligned parameter name and r-value annotation
    label_parts = [param_display]
    if analysis.correlation is not None:
        # Add r-value with dimmer color (50% opacity)
        label_parts.append(f"  <span style='opacity:0.5'>r = {analysis.correlation:.3f}</span>")

    annotations.append(dict(
        text="".join(label_parts),
        xref="paper",
        yref="paper",
        x=0,
        y=1.02,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=FONT_SIZE, color=TEXT_COLOR, family=FONT_FAMILY),
    ))

    fig.update_layout(
        xaxis_title=param_display,
        yaxis_title="plan_cost",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE, color=TEXT_COLOR),
        plot_bgcolor=PLOT_BGCOLOR,
        paper_bgcolor=PAPER_BGCOLOR,
        height=300,  # Fixed height to prevent growing
        margin=dict(l=00, r=00, t=5, b=5),  # Increased bottom margin for multi-line tick labels
        autosize=True,  # Allow width to be responsive
        transition=dict(duration=0),  # Disable animations
        annotations=annotations,
    )

    # Configure axes
    fig.update_xaxes(
        type="log" if use_log else "linear",  # Use log scale if parameter spans >1 order of magnitude
        gridcolor=GRID_COLOR,
        gridwidth=GRID_WIDTH,
        color=TEXT_COLOR,
        tickvals=tickvals,
        ticktext=ticktext,
        tickfont=dict(size=FONT_SIZE - 1, family=FONT_FAMILY, color=TEXT_COLOR),  # Slightly smaller for compactness
        title_font=dict(size=FONT_SIZE, family=FONT_FAMILY, color=TEXT_COLOR),
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        gridwidth=GRID_WIDTH,
        color=TEXT_COLOR,
        title_font=dict(size=FONT_SIZE, family=FONT_FAMILY, color=TEXT_COLOR),
        zeroline=False,
    )

    return fig


def create_sweep_plots_for_benchmark(bench_df: pd.DataFrame) -> list[dict]:
    """
    Generate sweep plots for a single benchmark's data.

    Args:
        bench_df: DataFrame filtered for one benchmark

    Returns:
        List of dicts with 'param', 'figure', 'analysis' keys
    """
    sweep_params = identify_sweep_parameters(bench_df)

    results = []
    for param in sweep_params:
        analysis = analyze_parameter_sweep(bench_df, param)

        # Skip if no valid groups
        if not analysis.groups:
            continue

        fig = create_sweep_figure(analysis)

        results.append({
            "param": param,
            "figure": fig,
            "analysis": analysis,
        })

    return results


def create_all_sweep_plots(df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Create all parameter sweep plots organized by benchmark.

    Args:
        df: Full experiment DataFrame

    Returns:
        Dict mapping benchmark_name -> list of sweep plot dicts
    """
    if "params.benchmark_name" not in df.columns:
        return {}

    benchmarks = sorted(df["params.benchmark_name"].unique())
    sweep_plots_by_benchmark = {}

    for benchmark in benchmarks:
        bench_df = df[df["params.benchmark_name"] == benchmark].copy()
        sweep_plots = create_sweep_plots_for_benchmark(bench_df)
        if sweep_plots:
            sweep_plots_by_benchmark[benchmark] = sweep_plots

    return sweep_plots_by_benchmark

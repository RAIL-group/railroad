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
    CATPPUCCIN_BLUE,
    CATPPUCCIN_GREEN,
    CATPPUCCIN_RED,
    PLOT_BGCOLOR,
    PAPER_BGCOLOR,
    TEXT_COLOR,
    GRID_COLOR,
    GRID_WIDTH,
)
from .helpers import _success_mask


@dataclass
class SweepGroup:
    """A group of runs where only one parameter varies."""
    sweep_param: str           # The parameter being swept (e.g., "params.mcts.iterations")
    fixed_params: dict         # Other parameters that are constant in this group
    param_values: list         # Values of the sweep parameter (numeric, sorted)
    plan_costs: list           # Corresponding plan_cost values
    run_ids: list              # Run IDs for each point
    success_mask: list         # Whether each run succeeded


@dataclass
class SweepAnalysis:
    """Analysis results for a parameter sweep."""
    param_name: str
    groups: list               # list[SweepGroup]
    correlation: Optional[float]  # Pearson correlation across all points
    slope: Optional[float]     # Linear fit slope
    intercept: Optional[float] # Linear fit intercept


def identify_sweep_parameters(df: pd.DataFrame) -> list[str]:
    """
    Identify which params.* columns have >1 unique value and are numeric-convertible.

    Args:
        df: DataFrame with params.* columns

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
            # Skip non-numeric parameters
            continue

    return sweep_params


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

    # Group by fixed parameters
    if fixed_param_cols:
        grouped = df.groupby(fixed_param_cols, dropna=False)
    else:
        # No fixed params, entire df is one group
        grouped = [(None, df)]

    for group_key, group_df in grouped:
        if len(group_df) < min_group_size:
            continue

        # Convert sweep_param to numeric and sort
        sweep_values = pd.to_numeric(group_df[sweep_param], errors='coerce')
        valid_mask = ~sweep_values.isna()

        if valid_mask.sum() < min_group_size:
            continue

        group_df_valid = group_df[valid_mask].copy()
        sweep_values_valid = sweep_values[valid_mask]

        # Sort by sweep parameter
        sort_idx = sweep_values_valid.argsort()

        # Extract fixed params dict
        if fixed_param_cols:
            if isinstance(group_key, tuple):
                fixed_params = dict(zip(fixed_param_cols, group_key))
            else:
                fixed_params = {fixed_param_cols[0]: group_key}
        else:
            fixed_params = {}

        # Get plan_cost and success
        plan_costs = group_df_valid["metrics.plan_cost"].iloc[sort_idx].tolist() if "metrics.plan_cost" in group_df_valid.columns else []
        success = _success_mask(group_df_valid["metrics.success"]).iloc[sort_idx].tolist() if "metrics.success" in group_df_valid.columns else [True] * len(sort_idx)
        run_ids = group_df_valid["run_id"].iloc[sort_idx].tolist() if "run_id" in group_df_valid.columns else []

        groups.append(SweepGroup(
            sweep_param=sweep_param,
            fixed_params=fixed_params,
            param_values=sweep_values_valid.iloc[sort_idx].tolist(),
            plan_costs=plan_costs,
            run_ids=run_ids,
            success_mask=success,
        ))

    return groups


def compute_sweep_correlation(groups: list[SweepGroup]) -> tuple:
    """
    Compute linear fit and correlation across all points in sweep groups.

    Args:
        groups: List of SweepGroup objects

    Returns:
        (correlation, slope, intercept) - None values if insufficient data
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

    # Compute Pearson correlation
    x_arr = np.array(all_x)
    y_arr = np.array(all_y)

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
    )

    fig = go.Figure()

    # Collect all data organized by parameter value
    data_by_param = {}
    for group in analysis.groups:
        for param_val, plan_cost, success in zip(
            group.param_values,
            group.plan_costs,
            group.success_mask
        ):
            if param_val not in data_by_param:
                data_by_param[param_val] = []
            if plan_cost is not None and not np.isnan(plan_cost):
                data_by_param[param_val].append(plan_cost)

    # Sort parameter values
    sorted_params = sorted(data_by_param.keys())

    # Check if log scale should be used
    use_log = should_use_log_scale(sorted_params)

    # Create violin plot for each parameter value
    mean_x = []
    mean_y = []

    for param_val in sorted_params:
        costs = data_by_param[param_val]
        if not costs:
            continue

        # Create vertical violin plot
        fig.add_trace(go.Violin(
            x=[str(param_val)] * len(costs),
            y=costs,
            name=str(param_val),
            fillcolor=VIOLIN_FILL,
            line=dict(color=VIOLIN_LINE, width=VIOLIN_LINE_WIDTH),
            width=VIOLIN_WIDTH,
            box_visible=False,
            meanline_visible=False,  # We'll add custom mean markers
            points="all",
            pointpos=0,
            jitter=0.5,
            marker=dict(
                size=POINT_SIZE,
                color=POINT_COLOR,
                line=dict(color=POINT_OUTLINE, width=POINT_OUTLINE_WIDTH),
            ),
            hoveron="points",
            hovertemplate=f"Plan cost: %{{y}}<br>{analysis.param_name.replace('params.', '')}: {param_val}<extra></extra>",
            showlegend=False,
        ))

        # Calculate mean
        mean_cost = np.mean(costs)
        mean_x.append(str(param_val))
        mean_y.append(mean_cost)

        # Add mean marker
        fig.add_trace(go.Scatter(
            x=[str(param_val)],
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

    # Add line connecting means
    if len(mean_x) > 1:
        fig.add_trace(go.Scatter(
            x=mean_x,
            y=mean_y,
            mode="lines",
            line=dict(color=MEAN_MARKER_COLOR, width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Add correlation annotation if available
    if analysis.correlation is not None:
        fig.add_annotation(
            text=f"r = {analysis.correlation:.3f}",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(size=FONT_SIZE, color=TEXT_COLOR, family=FONT_FAMILY),
            bgcolor="rgba(0,0,0,0.3)",
            borderpad=4,
        )

    # Layout
    param_display = analysis.param_name.replace("params.", "")

    fig.update_layout(
        title=dict(
            text=f"Plan Cost vs {param_display}",
            font=dict(size=14, family=FONT_FAMILY, color=TEXT_COLOR),
        ),
        xaxis_title=param_display,
        yaxis_title="plan_cost",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE, color=TEXT_COLOR),
        plot_bgcolor=PLOT_BGCOLOR,
        paper_bgcolor=PAPER_BGCOLOR,
        height=300,  # Fixed height to prevent growing
        margin=dict(l=60, r=30, t=40, b=50),
        autosize=False,  # Disable auto-sizing
        transition=dict(duration=0),  # Disable animations
    )

    # Configure axes
    fig.update_xaxes(
        type="category",  # Use category type for discrete parameter values
        gridcolor=GRID_COLOR,
        gridwidth=GRID_WIDTH,
        color=TEXT_COLOR,
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        gridwidth=GRID_WIDTH,
        color=TEXT_COLOR,
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

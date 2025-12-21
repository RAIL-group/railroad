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
# STYLING CONSTANTS
# =============================================================================

# Font Configuration
FONT_FAMILY = "Iosevka, monospace"
FONT_SIZE = 12
FONT_SIZE_TITLE = 16

# Status Colors (matching progress.py)
COLOR_SUCCESS = "#228B22"      # Forest green
COLOR_FAILURE = "#DC143C"      # Crimson
COLOR_ERROR = "#FF8C00"        # Orange
COLOR_TIMEOUT = "#FFD700"      # Gold

# Status Symbols (matching progress.py)
SYMBOL_SUCCESS = "\u2713"      # checkmark
SYMBOL_FAILURE = "\u2717"      # x
SYMBOL_ERROR = "\u26a0"        # warning
SYMBOL_TIMEOUT = "\u23f1"      # stopwatch

# Violin Plot Styling
VIOLIN_FILL = "rgba(31, 119, 180, 0.05)"
VIOLIN_LINE = "rgba(31, 119, 180, 0.4)"
VIOLIN_LINE_WIDTH = 0.35
VIOLIN_WIDTH = 0.25

# Data Point Markers
POINT_SIZE = 5
POINT_COLOR = "rgba(60, 60, 60, 0.4)"
POINT_OUTLINE = "rgba(255, 255, 255, 0.7)"
POINT_OUTLINE_WIDTH = 0.5

# Mean Marker
MEAN_MARKER_SIZE = 7
MEAN_MARKER_COLOR = "rgba(120, 120, 255, 1.0)"
MEAN_MARKER_WIDTH = 2

# Failed Run Markers
FAILED_MARKER_SYMBOL = "x-thin"
FAILED_MARKER_SIZE = 5
FAILED_MARKER_WIDTH = 1
FAILED_MARKER_COLOR = "rgba(220, 20, 60, 0.7)"

# Layout & Spacing
MARGIN = dict(l=0, r=10, t=40, b=10)
HEIGHT_PER_CASE = 40
MIN_PLOT_HEIGHT = 280
ANNOTATION_PADDING = 0.5
ANNOTATION_X_OFFSET = 0.3  # Fraction of dx to offset annotations

# Grid Styling
GRID_COLOR = "rgba(0, 0, 0, 0.05)"
GRID_WIDTH = 0.5
PLOT_BGCOLOR = "rgba(245, 245, 245, 0.0)"
PAPER_BGCOLOR = "white"
ANNOTATION_BGCOLOR = "rgba(255, 255, 255, 0.9)"


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
    """Format case parameters consistently (plain text version)."""
    parts = [f"{k}={v}" for k, v in params.items() if v is not None]
    return ", ".join(parts)


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
) -> go.Violin:
    """
    Create a single violin trace for successful runs.

    Args:
        plan_costs: Series of plan_cost values
        case_label: Y-axis label (e.g., "Case 5")
        run_ids: Series of run_id values for click handling

    Returns:
        Configured go.Violin trace
    """
    return go.Violin(
        y=[case_label] * len(plan_costs),
        x=plan_costs,
        customdata=run_ids.tolist(),
        name=case_label,
        orientation="h",
        fillcolor=VIOLIN_FILL,
        line=dict(color=VIOLIN_LINE, width=VIOLIN_LINE_WIDTH),
        width=VIOLIN_WIDTH,
        box_visible=False,
        meanline_visible=True,
        points="all",
        pointpos=0,
        jitter=0.5,
        marker=dict(
            size=POINT_SIZE,
            color=POINT_COLOR,
            line=dict(color=POINT_OUTLINE, width=POINT_OUTLINE_WIDTH),
            symbol="circle",
        ),
        hoveron="points",
        hovertemplate="Plan cost: %{x}<br>Run ID: %{customdata}<br>Click for log<extra></extra>",
        showlegend=False,
    )


def create_failed_points_trace(
    plan_costs: pd.Series,
    case_label: str,
    run_ids: pd.Series,
    dx: float,
) -> go.Violin:
    """
    Create X markers for failed runs.

    Args:
        plan_costs: Series of plan_cost values (will be jittered)
        case_label: Y-axis label
        run_ids: Series of run_id values for click handling
        dx: Range of x values for jitter scaling

    Returns:
        Configured go.Violin trace showing only points as X markers
    """
    # Add jitter to failed points
    jittered_costs = plan_costs + np.random.uniform(-0.01 * dx, 0.01 * dx, size=len(plan_costs))

    return go.Violin(
        y=[case_label] * len(plan_costs),
        x=jittered_costs,
        customdata=run_ids.tolist(),
        name=f"{case_label} failed",
        orientation="h",
        # Hide violin body entirely (points-only trace)
        fillcolor="rgba(0,0,0,0)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        # No summary stats
        box_visible=False,
        meanline_visible=False,
        # Points with jitter
        points="all",
        pointpos=0,
        jitter=0.5,
        # Thin red X
        marker=dict(
            symbol=FAILED_MARKER_SYMBOL,
            size=FAILED_MARKER_SIZE,
            line=dict(color=FAILED_MARKER_COLOR, width=FAILED_MARKER_WIDTH),
        ),
        hoveron="points",
        hovertemplate="FAILED<br>Plan cost: %{x}<br>Run ID: %{customdata}<br>Click for log<extra></extra>",
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

    Returns:
        Plotly annotation dict
    """
    # Format status count
    status_html = format_status_count(n_success, n_total)

    # Format parameters
    param_parts = [f"{k}={v}" for k, v in params.items() if v is not None]
    param_str = f"{status_html} Case {case_idx:{case_width}d}: " + ", ".join(param_parts)

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
            font=dict(family=FONT_FAMILY, size=FONT_SIZE),
            bgcolor=PAPER_BGCOLOR,
        ),
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        yshift=4,
        font=dict(size=FONT_SIZE, family=FONT_FAMILY),
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

        # Determine success mask
        if "metrics.success" in case_data.columns:
            mask = _success_mask(case_data["metrics.success"])
        else:
            mask = pd.Series([True] * len(case_data), index=case_data.index)

        success_data = case_data[mask].copy()
        failed_data = case_data[~mask].copy()

        # Fill NaN plan_cost with xmax for display
        if "metrics.plan_cost" in success_data.columns:
            success_data["metrics.plan_cost"] = success_data["metrics.plan_cost"].fillna(xmax)
        if "metrics.plan_cost" in failed_data.columns:
            failed_data["metrics.plan_cost"] = failed_data["metrics.plan_cost"].fillna(xmax)

        # Violin trace for successful runs
        if not success_data.empty and "metrics.plan_cost" in success_data.columns:
            run_ids = success_data["run_id"] if "run_id" in success_data.columns else pd.Series([""] * len(success_data))
            fig.add_trace(create_violin_trace(
                success_data["metrics.plan_cost"],
                case_label,
                run_ids,
            ))

            # Mean marker
            case_mean = float(success_data["metrics.plan_cost"].mean())
            fig.add_trace(create_mean_marker(case_mean, case_label))

        # Failed runs as X markers
        if not failed_data.empty and "metrics.plan_cost" in failed_data.columns:
            run_ids = failed_data["run_id"] if "run_id" in failed_data.columns else pd.Series([""] * len(failed_data))
            fig.add_trace(create_failed_points_trace(
                failed_data["metrics.plan_cost"],
                case_label,
                run_ids,
                dx,
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
        n_success = int(mask.sum())

        annotations.append(create_case_annotation(
            case_idx=case_idx,
            params=params,
            n_success=n_success,
            n_total=n_total,
            summary_stats=summary_stats,
            x_position=annote_xloc,
            case_label=case_label,
            case_width=case_width,
        ))

    # Calculate height
    height = max(MIN_PLOT_HEIGHT, len(case_order) * HEIGHT_PER_CASE)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{benchmark} - Plan Cost Distribution",
            font=dict(size=FONT_SIZE_TITLE, family=FONT_FAMILY),
            x=0,  # Left-align title
            xanchor="left",
        ),
        xaxis_title="Plan Cost",
        height=height,
        margin=MARGIN,
        annotations=annotations,
        font=dict(size=FONT_SIZE, family=FONT_FAMILY),
        plot_bgcolor=PLOT_BGCOLOR,
        paper_bgcolor=PAPER_BGCOLOR,
    )

    # Force the y categories so cases with only failed points still show up
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=case_labels,
        ticks="",
        showticklabels=False,
    )

    # Set x-axis range
    fig.update_xaxes(
        range=[annote_xloc, (xmax * 1.05) if xmax != 0 else 1],
        gridcolor=GRID_COLOR,
        gridwidth=GRID_WIDTH,
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
            html.Div(id="main-content"),
            create_log_modal(),
        ],
        style={"padding": "20px", "fontFamily": FONT_FAMILY},
    )


def build_content_layout(experiment_name: str, figures: list[dict], df: pd.DataFrame) -> list:
    """
    Build the content layout with all graphs and data sample.

    Args:
        experiment_name: Name of the experiment
        figures: List of figure dicts from create_violin_plots_by_benchmark
        df: DataFrame for raw data sample

    Returns:
        List of Dash components
    """
    children = [
        html.H1(
            f"Benchmark Results: {experiment_name}",
            style={"fontFamily": FONT_FAMILY},
        ),
        html.Hr(),
    ]

    # Add violin plot for each benchmark
    for i, plot_info in enumerate(figures):
        children.extend([
            html.Div([
                dcc.Graph(
                    id={"type": "graph", "index": i},
                    figure=plot_info["figure"],
                )
            ]),
            html.Hr(),
        ])

    # Add raw data sample
    cols = [
        "params.benchmark_name",
        "params.case_idx",
        "params.repeat_idx",
        "metrics.plan_cost",
        "metrics.success",
    ]
    existing_cols = [c for c in cols if c in df.columns]

    children.append(
        html.Div([
            html.H3("Raw Data Sample", style={"fontFamily": FONT_FAMILY}),
            html.Pre(
                df[existing_cols].head(10).to_string(),
                style={"fontFamily": FONT_FAMILY, "fontSize": f"{FONT_SIZE}px"},
            ),
        ])
    )

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
        df, experiment_name = load_latest_experiment()
        figures = create_violin_plots_by_benchmark(df)
        content = build_content_layout(experiment_name, figures, df)

        # Store minimal data for click handling
        # Just store run_id to artifact_uri mapping
        store_data = {}
        if "run_id" in df.columns and "artifact_uri" in df.columns:
            for _, row in df.iterrows():
                store_data[row["run_id"]] = row.get("artifact_uri", "")

        return content, store_data
    except Exception as e:
        return [html.Div(f"Error loading data: {e}", style={"color": "red"})], {}


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

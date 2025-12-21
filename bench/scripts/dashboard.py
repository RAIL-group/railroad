#!/usr/bin/env python3
"""
Interactive Plotly Dash dashboard for benchmark results visualization.

Usage:
    uv run bench/scripts/dashboard.py
"""

import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import sys


from bench.analysis import BenchmarkAnalyzer


# Needed to opt-in to new behavior and surpress warnings
pd.set_option("future.no_silent_downcasting", True)


def load_latest_experiment():
    """Load the most recent benchmark experiment."""
    analyzer = BenchmarkAnalyzer()
    experiments = analyzer.list_experiments()

    if experiments.empty:
        raise ValueError("No benchmark experiments found in MLflow database")

    # Get the most recent experiment
    latest_exp = experiments.iloc[0]
    print(f"Loading experiment: {latest_exp['name']}")
    print(f"Created at: {latest_exp['creation_time']}")

    # Load all runs
    df = analyzer.load_experiment(latest_exp["name"])

    if df.empty:
        raise ValueError(f"No runs found in experiment {latest_exp['name']}")

    print(f"Loaded {len(df)} runs")
    return df, latest_exp["name"]


def _success_mask(series: pd.Series) -> pd.Series:
    """
    Robustly interpret metrics.success as a boolean mask.
    Handles bool, 0/1, floats, and missing.
    """
    if series is None:
        return pd.Series(False)
    s = series.fillna(0)
    # Convert common representations to float, then threshold
    try:
        s = s.astype(float)
    except Exception:
        # Fallback: treat truthy strings like "true"/"1" as success
        s = s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
        return s
    return s > 0.5


def create_violin_plots_by_benchmark(df: pd.DataFrame):
    """Create horizontal violin plots for each benchmark, with cases stacked vertically."""
    df = df.copy()

    if "case_id" not in df.columns:
        df["case_id"] = df["params.benchmark_name"] + "_case_" + df["params.case_idx"].astype(str)

    if "metrics.plan_cost" not in df.columns:
        return []

    # Get unique benchmarks
    benchmarks = sorted(df["params.benchmark_name"].unique())
    figures = []

    for benchmark in benchmarks:
        # Filter data for this benchmark
        bench_df = df[df["params.benchmark_name"] == benchmark].copy()

        # Convert case_idx to int for proper sorting
        bench_df["case_idx_int"] = bench_df["params.case_idx"].astype(int)

        # Get unique cases sorted by integer case index (reversed for low-to-high display)
        unique_cases = bench_df.groupby("case_idx_int").first().sort_index()
        case_order = unique_cases.index.tolist()[::-1]  # Reverse for low-to-high

        # Establish categorical y labels up-front so "failed-only" cases render
        case_labels = [f"Case {i}" for i in case_order]

        fig = go.Figure()
        annotations = []

        # Compute limits across all non-NaN plan_cost (success + failed)
        all_cost = bench_df["metrics.plan_cost"].dropna()
        if all_cost.empty:
            # No usable x data at all for this benchmark
            xmin, xmax = 0.0, 1.0
        else:
            xmin, xmax = float(all_cost.min()), float(all_cost.max())
        dx = max(xmax - xmin, 1.0)

        # Place annotation block to the left of xmin
        annote_xloc = 0.95 * xmin - 0.3 * dx

        for case_idx in case_order:
            case_label = f"Case {case_idx}"
            case_data = bench_df[bench_df["case_idx_int"] == case_idx].copy()

            if "metrics.success" in case_data.columns:
                mask = _success_mask(case_data["metrics.success"])
            else:
                mask = pd.Series([True] * len(case_data), index=case_data.index)

            success_data = case_data[mask].fillna(xmax)
            failed_data = case_data[~mask].fillna(xmax)

            # Violin: ONLY successful runs
            if not success_data.empty:
                fig.add_trace(
                    go.Violin(
                        y=[case_label] * len(success_data),
                        x=success_data["metrics.plan_cost"],
                        name=case_label,
                        orientation="h",
                        fillcolor="rgba(31, 119, 180, 0.05)",
                        line=dict(color="rgba(31, 119, 180, 0.4)", width=0.35),
                        width=0.25,
                        box_visible=False,
                        meanline_visible=True,
                        points="all",
                        pointpos=0,
                        jitter=0.5,
                        marker=dict(
                            size=5,
                            color="rgba(60, 60, 60, 0.4)",
                            line=dict(color="rgba(255, 255, 255, 0.7)", width=0.5),
                            symbol="circle",
                        ),
                        hoveron="points",
                        hovertemplate="Plan cost: %{x}<br>",
                        showlegend=False,
                    )
                )

                # Mean marker: ONLY successful runs
                case_mean = float(success_data["metrics.plan_cost"].mean())
                fig.add_trace(
                    go.Scatter(
                        x=[case_mean],
                        y=[case_label],
                        mode="markers",
                        marker=dict(
                            symbol="line-ns",
                            size=7,
                            color="rgba(255, 255, 255, 0)",
                            line=dict(color="rgba(120, 120, 255, 1.0)", width=2),
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            # Failed runs: overlay as red X
            if not failed_data.empty:
                import numpy as np
                failed_data["metrics.plan_cost"] += np.random.uniform(-0.01 * dx, 0.01 * dx, size=len(failed_data))
                fig.add_trace(go.Violin(
                    y=[case_label] * len(failed_data),
                    x=failed_data["metrics.plan_cost"],

                    name=f"{case_label} failed",
                    orientation="h",

                    # Hide violin body entirely (points-only trace)
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(color="rgba(0,0,0,0)", width=0),

                    # No summary stats (avoid implying a distribution summary)
                    box_visible=False,
                    meanline_visible=False,

                    # Points with jitter
                    points="all",
                    pointpos=0,
                    jitter=0.5,

                    # Thin red X
                    marker=dict(
                        symbol="x-thin",
                        size=5,
                        line=dict(color="rgba(220, 20, 60, 0.7)", width=1),  # thin X stroke
                    ),

                    hoveron="points",
                    hovertemplate="FAILED<br>Plan cost: %{x}<br><extra></extra>",
                    showlegend=False,
                ))

            # Case parameter annotation (from first row in this case)
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

            # Success count for this case
            n_total = len(case_data)
            n_success = int(mask.sum())
            success_color = "#4f8f6f" if n_success == n_total else "#b55a5a"

            param_parts = [f"{k}={v}" for k, v in params.items() if v is not None]
            success_html = (
                f"<span style='color:{success_color}; font-weight:bold;'>"
                f"{n_success}/{n_total}</span>"
            )
            param_str = (success_html
                         + f" Case {case_idx:2d}: "
                         + ", ".join(param_parts)
                         )

            annotations.append(
                dict(
                    x=annote_xloc,
                    y=case_label,
                    xref="x",
                    yref="y",
                    text=param_str,
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    yshift=4,
                    font=dict(size=12, family="monospace"),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    borderpad=0.5,
                )
            )

        height = max(280, len(case_order) * 40)

        fig.update_layout(
            title=f"{benchmark} - Plan Cost Distribution",
            xaxis_title="Plan Cost",
            height=height,
            margin=dict(l=0, r=10, t=40, b=10),
            annotations=annotations,
            font=dict(size=12, family="monospace"),
            plot_bgcolor="rgba(245, 245, 245, 0.0)",
            paper_bgcolor="white",
        )

        # Force the y categories so cases with only failed points still show up
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=case_labels,
            ticks="",
            showticklabels=False,
        )

        # Keep x range wide enough for left annotations and data
        fig.update_xaxes(
            range=[annote_xloc, (xmax * 1.05) if xmax != 0 else 1],
            gridcolor="rgba(0, 0, 0, 0.05)",
            gridwidth=0.5,
        )

        figures.append({"benchmark": benchmark, "figure": fig})

    return figures


# Initialize app
app = dash.Dash(__name__)

# Load data
try:
    df, experiment_name = load_latest_experiment()
    violin_plots = create_violin_plots_by_benchmark(df)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Create layout
layout_children = [
    html.H1(f"Benchmark Results: {experiment_name}"),
    html.Hr(),
]

# Add violin plot for each benchmark
for plot_info in violin_plots:
    layout_children.extend(
        [
            html.Div([dcc.Graph(figure=plot_info["figure"])]),
            html.Hr(),
        ]
    )

# Add raw data sample
cols = [
    "params.benchmark_name",
    "params.case_idx",
    "params.repeat_idx",
    "metrics.plan_cost",
    "metrics.success",
]
existing_cols = [c for c in cols if c in df.columns]

layout_children.append(
    html.Div(
        [
            html.H3("Raw Data Sample"),
            html.Pre(df[existing_cols].head(10).to_string()),
        ]
    )
)

app.layout = html.Div(layout_children, style={"padding": "20px"})


if __name__ == "__main__":
    print("\nStarting Dash server...")
    print("Open http://127.0.0.1:8050/ in your browser")
    app.run(debug=True)

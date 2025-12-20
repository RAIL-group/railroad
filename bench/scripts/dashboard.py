#!/usr/bin/env python3
"""
Interactive Plotly Dash dashboard for benchmark results visualization.

Usage:
    uv run python bench/scripts/dashboard.py
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import sys

# Add bench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.analysis import BenchmarkAnalyzer


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
    df = analyzer.load_experiment(latest_exp['name'])

    if df.empty:
        raise ValueError(f"No runs found in experiment {latest_exp['name']}")

    print(f"Loaded {len(df)} runs")

    return df, latest_exp['name']


def prepare_grouped_data(df):
    """Group data by benchmark and case, computing statistics."""
    # Create a case identifier combining benchmark name and case index
    df['case_id'] = df['params.benchmark_name'] + '_case_' + df['params.case_idx'].astype(str)

    # Group by case_id and compute statistics for plan_cost
    if 'metrics.plan_cost' not in df.columns:
        print("Warning: metrics.plan_cost not found in data")
        return pd.DataFrame()

    grouped = df.groupby('case_id').agg({
        'metrics.plan_cost': ['mean', 'std', 'min', 'max', 'count'],
        'metrics.success': ['mean'],
        'params.benchmark_name': 'first',
        'params.case_idx': 'first',
    }).reset_index()

    # Flatten column names
    grouped.columns = ['case_id', 'plan_cost_mean', 'plan_cost_std', 'plan_cost_min',
                       'plan_cost_max', 'repeat_count', 'success_rate', 'benchmark', 'case_idx']

    # Sort by benchmark and case index
    grouped = grouped.sort_values(['benchmark', 'case_idx'])

    print(f"\nGrouped into {len(grouped)} unique cases")
    print(f"Benchmarks: {grouped['benchmark'].unique()}")

    return grouped


def create_violin_plots_by_benchmark(df):
    """Create horizontal violin plots for each benchmark, with cases stacked vertically."""
    if 'case_id' not in df.columns:
        df['case_id'] = df['params.benchmark_name'] + '_case_' + df['params.case_idx'].astype(str)

    if 'metrics.plan_cost' not in df.columns:
        return []

    # Get unique benchmarks
    benchmarks = sorted(df['params.benchmark_name'].unique())

    figures = []

    for benchmark in benchmarks:
        # Filter data for this benchmark
        bench_df = df[df['params.benchmark_name'] == benchmark].copy()

        # Convert case_idx to int for proper sorting
        bench_df['case_idx_int'] = bench_df['params.case_idx'].astype(int)

        # Get unique cases sorted by integer case index (reversed for low-to-high display)
        unique_cases = bench_df.groupby('case_idx_int').first().sort_index()
        case_order = unique_cases.index.tolist()[::-1]  # Reverse for low-to-high

        # Create horizontal violin plot
        fig = go.Figure()

        # Prepare annotations for case parameters
        annotations = []

        # Add a violin for each case (horizontal orientation)
        for y_position, case_idx in enumerate(case_order):
            case_data = bench_df[bench_df['case_idx_int'] == case_idx]

            # Add violin with individual points
            fig.add_trace(go.Violin(
                y=[f"Case {case_idx}"] * len(case_data),
                x=case_data['metrics.plan_cost'],
                name=f"Case {case_idx}",
                orientation='h',

                # Violin appearance
                fillcolor='rgba(31, 119, 180, 0.08)',  # translucent blue
                line=dict(color='rgba(31, 119, 180, 0.4)', width=0.5),

                # Box & mean
                box_visible=False,
                meanline_visible=False,

                # Points
                points='all',
                pointpos=0,
                jitter=0.5,
                marker=dict(
                    size=6,
                    color='rgba(20, 20, 20, 0.8)',   # dark points
                    line=dict(
                        color='rgba(255, 255, 255, 0.5)',
                        width=0.5
                    )
                ),

                showlegend=False,
            ))

            # Get case parameters for annotation
            case_row = case_data.iloc[0]
            params = {}
            for col in bench_df.columns:
                if col.startswith('params.') and col not in ['params.benchmark_name', 'params.case_idx', 'params.repeat_idx']:
                    param_name = col.replace('params.', '')
                    params[param_name] = case_row[col]

            # Format parameters
            param_parts = [f"{k}={v}" for k, v in params.items()
                           if v is not None]
            param_str = ", ".join(param_parts)

            # Compute a good x position for the annotation (in data coordinates)
            case_max = case_data['metrics.plan_cost'].max()
            case_min = case_data['metrics.plan_cost'].min()

            # Put text just to the right of the case's max (or a small constant if span is 0)
            annotations.append(dict(
                x=bench_df['metrics.plan_cost'].min(),
                y=f"Case {case_idx}",   # use the same categorical y value as the violin
                xref='x',               # data coordinates
                yref='y',               # categorical axis coordinates
                text=param_str,
                showarrow=False,
                xanchor='left',
                yanchor='bottom',       # sit slightly above the centerline; adjust if desired
                yshift=8,
                font=dict(size=10, family='monospace'),
                borderpad=0,
            ))

        # Update layout for this benchmark
        height = max(280, len(case_order) * 45)  # Adjust height based on number of cases
        fig.update_layout(
            title=f"{benchmark} - Plan Cost Distribution",
            xaxis_title="Plan Cost",
            height=height,
            margin=dict(l=10, r=10, t=40, b=10),
            annotations=annotations,
            font=dict(size=10, family='monospace'),
        )
        xmax = bench_df['metrics.plan_cost'].max()
        xmin = bench_df['metrics.plan_cost'].min()
        fig.update_xaxes(range=[xmin * 0.95, xmax * 1.05 if xmax != 0 else 1])
        figures.append({
            'benchmark': benchmark,
            'figure': fig,
        })

    return figures


# Initialize app
app = dash.Dash(__name__)

# Load data
try:
    df, experiment_name = load_latest_experiment()
    grouped_df = prepare_grouped_data(df)
    violin_plots = create_violin_plots_by_benchmark(df)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Create layout
layout_children = [
    html.H1(f"Benchmark Results: {experiment_name}"),
    html.P(f"Total runs: {len(df)} | Unique cases: {len(grouped_df)}"),
    html.Hr(),
]

# Add violin plot for each benchmark
for plot_info in violin_plots:
    # Add the violin plot (case parameters are now shown as annotations)
    layout_children.extend([
        html.Div([
            dcc.Graph(figure=plot_info['figure']),
        ]),
        html.Hr(),
    ])

# Add summary statistics section
layout_children.extend([
    html.Div([
        html.H3("Summary Statistics"),
        html.Pre(grouped_df.to_string()),
    ]),
    html.Hr(),
])

# Add raw data sample
layout_children.append(
    html.Div([
        html.H3("Raw Data Sample"),
        html.Pre(df[['params.benchmark_name', 'params.case_idx', 'params.repeat_idx',
                     'metrics.plan_cost', 'metrics.success']].head(10).to_string()),
    ])
)

app.layout = html.Div(layout_children, style={'padding': '20px'})


if __name__ == '__main__':
    print("\nStarting Dash server...")
    print("Open http://127.0.0.1:8050/ in your browser")
    app.run(debug=True)

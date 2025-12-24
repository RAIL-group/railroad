"""
Dash layout building functions.
"""

import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc

from .styles import (
    FONT_FAMILY,
    FONT_SIZE,
    CATPPUCCIN_BASE,
    CATPPUCCIN_SAPPHIRE,
    TEXT_COLOR,
    SYMBOL_SUCCESS,
    SYMBOL_FAILURE,
)
from .helpers import build_experiment_summary_block


def create_log_modal() -> dbc.Modal:
    """Create the modal component for displaying log.html artifacts."""
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle(id="modal-title", className="pre-text"),
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
            "fontSize": f"{FONT_SIZE}px",
            "backgroundColor": CATPPUCCIN_BASE,
            "color": TEXT_COLOR,
            "minHeight": "100vh",
        },
    )


def build_content_layout(
    experiment_name: str,
    figures: list[dict],
    df: pd.DataFrame,
    metadata: dict = None,
    summary: dict = None,
    sweep_plots_by_benchmark: dict = None,
) -> list:
    """
    Build the content layout with all graphs and data sample.

    Args:
        experiment_name: Name of the experiment
        figures: List of figure dicts from create_violin_plots_by_benchmark
        df: DataFrame for raw data sample
        metadata: Optional experiment metadata with benchmark descriptions
        summary: Optional experiment summary with stats
        sweep_plots_by_benchmark: Dict mapping benchmark_name -> list of sweep plot dicts

    Returns:
        List of Dash components
    """
    children = [
        dcc.Link("← Back to Experiment List", href="/", className="link", style={
            "marginBottom": "10px",
            "display": "block",
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
            className="pre-text text-base text-bold text-underline"
        ))

        # Total runs and success rate
        success_rate_class = "status-success" if summary['success_rate'] > 0.8 else ("status-error" if summary['success_rate'] > 0.5 else "status-failure")
        summary_children.append(html.Pre([
            "  Total runs: ",
            html.Span(f"{summary['total_runs']}", style={"color": CATPPUCCIN_SAPPHIRE}),
            " | Success rate: ",
            html.Span(f"{summary['success_rate']:.1%}", className=success_rate_class),
        ], className="pre-text text-base"))

        # Benchmarks with clickable links
        if summary.get("benchmarks"):
            summary_children.append(html.Pre(
                "  Benchmarks:",
                className="pre-text text-base"
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
                    status_parts.append(html.Span(f"{SYMBOL_SUCCESS}{n_success}", className="status-success"))
                    status_parts.append("/")
                if n_failure > 0:
                    status_parts.append(html.Span(f"{SYMBOL_FAILURE}{n_failure}", className="status-failure"))
                    status_parts.append("/")

                # Benchmark line with clickable link
                summary_children.append(html.Pre([
                    "    ",
                    html.A(bench_name, href=f"#benchmark-{bench_name}", className="link"),
                    " ",
                    *status_parts,
                    f"{n_total} ",
                    html.Span(f"({success_rate:.1%})", className="text-dimmed"),
                ], className="pre-text text-base"))

                # Description if available
                if description:
                    summary_children.append(html.Pre(
                        f"      {description}",
                        className="pre-text text-base text-secondary"
                    ))

        children.extend(summary_children)
    else:
        # Fallback if no summary available
        children.append(html.Pre(
            f"# Benchmark Results: {experiment_name}",
            className="pre-text text-base text-bold"
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
                status_parts.append(html.Span(f"{SYMBOL_SUCCESS}{n_success}", className="status-success"))
                status_parts.append("/")
            if n_failure > 0:
                status_parts.append(html.Span(f"{SYMBOL_FAILURE}{n_failure}", className="status-failure"))
                status_parts.append("/")

            # Benchmark header with anchor
            children.append(html.Div(id=f"benchmark-{bench_name}"))
            children.append(html.Pre([
                html.Span(f"## {bench_name}", className="text-bold text-underline"),
                " ",
                *status_parts,
                f"{n_total} ",
                html.Span(f"({success_rate:.1%})", className="text-dimmed"),
            ], className="pre-text text-base"))

            if description:
                children.append(html.Pre(
                    f"   {description}",
                    className="pre-text text-base text-secondary"
                ))

        children.append(html.Br())
        children.append(html.Pre(
            "Plan Cost",
            className="pre-text text-base"
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

        # Add sweep plots for this benchmark if available
        if sweep_plots_by_benchmark and bench_name in sweep_plots_by_benchmark:
            sweep_plots = sweep_plots_by_benchmark[bench_name]
            if sweep_plots:
                children.append(html.Pre(
                    "   Parameter Sweeps:",
                    className="pre-text text-base"
                ))

                # Create 2-column grid
                grid_children = []
                for plot_info in sweep_plots:
                    param_name = plot_info["param"].replace("params.", "")
                    analysis = plot_info["analysis"]

                    # Correlation annotation
                    corr_text = ""
                    if analysis.correlation is not None:
                        corr_text = f" (r = {analysis.correlation:.3f})"

                    grid_children.append(
                        html.Div([
                            html.Pre(
                                f"      {param_name}{corr_text}",
                                className="pre-text text-base"
                            ),
                            dcc.Graph(
                                id={"type": "sweep-graph", "benchmark": bench_name, "param": param_name},
                                figure=plot_info["figure"],
                                config={
                                    "displayModeBar": False,
                                    "responsive": False,
                                    "staticPlot": False,
                                },
                                style={"height": "300px"},  # Enforce fixed height
                            ),
                        ], style={"padding": "10px"})
                    )

                children.append(html.Div(
                    grid_children,
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(2, 1fr)",
                        "gap": "20px",
                    }
                ))
                children.append(html.Br())

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
                className="pre-text status-failure"
            )
        ]

    # Build formatted output with Rich-style colors
    children = []

    # Header
    children.append(html.Pre(
        f"Benchmark Experiments (showing {len(experiments)} most recent)",
        className="pre-text text-base"
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

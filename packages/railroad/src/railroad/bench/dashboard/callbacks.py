"""
Dash callback functions and Flask routes.
"""

import time
import json
from dash import Output, Input, State, ALL, callback_context, html
from flask import send_file
import mlflow

from railroad.bench.analysis import BenchmarkAnalyzer
from .data import (
    load_all_experiments_with_summaries,
    load_experiment_by_name,
    _artifact_cache,
)
from .figures import create_violin_plots_by_benchmark
from .layouts import build_content_layout, build_experiment_list_layout
from .sweeps import create_all_sweep_plots


def register_callbacks(app):
    """
    Register all Dash callbacks and Flask routes for the app.

    Args:
        app: Dash app instance
    """

    # Flask route to serve artifact files
    @app.server.route("/artifact/<run_id>/<path:filename>")
    def serve_artifact(run_id: str, filename: str):
        """Serve an artifact file for a given run."""
        cache_key = f"{run_id}/{filename}"

        if cache_key not in _artifact_cache:
            downloaded = False

            # Try gzipped version first for HTML files
            if filename.endswith('.html'):
                try:
                    local_path = mlflow.artifacts.download_artifacts(  # type: ignore[possibly-missing-attribute]
                        run_id=run_id,
                        artifact_path=filename + '.gz',
                    )
                    _artifact_cache[cache_key] = local_path
                    downloaded = True
                except Exception:
                    pass

            if not downloaded:
                try:
                    local_path = mlflow.artifacts.download_artifacts(  # type: ignore[possibly-missing-attribute]
                        run_id=run_id,
                        artifact_path=filename,
                    )
                    _artifact_cache[cache_key] = local_path
                except Exception as e:
                    print(f"Could not download artifact {filename} for {run_id}: {e}")
                    return f"Artifact not found: {e}", 404

        local_path = _artifact_cache[cache_key]

        # Serve gzipped files with proper headers for transparent decompression
        if local_path.endswith('.gz'):
            response = send_file(local_path)
            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Content-Type'] = 'text/html'
            return response

        return send_file(local_path)

    @app.callback(
        [Output("main-content", "children"), Output("data-store", "data")],
        [Input("url", "pathname")],
        prevent_initial_call=False,
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

                    # Generate sweep plots
                    sweep_plots_by_benchmark = create_all_sweep_plots(df)

                    content = build_content_layout(
                        experiment_name, figures, df, metadata, summary,
                        sweep_plots_by_benchmark=sweep_plots_by_benchmark
                    )

                    # Store minimal data for click handling
                    # Just store run_id to artifact_uri mapping
                    store_data = {}
                    if "run_id" in df.columns and "artifact_uri" in df.columns:
                        for _, row in df.iterrows():
                            store_data[row["run_id"]] = row.get("artifact_uri", "")

                    return content, store_data
                except ValueError as e:
                    return [html.Pre(
                        f"Experiment not found: {e}",
                        className="pre-text status-failure"
                    )], {}

            else:
                # Unknown path
                return [html.Pre(
                    f"Page not found: {pathname}",
                    className="pre-text status-failure"
                )], {}

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return [html.Div([
                html.H3("Error loading data:", className="status-failure"),
                html.Pre(str(e), className="pre-text text-base"),
                html.Details([
                    html.Summary("Full traceback", className="pre-text text-base"),
                    html.Pre(error_detail, className="pre-text", style={"fontSize": "10px"}),
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

                                children = []

                                # Only include plot image if the artifact exists
                                try:
                                    mlflow.artifacts.download_artifacts(  # type: ignore[possibly-missing-attribute]
                                        run_id=run_id,
                                        artifact_path="plot.jpg",
                                    )
                                    plot_url = f"/artifact/{run_id}/plot.jpg?t={timestamp}"
                                    children.append(html.Img(
                                        src=plot_url,
                                        style={
                                            "width": "100%",
                                            "maxHeight": "50vh",
                                            "objectFit": "contain",
                                            "marginBottom": "10px",
                                        },
                                        key=f"plot-{run_id}-{timestamp}",
                                    ))
                                except Exception:
                                    pass

                                # Create a new iframe component with unique key to force reload
                                children.append(html.Iframe(
                                    src=artifact_url,
                                    style={
                                        "width": "100%",
                                        "height": "80vh",
                                        "border": "none",
                                        "zoom": "0.75",
                                    },
                                    key=f"iframe-{run_id}-{timestamp}",  # Unique key forces recreation
                                ))

                                return True, html.Div(children), title
            except Exception as e:
                print(f"Error parsing trigger ID: {e}")

        return False, None, ""


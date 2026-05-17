"""
Main entry point for the benchmark dashboard.
"""

from pathlib import Path

import dash
import dash_bootstrap_components as dbc

from .styles import STYLE_CLASSES
from .layouts import create_main_layout
from .callbacks import register_callbacks


def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    # Inject custom CSS styles
    app.index_string = f'''<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <link href="https://cdnjs.cloudflare.com/ajax/libs/Iosevka/6.0.0/iosevka/iosevka.min.css" rel="stylesheet">
        <style>
{STYLE_CLASSES}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

    app.layout = create_main_layout()
    register_callbacks(app)

    return app


# Module-level app for `uv run bench/bench/dashboard/app.py`
app = create_app()


def _reloader_scope() -> dict:
    """Restrict the Werkzeug reloader to the ``dashboard/`` directory.

    By default the reloader watches every imported module, so editing any file
    in the (large) ``railroad`` package restarts the dashboard. We exclude the
    rest of the package and watch only ``dashboard/`` so unrelated edits don't
    trigger a restart.

    Werkzeug's ``exclude_patterns`` are fnmatch globs where ``*`` also matches
    ``/``, so a single pattern can't say "railroad but not dashboard". Instead
    we exclude each non-dashboard sibling path explicitly (their absolute paths
    are never prefixes of the dashboard path, so dashboard files survive the
    filter), and pass the dashboard ``.py`` files via ``extra_files``.
    """
    dashboard_dir = Path(__file__).resolve().parent
    bench_dir = dashboard_dir.parent
    railroad_pkg = bench_dir.parent

    exclude: list[str] = []
    for child in railroad_pkg.iterdir():
        if child != bench_dir:
            exclude.append(f"{child}*")
    for child in bench_dir.iterdir():
        if child != dashboard_dir:
            exclude.append(f"{child}*")

    extra_files = [str(p) for p in dashboard_dir.rglob("*.py")]
    return {"exclude_patterns": exclude, "extra_files": extra_files}


def main():
    """Entry point for benchmarks-dashboard command."""
    print("\nStarting Dash server...")
    print("Open http://127.0.0.1:8050/ in your browser")
    run_options: dict = {}
    try:
        run_options = _reloader_scope()
    except Exception as e:
        # Any path/Werkzeug-signature surprise: fall back to the default
        # (watch-everything) reloader rather than failing to start.
        print(f"(reloader scoping disabled: {e})")
    app.run(debug=True, dev_tools_ui=False, host="0.0.0.0", **run_options)


if __name__ == "__main__":
    main()

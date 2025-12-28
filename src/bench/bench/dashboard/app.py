"""
Main entry point for the benchmark dashboard.
"""

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


def main():
    """Entry point for benchmarks-dashboard command."""
    print("\nStarting Dash server...")
    print("Open http://127.0.0.1:8050/ in your browser")
    app.run(debug=True, dev_tools_ui=False)


if __name__ == "__main__":
    main()

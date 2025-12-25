#!/usr/bin/env python3
"""
Interactive Plotly Dash dashboard for benchmark results visualization.

This is a thin wrapper around the bench.dashboard module.

Usage:
    uv run bench/scripts/dashboard.py
"""

from bench.dashboard import app

if __name__ == "__main__":
    print("\nStarting Dash server...")
    print("Open http://127.0.0.1:8050/ in your browser")
    app.run(debug=True, dev_tools_ui=False)

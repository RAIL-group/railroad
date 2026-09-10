"""Regression tests for parameter-sweep plotting.

Focus: non-finite sweep values must not crash figure construction. A swept
parameter logged as ``inf`` (e.g. ``time_between_arrivals`` for a "no
interruptions" baseline) used to reach ``np.log10(inf)`` and blow up the
per-point jitter with ``OverflowError: Range exceeds valid bounds``.
"""

import numpy as np
import pandas as pd

from railroad.bench.dashboard.sweeps import (
    analyze_parameter_sweep,
    create_all_sweep_plots,
    create_sweep_figure,
)


def _df(sweep_values, *, success, timeout):
    """Build a one-benchmark runs DataFrame with a single swept parameter."""
    n = len(sweep_values)
    return pd.DataFrame({
        "params.benchmark_name": ["b"] * n,
        "params.case_idx": ["0"] * n,
        "params.repeat_idx": [str(i) for i in range(n)],
        "params.time_between_arrivals": [str(v) for v in sweep_values],
        "metrics.plan_cost": [10.0 + i for i in range(n)],
        "metrics.success": [float(s) for s in success],
        "metrics.timeout": [float(t) for t in timeout],
        "metrics.wall_time": [1.0] * n,
        "run_id": [f"r{i}" for i in range(n)],
    })


def test_infinite_sweep_value_does_not_crash():
    # inf value present, plus a failure and a timeout so the jittered-x
    # branches (previously the crash site) actually execute.
    df = _df(
        [100.0, 100.0, 10000.0, float("inf"), float("inf")],
        success=[1, 0, 1, 0, 0],
        timeout=[0, 0, 0, 1, 0],
    )

    analysis = analyze_parameter_sweep(df, "params.time_between_arrivals")
    fig = create_sweep_figure(analysis)

    assert fig is not None
    # Every x coordinate placed on the figure must be finite.
    for trace in fig.data:
        xs = np.asarray(trace.x, dtype=float)
        assert np.isfinite(xs).all(), f"non-finite x in trace {trace.name!r}"


def test_zero_value_under_log_scale_does_not_crash():
    # 0 in the sweep + positive values spanning >1 order of magnitude trips the
    # log-scale branch, where np.log10(0) == -inf would poison dx.
    df = _df(
        [0.0, 5.0, 100.0, 100.0],
        success=[1, 1, 1, 0],
        timeout=[0, 0, 0, 0],
    )

    analysis = analyze_parameter_sweep(df, "params.time_between_arrivals")
    fig = create_sweep_figure(analysis)

    for trace in fig.data:
        xs = np.asarray(trace.x, dtype=float)
        assert np.isfinite(xs).all()


def test_create_all_sweep_plots_survives_infinite_value():
    df = _df(
        [111.0, 259.0, 426.0, float("inf")],
        success=[1, 0, 1, 1],
        timeout=[0, 1, 0, 0],
    )

    plots = create_all_sweep_plots(df)

    assert plots  # a figure was produced for benchmark "b"
    assert plots["b"][0]["figure"] is not None

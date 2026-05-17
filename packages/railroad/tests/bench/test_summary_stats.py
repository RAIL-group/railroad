"""Per-benchmark summary stats + their shared dashboard rendering.

Covers the avg_plan_cost / avg_wall_time fields added to the cached summary
and the shared stat-span renderer used by the landing TOC, experiment TOC,
and per-section headers.
"""

import pandas as pd

from railroad.bench.analysis import BenchmarkAnalyzer
from railroad.bench.dashboard.helpers import (
    build_benchmark_stat_spans,
    build_benchmark_stats_line,
)


def _spans_text(spans: list) -> str:
    out = []
    for s in spans:
        if isinstance(s, str):
            out.append(s)
        else:  # html.Span
            out.append(str(s.children))
    return " ".join(out)


def test_summary_includes_avg_cost_and_wall_time():
    df = pd.DataFrame({
        "params.benchmark_name": ["a", "a", "b", "b"],
        "metrics.success": [1.0, 0.0, 1.0, 1.0],
        "metrics.plan_cost": [10.0, 20.0, 4.0, 6.0],
        "metrics.wall_time": [1.0, 3.0, 2.0, 2.0],
    })
    summary = BenchmarkAnalyzer().get_experiment_summary("exp", df=df)

    a = summary["success_by_benchmark"]["a"]
    b = summary["success_by_benchmark"]["b"]
    assert a["avg_plan_cost"] == 15.0 and a["avg_wall_time"] == 2.0
    assert a["success_rate"] == 0.5 and a["total_runs"] == 2
    assert b["avg_plan_cost"] == 5.0 and b["avg_wall_time"] == 2.0


def test_summary_missing_metric_columns_yield_none():
    df = pd.DataFrame({
        "params.benchmark_name": ["a", "a"],
        "metrics.success": [1.0, 1.0],
    })
    summary = BenchmarkAnalyzer().get_experiment_summary("exp", df=df)
    a = summary["success_by_benchmark"]["a"]
    assert a["avg_plan_cost"] is None and a["avg_wall_time"] is None


def test_stat_spans_carry_only_status_and_rate():
    """Cost/time moved to their own line, not the status span."""
    text = _spans_text(build_benchmark_stat_spans({
        "success_rate": 0.5,
        "total_runs": 2,
        "avg_plan_cost": 15.0,
        "avg_wall_time": 2.0,
    }))
    assert "50.0%" in text
    assert "cost" not in text and "time" not in text


def test_stats_line_renders_cost_and_time_on_its_own_line():
    line = build_benchmark_stats_line({
        "avg_plan_cost": 15.0,
        "avg_wall_time": 2.0,
    }, indent="  ")
    assert line is not None
    text = _spans_text(line.children)
    assert line.children[0] == "  "
    assert "Avg. Cost:" in text and "15.00" in text
    assert "Avg. Time:" in text and "2.00s" in text
    assert "|" in text


def test_stats_line_single_stat_has_no_separator():
    line = build_benchmark_stats_line({"avg_plan_cost": 7.0, "avg_wall_time": None})
    text = _spans_text(line.children)
    assert "Avg. Cost:" in text and "7.00" in text
    assert "Avg. Time:" not in text and "|" not in text


def test_stats_line_none_when_no_stats():
    assert build_benchmark_stats_line({"avg_plan_cost": None, "avg_wall_time": None}) is None
    assert build_benchmark_stats_line({}) is None

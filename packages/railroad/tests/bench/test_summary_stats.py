"""Per-benchmark summary stats + their shared dashboard rendering.

Covers the avg_plan_cost / avg_wall_time fields added to the cached summary
and the shared stat-span renderer used by the landing TOC, experiment TOC,
and per-section headers.
"""

import pandas as pd

from railroad.bench.analysis import BenchmarkAnalyzer
from railroad.bench.dashboard.helpers import build_benchmark_stat_spans


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


def test_stat_spans_render_cost_and_time_when_present():
    text = _spans_text(build_benchmark_stat_spans({
        "success_rate": 0.5,
        "total_runs": 2,
        "avg_plan_cost": 15.0,
        "avg_wall_time": 2.0,
    }))
    assert "cost 15.00" in text
    assert "t 2.00s" in text
    assert "50.0%" in text


def test_stat_spans_omit_missing_stats():
    text = _spans_text(build_benchmark_stat_spans({
        "success_rate": 1.0,
        "total_runs": 1,
        "avg_plan_cost": None,
        "avg_wall_time": None,
    }))
    assert "cost" not in text
    assert "t " not in text

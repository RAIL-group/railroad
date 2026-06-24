"""Tests for partial-result capture when a benchmark task times out."""

import signal
import time

import pytest

from railroad.bench.plan import Task, TaskStatus
from railroad.bench.parallel import _execute_task_worker, TimeoutError as HarnessTimeout
from railroad.bench.registry import BenchmarkCase
from railroad.bench.benchmarks._helpers import capture_timeout_log


pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"),
    reason="signal-based timeout requires SIGALRM (Unix only)",
)


class _FakeConsole:
    def export_html(self, inline_styles=True):
        return "<partial/>"


class _FakeDashboard:
    """Minimal stand-in exposing the bits capture_timeout_log touches."""

    def __init__(self, print_raises=False):
        self.console = _FakeConsole()
        self.printed = False
        self._print_raises = print_raises

    def print_history(self):
        self.printed = True
        if self._print_raises:
            raise RuntimeError("render blew up")


def _make_task(fn, timeout=1.0):
    return Task(
        id="t::case_0_0",
        benchmark_name="t::case",
        benchmark_fn=fn,
        case_idx=0,
        repeat_idx=0,
        params={},
        timeout=timeout,
    )


# --- capture_timeout_log (unit) -------------------------------------------- #


def test_capture_timeout_log_stashes_result_and_reraises():
    case = BenchmarkCase(benchmark_name="b", case_idx=0, repeat_idx=0, params={})
    dash = _FakeDashboard()

    with pytest.raises(HarnessTimeout):
        with capture_timeout_log(case, dash):
            raise HarnessTimeout("boom")

    assert dash.printed
    result = case.get_timeout_result()
    assert result == {"success": False, "log_html": "<partial/>"}


def test_capture_timeout_log_passes_through_when_no_timeout():
    case = BenchmarkCase(benchmark_name="b", case_idx=0, repeat_idx=0, params={})
    dash = _FakeDashboard()

    with capture_timeout_log(case, dash):
        pass

    assert not dash.printed
    assert case.get_timeout_result() is None


def test_capture_timeout_log_survives_render_failure():
    """A failure while rendering the partial log must not mask the timeout."""
    case = BenchmarkCase(benchmark_name="b", case_idx=0, repeat_idx=0, params={})
    dash = _FakeDashboard(print_raises=True)

    with pytest.raises(HarnessTimeout):
        with capture_timeout_log(case, dash):
            raise HarnessTimeout("boom")

    assert case.get_timeout_result() is None


# --- worker integration ----------------------------------------------------- #


def test_timeout_logs_captured_partial_result():
    """A timed-out task logs the partial result stashed by capture_timeout_log."""

    def slow_bench(case: BenchmarkCase):
        dash = _FakeDashboard()
        with capture_timeout_log(case, dash):
            time.sleep(5)  # exceeds the 1s timeout
        return {"success": True, "log_html": "<full/>"}

    task = _execute_task_worker(_make_task(slow_bench))

    assert task.status == TaskStatus.TIMEOUT
    assert task.result == {"success": False, "log_html": "<partial/>"}


def test_timeout_without_capture_has_no_result():
    """Without capture_timeout_log, a timeout still reports TIMEOUT, no result."""

    def slow_bench(case: BenchmarkCase):
        time.sleep(5)
        return {"success": True}

    task = _execute_task_worker(_make_task(slow_bench))

    assert task.status == TaskStatus.TIMEOUT
    assert task.result is None


def test_success_path_keeps_real_result():
    """A fast task returns its real result; no partial result is stashed."""

    def fast_bench(case: BenchmarkCase):
        dash = _FakeDashboard()
        with capture_timeout_log(case, dash):
            pass
        return {"success": True, "log_html": "<full/>"}

    task = _execute_task_worker(_make_task(fast_bench, timeout=30.0))

    assert task.status == TaskStatus.SUCCESS
    assert task.result == {"success": True, "log_html": "<full/>"}


def test_benchmark_case_timeout_result_roundtrip():
    case = BenchmarkCase(benchmark_name="b", case_idx=0, repeat_idx=0, params={})
    assert case.get_timeout_result() is None

    case.set_timeout_result({"ok": True})
    assert case.get_timeout_result() == {"ok": True}

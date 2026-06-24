"""Shared helpers for benchmark execution."""

from contextlib import contextmanager

from railroad.bench.parallel import TimeoutError as _HarnessTimeout
from railroad.bench.registry import BenchmarkCase


@contextmanager
def capture_timeout_log(case: BenchmarkCase, dashboard):
    """Stash the in-progress dashboard log if the run hits the harness timeout.

    Wrap the planning loop with this. If the harness's timeout fires mid-loop it
    raises inside the ``with`` block; this renders the in-progress dashboard and
    stashes its HTML on ``case`` so the executor still logs it as an artifact
    (mirroring the CLI, where a keyboard interrupt prints the in-progress run).
    The timeout is re-raised so the run is still recorded as a timeout.

    Args:
        case: The benchmark case for this run (used to hand the log back to the
            executor).
        dashboard: The ``PlannerDashboard`` driving the run; its recording
            console is exported to HTML on timeout.
    """
    try:
        yield
    except _HarnessTimeout:
        try:
            dashboard.print_history()
            case.set_timeout_result({
                "success": False,
                "log_html": dashboard.console.export_html(inline_styles=True),
            })
        except Exception as e:
            print(f"Failed to capture partial timeout log: {e}")
        raise

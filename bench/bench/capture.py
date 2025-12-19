"""
Stdout/stderr capture utilities for benchmark execution.

Provides context managers to capture output streams.
"""

import sys
import io
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class CapturedOutput:
    """Container for captured stdout and stderr."""
    stdout: str
    stderr: str


@contextmanager
def capture_output():
    """
    Context manager to capture stdout and stderr.

    Usage:
        with capture_output() as captured:
            print("Hello")
            print("Error", file=sys.stderr)

        print(captured.stdout)  # "Hello\n"
        print(captured.stderr)  # "Error\n"

    Yields:
        CapturedOutput object with stdout and stderr strings
    """
    # Create string buffers
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    # Save original streams
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Create captured output container
    captured = CapturedOutput(stdout="", stderr="")

    try:
        # Redirect streams
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        yield captured

    finally:
        # Restore original streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Get captured content
        captured.stdout = stdout_buffer.getvalue()
        captured.stderr = stderr_buffer.getvalue()

        # Close buffers
        stdout_buffer.close()
        stderr_buffer.close()

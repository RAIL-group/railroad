"""
Parallel benchmark execution using process pools.

Provides process-based parallelism for benchmark tasks.
"""

import os
import sys
import time
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from .plan import ExecutionPlan, Task, TaskStatus
from .progress import ProgressDisplay
from .tracking import MLflowTracker
from .registry import BenchmarkCase
from .capture import capture_output


class TimeoutError(Exception):
    """Raised when a task exceeds its timeout."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Task timeout exceeded")


def _handle_interrupt(progress: ProgressDisplay, message: str = "Interrupted by user. Exiting..."):
    """Handle keyboard interrupt by cleaning up display and exiting."""
    try:
        if progress.live:
            progress.live.stop()
    except Exception:
        pass
    # Show cursor on both stdout and stderr
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()
    sys.stderr.write("\033[?25h")
    print(f"\n\n{message}", file=sys.stderr)
    sys.stderr.flush()
    os._exit(1)


def _execute_task_worker(task: Task, mlflow_uri: Optional[str] = None) -> Task:
    """
    Worker function for parallel execution.

    Executed in separate process. Mirrors the logic from runner._execute_task
    but doesn't require progress display.

    Args:
        task: Task to execute
        mlflow_uri: MLflow tracking URI

    Returns:
        Completed task with results
    """
    # Update status
    task.status = TaskStatus.RUNNING

    # Restore the benchmark function by importing from bench registry
    if task.benchmark_fn is None:
        try:
            # Use entry-point discovery to load all benchmarks
            from railroad.bench.discovery import discover_benchmarks

            # Find the benchmark by name in the registry
            all_benchmarks = discover_benchmarks()
            for bench in all_benchmarks:
                if bench.name == task.benchmark_name:
                    task.benchmark_fn = bench.fn
                    break

            if task.benchmark_fn is None:
                raise ValueError(f"Benchmark {task.benchmark_name} not found in registry")
        except Exception as e:
            task.status = TaskStatus.FAILURE
            task.error = f"Failed to load benchmark function: {e}"
            return task

    # Create BenchmarkCase
    case = BenchmarkCase(
        benchmark_name=task.benchmark_name,
        case_idx=task.case_idx,
        repeat_idx=task.repeat_idx,
        params=task.params,
    )

    # Execute with timeout and capture
    start_time = time.perf_counter()

    # Check if we can use signal-based timeout (Unix only)
    use_signal_timeout = hasattr(signal, 'SIGALRM')

    try:
        with capture_output() as captured:
            if use_signal_timeout:
                # Set timeout using signal (Unix)
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(int(task.timeout))

            try:
                result = task.benchmark_fn(case)
                task.result = result
                task.status = TaskStatus.SUCCESS

                # Check if benchmark reported failure via "success" field
                if isinstance(result, dict) and "success" in result:
                    if not result["success"]:
                        task.status = TaskStatus.FAILURE
                        task.error = "Benchmark reported success=False"

            except TimeoutError:
                task.status = TaskStatus.TIMEOUT
                task.error = f"Task exceeded timeout of {task.timeout}s"
            except Exception as e:
                task.status = TaskStatus.FAILURE
                task.error = str(e)
                import traceback
                task.stderr = (captured.stderr or "") + "\n" + traceback.format_exc()
            finally:
                if use_signal_timeout:
                    signal.alarm(0)  # Cancel alarm
                    signal.signal(signal.SIGALRM, old_handler)

        task.stdout = captured.stdout
        if task.status != TaskStatus.FAILURE:
            task.stderr = captured.stderr

    except Exception as e:
        # Catastrophic failure
        task.status = TaskStatus.FAILURE
        task.error = f"Capture failed: {str(e)}"

    task.wall_time = time.perf_counter() - start_time

    return task


class ParallelExecutor:
    """
    Parallel task execution using process pool.

    Distributes tasks across multiple worker processes.
    """

    def __init__(
        self,
        num_workers: int,
        tracker: MLflowTracker,
        progress: ProgressDisplay,
    ):
        """
        Initialize parallel executor.

        Args:
            num_workers: Number of worker processes
            tracker: MLflow tracker for logging
            progress: Progress display for updates
        """
        self.num_workers = num_workers
        self.tracker = tracker
        self.progress = progress

    def execute(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Execute tasks in parallel using process pool.

        Args:
            plan: Execution plan with tasks

        Returns:
            Completed execution plan
        """
        # Set up signal handler for immediate exit on Ctrl-C
        def signal_handler(signum, frame):
            _handle_interrupt(self.progress, "Interrupted by user. Exiting immediately...")

        old_handler = signal.signal(signal.SIGINT, signal_handler)

        # Get MLflow URI from tracker
        mlflow_uri = self.tracker.tracking_uri if hasattr(self.tracker, 'tracking_uri') else None

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks at once
                futures = {}
                for task in plan.tasks:
                    future = executor.submit(_execute_task_worker, task, mlflow_uri)  # type: ignore[arg-type]
                    futures[future] = task

                # Mark first batch as running (up to num_workers)
                tasks_to_mark = list(plan.tasks)[:self.num_workers]
                for task in tasks_to_mark:
                    self.progress.mark_task_started(task)

                # Track which tasks we've marked as started
                marked_count = len(tasks_to_mark)

                # Process completions
                try:
                    for future in as_completed(futures):
                        original_task = futures[future]

                        try:
                            completed_task = future.result()

                            # Update plan with results
                            idx = plan.tasks.index(original_task)
                            plan.tasks[idx] = completed_task

                            # Log to MLflow
                            try:
                                self.tracker.log_task(completed_task)
                            except Exception as e:
                                print(f"Warning: Failed to log task to MLflow: {e}", file=sys.stderr)

                            # Update progress (this will remove from running tasks)
                            self.progress.update_task(completed_task)

                            # Mark next task as running if we haven't marked all yet
                            if marked_count < len(plan.tasks):
                                next_task = plan.tasks[marked_count]
                                self.progress.mark_task_started(next_task)
                                marked_count += 1

                        except Exception as e:
                            # Handle worker crash
                            original_task.status = TaskStatus.FAILURE
                            original_task.error = f"Worker crashed: {e}"
                            self.progress.update_task(original_task)

                            # Mark next task as running if we haven't marked all yet
                            if marked_count < len(plan.tasks):
                                next_task = plan.tasks[marked_count]
                                self.progress.mark_task_started(next_task)
                                marked_count += 1

                except KeyboardInterrupt:
                    _handle_interrupt(self.progress)

        except KeyboardInterrupt:
            _handle_interrupt(self.progress)
        finally:
            # Restore old signal handler
            signal.signal(signal.SIGINT, old_handler)

        return plan

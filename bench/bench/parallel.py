"""
Parallel benchmark execution using process pools.

Provides process-based parallelism for benchmark tasks.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import signal
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
            # Import benchmarks module to trigger decorator registration
            import benchmarks
            from bench.registry import get_all_benchmarks

            # Find the benchmark by name in the registry
            all_benchmarks = get_all_benchmarks()
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
            except TimeoutError as e:
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
        # Get MLflow URI from tracker
        mlflow_uri = self.tracker.tracking_uri if hasattr(self.tracker, 'tracking_uri') else None

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                futures = {}
                for task in plan.tasks:
                    future = executor.submit(_execute_task_worker, task, mlflow_uri)
                    futures[future] = task

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
                                import sys
                                print(f"Warning: Failed to log task to MLflow: {e}", file=sys.stderr)

                            # Update progress
                            self.progress.update_task(completed_task)

                        except Exception as e:
                            # Handle worker crash
                            original_task.status = TaskStatus.FAILURE
                            original_task.error = f"Worker crashed: {e}"
                            self.progress.update_task(original_task)

                except KeyboardInterrupt:
                    # Cancel all pending futures
                    for future in futures:
                        future.cancel()
                    # Shutdown executor without waiting
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

        except KeyboardInterrupt:
            import sys
            print("\n\nInterrupted by user. Shutting down workers...", file=sys.stderr)
            raise

        return plan

"""
Main benchmark execution engine.

Orchestrates benchmark discovery, plan creation, execution, and logging.
"""

import time
import subprocess
import socket
import getpass
import signal
import sys
from pathlib import Path
from typing import List, Optional

from .plan import ExecutionPlan, Task, TaskStatus
from .registry import BenchmarkCase, Benchmark
from .tracking import MLflowTracker
from .progress import ProgressDisplay
from .capture import capture_output


class TimeoutError(Exception):
    """Raised when a task exceeds its timeout."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Task timeout exceeded")


class BenchmarkRunner:
    """
    Main orchestrator for benchmark execution.

    Responsibilities:
    - Create execution plan from benchmarks
    - Launch MLflow experiment
    - Execute tasks (sequential or parallel)
    - Aggregate and log results
    - Handle failures and timeouts
    """

    def __init__(
        self,
        benchmarks: List[Benchmark],
        num_repeats: int = 3,
        parallel: int = 1,
        mlflow_tracking_uri: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            benchmarks: List of benchmarks to run
            num_repeats: Number of repeats per case (default: 3)
            parallel: Number of parallel workers (default: 1)
            mlflow_tracking_uri: MLflow tracking URI (default: sqlite:///mlflow.db)
            tags: Filter benchmarks by tags (default: None, run all)
        """
        self.benchmarks = benchmarks
        self.num_repeats = num_repeats
        self.parallel = parallel
        self.tracker = MLflowTracker(tracking_uri=mlflow_tracking_uri)
        self.filter_tags = tags

    def create_plan(self) -> ExecutionPlan:
        """
        Materialize the complete execution plan.

        Expands all benchmarks × cases × repeats into individual tasks.

        Returns:
            ExecutionPlan with all tasks
        """
        tasks = []

        # Get benchmarks (optionally filtered by tags)
        if self.filter_tags:
            benchmarks = [
                b for b in self.benchmarks
                if any(tag in b.tags for tag in self.filter_tags)
            ]
        else:
            benchmarks = self.benchmarks

        for benchmark in benchmarks:
            for case_idx, params in enumerate(benchmark.cases):
                for repeat_idx in range(self.num_repeats):
                    task = Task(
                        id=f"{benchmark.name}_{case_idx}_{repeat_idx}",
                        benchmark_name=benchmark.name,
                        benchmark_fn=benchmark.fn,
                        case_idx=case_idx,
                        repeat_idx=repeat_idx,
                        params=params,
                        timeout=benchmark.timeout,
                        tags=benchmark.tags,
                    )
                    tasks.append(task)

        # Gather provenance metadata
        metadata = self._collect_metadata()

        return ExecutionPlan(tasks=tasks, metadata=metadata)

    def _collect_metadata(self) -> dict:
        """
        Collect environment provenance metadata.

        Returns:
            Dictionary with git hash, hostname, timestamp, etc.
        """
        # Git info
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parent.parent.parent,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except:
            git_hash = "unknown"

        try:
            git_dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=Path(__file__).parent.parent.parent,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip())
        except:
            git_dirty = True

        return {
            "timestamp": time.time(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "git_hash": git_hash,
            "git_dirty": git_dirty,
            "num_repeats": self.num_repeats,
            "parallel_workers": self.parallel,
            "num_benchmarks": len(self.benchmarks),
        }

    def dry_run(self, plan: ExecutionPlan):
        """
        Display execution plan without running.

        Args:
            plan: Execution plan to display
        """
        from .dryrun import format_dry_run
        format_dry_run(plan)

    def run(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Execute the plan and return completed plan with results.

        Args:
            plan: Execution plan to run

        Returns:
            Completed execution plan with results
        """
        # Create MLflow experiment with timestamp
        experiment_name = f"mrppddl_bench_{int(time.time())}"
        self.tracker.create_experiment(experiment_name, plan.metadata)

        # Start progress display
        with ProgressDisplay(plan) as progress:
            if True or self.parallel > 1:
                # Parallel execution
                from .parallel import ParallelExecutor
                executor = ParallelExecutor(
                    num_workers=self.parallel,
                    tracker=self.tracker,
                    progress=progress,
                )
                plan = executor.execute(plan)
            else:
                # Sequential execution
                plan = self._execute_sequential(plan, progress)

        return plan

    def _execute_sequential(self, plan: ExecutionPlan, progress: ProgressDisplay) -> ExecutionPlan:
        """
        Execute tasks one by one.

        Args:
            plan: Execution plan
            progress: Progress display

        Returns:
            Completed execution plan
        """
        for task in plan.tasks:
            self._execute_task(task, progress)
        return plan

    def _execute_task(self, task: Task, progress: Optional[ProgressDisplay]) -> Task:
        """
        Execute a single task with timeout, capture, and logging.

        Args:
            task: Task to execute
            progress: Progress display (optional)

        Returns:
            Completed task with results
        """
        # Update status
        task.status = TaskStatus.RUNNING

        # Mark task as started in progress display
        if progress:
            progress.mark_task_started(task)

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
                # Only override stderr if not failure (failure adds traceback)
                task.stderr = captured.stderr

        except Exception as e:
            # Catastrophic failure (capture itself failed)
            task.status = TaskStatus.FAILURE
            task.error = f"Capture failed: {str(e)}"

        task.wall_time = time.perf_counter() - start_time

        # Log to MLflow
        try:
            self.tracker.log_task(task)
        except Exception as e:
            print(f"Warning: Failed to log task to MLflow: {e}", file=sys.stderr)

        # Update progress
        if progress:
            progress.update_task(task)

        return task

"""
Execution plan data structures for benchmark harness.

Defines tasks, execution plans, and status tracking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from enum import Enum
from collections import defaultdict


class TaskStatus(Enum):
    """Status of a benchmark task."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """
    Single executable unit: one benchmark case × one repeat.

    Represents a single invocation of a benchmark function with specific parameters.
    """
    # Task identification
    id: str  # Unique: f"{benchmark_name}_{case_idx}_{repeat_idx}"
    benchmark_name: str
    benchmark_fn: Callable
    case_idx: int
    repeat_idx: int
    params: Dict[str, Any]
    timeout: float
    tags: List[str] = field(default_factory=list)

    # Execution state (populated during execution)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    wall_time: Optional[float] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

    def __str__(self):
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.benchmark_name}[{self.case_idx}]({param_str}) repeat={self.repeat_idx}"

    def __getstate__(self):
        """Custom pickle support - don't pickle the function."""
        state = self.__dict__.copy()
        # Remove the unpicklable function
        state['benchmark_fn'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickle support - function will be restored separately."""
        self.__dict__.update(state)


@dataclass
class ExecutionPlan:
    """
    Materialized execution plan: all benchmarks × cases × repeats.

    Created before execution, used for dry-run and progress tracking.
    Contains the complete list of tasks to execute and metadata about the run.
    """
    tasks: List[Task]
    metadata: Dict[str, Any]  # Git hash, timestamp, environment info

    def filter_by_tags(self, tags: List[str]) -> 'ExecutionPlan':
        """Return new plan with only tasks matching any of the given tags."""
        filtered = [t for t in self.tasks if any(tag in t.tags for tag in tags)]
        return ExecutionPlan(tasks=filtered, metadata=self.metadata)

    def group_by_benchmark(self) -> Dict[str, List[Task]]:
        """Group tasks by benchmark name."""
        grouped = defaultdict(list)
        for task in self.tasks:
            grouped[task.benchmark_name].append(task)
        return dict(grouped)

    @property
    def total_tasks(self) -> int:
        """Total number of tasks in the plan."""
        return len(self.tasks)

    @property
    def estimated_time(self) -> float:
        """
        Rough estimate of total execution time assuming sequential execution.
        Based on sum of all task timeouts.
        """
        return sum(t.timeout for t in self.tasks)

    def get_summary_stats(self) -> Dict[str, int]:
        """Get counts of tasks by status."""
        stats = {
            "total": len(self.tasks),
            "pending": sum(1 for t in self.tasks if t.status == TaskStatus.PENDING),
            "running": sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING),
            "success": sum(1 for t in self.tasks if t.status == TaskStatus.SUCCESS),
            "failure": sum(1 for t in self.tasks if t.status == TaskStatus.FAILURE),
            "timeout": sum(1 for t in self.tasks if t.status == TaskStatus.TIMEOUT),
        }
        return stats

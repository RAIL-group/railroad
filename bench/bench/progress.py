"""
Rich-based progress visualization for benchmark execution.

Provides live terminal display with progress bars and statistics.
"""

from rich.console import Console, Group
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from .plan import ExecutionPlan, Task, TaskStatus


class ProgressDisplay:
    """
    Rich-based live progress display.

    Shows overall progress + per-benchmark breakdown + statistics table.
    """

    def __init__(self, plan: ExecutionPlan):
        """
        Initialize progress display for a given execution plan.

        Args:
            plan: Execution plan with tasks to track
        """
        self.plan = plan
        self.console = Console()

        # Create progress bars
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
        )

        # Overall progress
        self.overall_task = self.progress.add_task(
            "Overall",
            total=plan.total_tasks,
        )

        # Per-benchmark progress
        self.benchmark_tasks = {}
        for benchmark_name, tasks in plan.group_by_benchmark().items():
            task_id = self.progress.add_task(
                f"  {benchmark_name}",
                total=len(tasks),
            )
            self.benchmark_tasks[benchmark_name] = task_id

        # Statistics
        self.stats = {
            "success": 0,
            "failure": 0,
            "timeout": 0,
        }

        self.live = None

    def __enter__(self):
        """Start live display."""
        self.live = Live(
            self._make_layout(),
            console=self.console,
            refresh_per_second=4
        )
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop live display."""
        if self.live:
            self.live.__exit__(exc_type, exc_val, exc_tb)
            self.live = None

    def _make_layout(self) -> Panel:
        """Create layout with progress bars + stats table."""
        # Stats table
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_row(
            f"[green]Success: {self.stats['success']}[/green]",
            f"[red]Failed: {self.stats['failure']}[/red]",
            f"[yellow]Timeout: {self.stats['timeout']}[/yellow]",
        )

        # Combine
        group = Group(
            self.progress,
            "",
            stats_table,
        )

        return Panel(group, title="Benchmark Progress", border_style="cyan")

    def update_task(self, task: Task):
        """
        Update progress after task completion.

        Args:
            task: Completed task
        """
        # Update overall
        self.progress.update(self.overall_task, advance=1)

        # Update benchmark-specific
        benchmark_task_id = self.benchmark_tasks.get(task.benchmark_name)
        if benchmark_task_id is not None:
            self.progress.update(benchmark_task_id, advance=1)

        # Update stats
        if task.status == TaskStatus.SUCCESS:
            self.stats["success"] += 1
        elif task.status == TaskStatus.FAILURE:
            self.stats["failure"] += 1
        elif task.status == TaskStatus.TIMEOUT:
            self.stats["timeout"] += 1

        # Refresh layout
        if self.live:
            self.live.update(self._make_layout())

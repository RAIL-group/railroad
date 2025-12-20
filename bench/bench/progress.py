"""
Rich-based progress visualization for benchmark execution.

Provides live terminal display with progress bars and statistics.
"""

from rich.console import Console, Group, RenderableType
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, ProgressColumn, Task as ProgressTask
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.style import Style
from rich.spinner import Spinner
from datetime import timedelta
from collections import defaultdict
from .plan import ExecutionPlan, Task, TaskStatus


class StatusBarColumn(BarColumn):
    """Custom bar column that changes color based on task status."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            complete_style="blue",  # Default in-progress color
            finished_style="green",  # Default finished color
            **kwargs
        )

    def render(self, task: ProgressTask) -> RenderableType:
        """Render bar with dynamic color based on status."""
        # Check if task has failure status
        has_failures = task.fields.get("has_failures", False)
        is_finished = task.finished

        if has_failures:
            # Turn red immediately when failure detected (like PyTorch)
            self.complete_style = Style(color="red")
            self.finished_style = Style(color="red")
        elif is_finished:
            # Green for successful completion
            self.complete_style = Style(color="green")
            self.finished_style = Style(color="green")
        else:
            # Blue while in progress with no failures
            self.complete_style = Style(color="blue")
            self.finished_style = Style(color="blue")

        return super().render(task)


class CompactTimeRemainingColumn(ProgressColumn):
    """Custom time remaining column that shows only minutes:seconds."""

    def render(self, task: ProgressTask) -> Text:
        """Render time remaining in mm:ss format."""
        remaining = task.time_remaining
        if remaining is None:
            return Text("-:--", style="progress.remaining")

        # Convert to minutes and seconds only
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)

        return Text(f"{minutes}:{seconds:02d}", style="progress.remaining")


class BenchmarkStatsColumn(ProgressColumn):
    """Custom column that shows benchmark/case stats or completed/total."""

    def __init__(self, benchmark_stats_ref, case_stats_ref, task_id_to_key_ref):
        """
        Initialize with reference to stats dicts.

        Args:
            benchmark_stats_ref: Reference to the benchmark_stats dictionary
            case_stats_ref: Reference to the case_stats dictionary
            task_id_to_key_ref: Reference to mapping from task_id to case_key
        """
        super().__init__()
        self.benchmark_stats = benchmark_stats_ref
        self.case_stats = case_stats_ref
        self.task_id_to_key = task_id_to_key_ref

    def render(self, task: ProgressTask) -> RenderableType:
        """Render the stats for this task."""
        # Check if this task has a case key mapping
        if task.id in self.task_id_to_key:
            case_key = self.task_id_to_key[task.id]
            stats = self.case_stats[case_key]
            parts = []
            if stats["success"] > 0:
                parts.append(f"[green]✓{stats['success']}[/green]")
            if stats["failure"] > 0:
                parts.append(f"[red]✗{stats['failure']}[/red]")
            if stats["timeout"] > 0:
                parts.append(f"[yellow]⏱{stats['timeout']}[/yellow]")

            result = "/".join(parts) + f"/{task.total}" if parts else f"{task.completed}/{task.total}"

            # Add aggregate metrics for cases
            metrics = []
            if stats["wall_time_count"] > 0:
                avg_wall_time = stats["wall_time_sum"] / stats["wall_time_count"]
                metrics.append(f"[dim]t={avg_wall_time:.1f}s[/dim]")
            if stats["plan_cost_count"] > 0:
                avg_plan_cost = stats["plan_cost_sum"] / stats["plan_cost_count"]
                metrics.append(f"[dim]cost={avg_plan_cost:.1f}[/dim]")

            if metrics:
                result += " " + " ".join(metrics)

            return Text.from_markup(result)

        # Check if this is a benchmark task (starts with "  " but not "    ")
        elif task.description.startswith("  ") and not task.description.startswith("    "):
            benchmark_name = task.description.strip()

            if benchmark_name in self.benchmark_stats:
                stats = self.benchmark_stats[benchmark_name]
                parts = []
                if stats["success"] > 0:
                    parts.append(f"[green]✓{stats['success']}[/green]")
                if stats["failure"] > 0:
                    parts.append(f"[red]✗{stats['failure']}[/red]")
                if stats["timeout"] > 0:
                    parts.append(f"[yellow]⏱{stats['timeout']}[/yellow]")

                if parts:
                    return Text.from_markup("/".join(parts) + f"/{task.total}")

        # Default: just show completed/total
        return Text(f" {task.completed}/{task.total}")


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

        # Per-benchmark and per-case statistics (create before Progress so we can pass to custom column)
        self.benchmark_stats = defaultdict(lambda: {
            "success": 0,
            "failure": 0,
            "timeout": 0,
        })
        self.case_stats = defaultdict(lambda: {
            "success": 0,
            "failure": 0,
            "timeout": 0,
            "wall_time_sum": 0.0,
            "wall_time_count": 0,
            "plan_cost_sum": 0.0,
            "plan_cost_count": 0,
        })

        # Mapping from progress task_id to case_key for proper stats lookup
        self.task_id_to_key = {}

        # Create progress bars with custom stats column and status-colored bars
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            StatusBarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BenchmarkStatsColumn(self.benchmark_stats, self.case_stats, self.task_id_to_key),
            CompactTimeRemainingColumn(),
        )

        # Overall progress
        self.overall_task = self.progress.add_task(
            "Overall",
            total=plan.total_tasks,
        )

        # Per-benchmark and per-case progress
        self.benchmark_tasks = {}
        self.case_tasks = {}  # Key: (benchmark_name, case_idx)

        # Get benchmark descriptions from metadata
        benchmark_descriptions = plan.metadata.get("benchmark_descriptions", {})

        for benchmark_name, tasks in plan.group_by_benchmark().items():
            # Add benchmark-level progress bar with description if available
            description = benchmark_descriptions.get(benchmark_name, "")
            if description:
                task_desc = f"  {benchmark_name}\n    [italic dim]{description}[/italic dim]"
            else:
                task_desc = f"  {benchmark_name}"

            benchmark_task_id = self.progress.add_task(
                task_desc,
                total=len(tasks),
            )
            self.benchmark_tasks[benchmark_name] = benchmark_task_id

            # Group by case and add case-level progress bars
            cases = defaultdict(list)
            for task in tasks:
                cases[task.case_idx].append(task)

            for case_idx in sorted(cases.keys()):
                case_tasks = cases[case_idx]
                num_repeats = len(case_tasks)

                # Format case parameters for display with rich syntax highlighting
                params = case_tasks[0].params
                param_parts = []
                for k, v in params.items():
                    param_parts.append(f"[cyan]{k}[/cyan]=[yellow]{v}[/yellow]")
                param_str = ", ".join(param_parts)

                case_task_id = self.progress.add_task(
                    f"    Case {case_idx}: {param_str}",
                    total=num_repeats,
                )
                case_key = (benchmark_name, case_idx)
                self.case_tasks[case_key] = case_task_id
                # Store mapping for stats lookup
                self.task_id_to_key[case_task_id] = case_key

        # Statistics
        self.stats = {
            "success": 0,
            "failure": 0,
            "timeout": 0,
        }

        # Track currently running tasks per benchmark
        self.running_tasks = defaultdict(list)

        self.live = None

    def __enter__(self):
        """Start live display."""
        self.live = Live(
            self._make_layout(),
            console=self.console,
            refresh_per_second=4,
            vertical_overflow="visible"
        )
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop live display."""
        if self.live:
            self.live.__exit__(exc_type, exc_val, exc_tb)
            self.live = None

    def _make_layout(self) -> Panel:
        """Create layout with progress bars + stats table + running tasks."""
        # Build components list
        # Only include the main progress, not individual bars if too many
        components = [self.progress]

        # Add running tasks display if any are running
        if any(self.running_tasks.values()):
            components.append("")

            # Add header
            components.append(Text("Ongoing Runs", style="bold cyan"))

            # Create table with spinner column
            running_display = Table.grid(padding=(0, 1))
            running_display.add_column()  # Spinner column
            running_display.add_column()  # Task description column

            for benchmark_name in sorted(self.running_tasks.keys()):
                tasks = self.running_tasks[benchmark_name]
                if tasks:
                    # Show benchmark name and running tasks
                    for task in tasks:
                        # Format params with rich syntax highlighting
                        param_parts = []
                        for k, v in task.params.items():
                            param_parts.append(f"[cyan]{k}[/cyan]=[yellow]{v}[/yellow]")
                        param_str = ", ".join(param_parts)

                        # Build task description: benchmark name is NOT dim, rest is dim
                        task_desc = f"[bold]{benchmark_name}[/bold] [dim]case {task.case_idx} repeat {task.repeat_idx}"
                        if param_str:
                            task_desc += f" ({param_str})"
                        task_desc += "[/dim]"

                        # Add spinner and task description using Rich's built-in Spinner
                        running_display.add_row(
                            Spinner("dots", style="cyan"),
                            task_desc
                        )

            components.append(running_display)

        # Stats table
        components.append("")
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_row(
            f"[green]Success: {self.stats['success']}[/green]",
            f"[red]Failed: {self.stats['failure']}[/red]",
            f"[yellow]Timeout: {self.stats['timeout']}[/yellow]",
        )
        components.append(stats_table)

        # Combine
        group = Group(*components)

        # Don't constrain height - let it scroll naturally
        return Panel(group, title="Benchmark Progress", border_style="cyan", height=None)

    def mark_task_started(self, task: Task):
        """
        Mark a task as started (add to running tasks display).

        Args:
            task: Task that just started
        """
        # Add to running tasks
        self.running_tasks[task.benchmark_name].append(task)

        # Refresh layout
        if self.live:
            self.live.update(self._make_layout())

    def update_task(self, task: Task):
        """
        Update progress after task completion.

        Args:
            task: Completed task
        """
        # Remove from running tasks (match by task ID since object may differ)
        running = self.running_tasks[task.benchmark_name]
        self.running_tasks[task.benchmark_name] = [
            t for t in running if t.id != task.id
        ]

        # Update overall
        self.progress.update(self.overall_task, advance=1)

        # Update benchmark-specific
        benchmark_task_id = self.benchmark_tasks.get(task.benchmark_name)
        if benchmark_task_id is not None:
            self.progress.update(benchmark_task_id, advance=1)

        # Update case-specific
        case_key = (task.benchmark_name, task.case_idx)
        case_task_id = self.case_tasks.get(case_key)
        if case_task_id is not None:
            self.progress.update(case_task_id, advance=1)

        # Update stats (overall, per-benchmark, and per-case)
        if task.status == TaskStatus.SUCCESS:
            self.stats["success"] += 1
            self.benchmark_stats[task.benchmark_name]["success"] += 1
            self.case_stats[case_key]["success"] += 1
        elif task.status == TaskStatus.FAILURE:
            self.stats["failure"] += 1
            self.benchmark_stats[task.benchmark_name]["failure"] += 1
            self.case_stats[case_key]["failure"] += 1

            # Mark tasks as having failures for red bar color
            self.progress.update(self.overall_task, has_failures=True)
            if benchmark_task_id is not None:
                self.progress.update(benchmark_task_id, has_failures=True)
            if case_task_id is not None:
                self.progress.update(case_task_id, has_failures=True)
        elif task.status == TaskStatus.TIMEOUT:
            self.stats["timeout"] += 1
            self.benchmark_stats[task.benchmark_name]["timeout"] += 1
            self.case_stats[case_key]["timeout"] += 1

            # Mark tasks as having failures for red bar color
            self.progress.update(self.overall_task, has_failures=True)
            if benchmark_task_id is not None:
                self.progress.update(benchmark_task_id, has_failures=True)
            if case_task_id is not None:
                self.progress.update(case_task_id, has_failures=True)

        # Track aggregate metrics for cases (wall_time and plan_cost)
        if task.wall_time is not None:
            self.case_stats[case_key]["wall_time_sum"] += task.wall_time
            self.case_stats[case_key]["wall_time_count"] += 1

        if task.result and isinstance(task.result, dict) and "plan_cost" in task.result:
            self.case_stats[case_key]["plan_cost_sum"] += float(task.result["plan_cost"])
            self.case_stats[case_key]["plan_cost_count"] += 1

        # Refresh layout
        if self.live:
            self.live.update(self._make_layout())

"""
Rich-based progress visualization for benchmark execution.

Provides live terminal display with progress bars and statistics.
"""

import os
from rich.console import Console, Group, RenderableType
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, ProgressColumn, Task as ProgressTask
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.style import Style
from collections import defaultdict
from .plan import ExecutionPlan, Task, TaskStatus


def _is_headless_environment() -> bool:
    """Detect if running in a headless environment where live dashboards don't work well.

    Checks for:
    - CI environments (GitHub Actions, GitLab CI, etc.) via CI env var
    - Claude Code via CLAUDECODE env var
    - Google Colab via COLAB_RELEASE_TAG env var
    - Jupyter notebooks via JPY_PARENT_PID or VSCODE_PID env vars
    """
    # CI environments (GitHub Actions, GitLab, Jenkins, etc.)
    if os.environ.get("CI"):
        return True

    # Claude Code
    if os.environ.get("CLAUDECODE"):
        return True

    # Google Colab
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True

    # Jupyter environments
    if os.environ.get("JPY_PARENT_PID"):
        return True

    return False


def get_cases_per_page() -> int:
    """Get the number of cases to display/process per page.

    In interactive mode, shows 20 cases per page for efficient batching.
    In non-interactive mode (CI, Claude Code, etc.), uses 1 case per page
    so each case completes all repeats before moving to the next.
    """
    return 1 if _is_headless_environment() else 20


class StatusBarColumn(BarColumn):
    """Segmented bar coloured by per-status counts on the task.

    Reads ``success_count`` / ``failure_count`` / ``error_count`` /
    ``timeout_count`` from ``task.fields`` and draws each as a proportional
    block. The unfinished portion is shown dim. If the producer has not yet
    pushed any counts (e.g. before the first task completes) the bar falls
    back to a plain blue completed / dim remaining rendering driven by
    ``task.completed`` and ``task.total``.
    """

    # Order chosen so the eye reads: good → degraded → bad.
    _SEGMENT_ORDER = ("success", "error", "failure", "timeout")
    _SEGMENT_STYLES = {
        "success": Style(color="dodger_blue1"),
        "error": Style(color="red3", bold=True),
        "failure": Style(color="red3"),
        "timeout": Style(color="yellow"),
    }
    _PENDING_STYLE = Style(color="bright_black")
    _IN_PROGRESS_STYLE = Style(color="dodger_blue1")

    def __init__(self):
        super().__init__()

    def render(self, task: "ProgressTask"):
        width = self.bar_width if isinstance(self.bar_width, int) else 40
        total = int(task.total) if task.total else 0
        if total <= 0 or width <= 0:
            return Text(" " * max(width, 0))

        counts = {
            name: int(task.fields.get(f"{name}_count", 0))
            for name in self._SEGMENT_ORDER
        }
        done = sum(counts.values())

        if done == 0:
            # No per-status info yet: fall back to a simple completed/pending
            # bar so something animates while the first task is running.
            in_progress = min(int(task.completed or 0), total)
            n_done = int(round(in_progress / total * width))
            bar = Text()
            bar.append("━" * n_done, style=self._IN_PROGRESS_STYLE)
            bar.append("━" * (width - n_done), style=self._PENDING_STYLE)
            return bar

        # Cells per status, proportional to its share of total. Floor first,
        # then hand out leftover cells (from rounding) to the largest
        # fractional remainders so the completed portion fills exactly
        # round(done/total * width) cells.
        target_done_cells = int(round(done / total * width))
        target_done_cells = min(target_done_cells, width)
        raw = {k: counts[k] / total * width for k in self._SEGMENT_ORDER}
        cells = {k: int(raw[k]) for k in self._SEGMENT_ORDER}
        # A status with a non-zero count should be visible if any segment will be.
        for k in self._SEGMENT_ORDER:
            if counts[k] > 0 and cells[k] == 0 and target_done_cells > sum(cells.values()):
                cells[k] = 1
        leftover = target_done_cells - sum(cells.values())
        if leftover > 0:
            remainders = sorted(
                ((raw[k] - cells[k], k) for k in self._SEGMENT_ORDER),
                reverse=True,
            )
            i = 0
            while leftover > 0 and i < len(remainders):
                cells[remainders[i][1]] += 1
                leftover -= 1
                i += 1
        elif leftover < 0:
            # Over-allocated: trim from the smallest visible segments.
            shrinkables = sorted(
                ((cells[k], k) for k in self._SEGMENT_ORDER if cells[k] > 0)
            )
            i = 0
            while leftover < 0 and shrinkables:
                _, k = shrinkables[i % len(shrinkables)]
                if cells[k] > 0:
                    cells[k] -= 1
                    leftover += 1
                i += 1

        bar = Text()
        for k in self._SEGMENT_ORDER:
            if cells[k] > 0:
                bar.append("━" * cells[k], style=self._SEGMENT_STYLES[k])
        pending = width - sum(cells[k] for k in self._SEGMENT_ORDER)
        if pending > 0:
            bar.append("━" * pending, style=self._PENDING_STYLE)
        return bar


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
                parts.append(f"[dodger_blue1]✓{stats['success']}[/dodger_blue1]")
            if stats["error"] > 0:
                parts.append(f"[red3 bold]⚠{stats['error']}[/red3 bold]")
            if stats["failure"] > 0:
                parts.append(f"[red3]✗{stats['failure']}[/red3]")
            if stats["timeout"] > 0:
                parts.append(f"[yellow]⏱{stats['timeout']}[/yellow]")

            result = "/".join(parts) + f"/{task.total}" if parts else f" {task.completed}/{task.total}"

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
                    parts.append(f"[dodger_blue1]✓{stats['success']}[/dodger_blue1]")
                if stats["error"] > 0:
                    parts.append(f"[red3 bold]⚠{stats['error']}[/red3 bold]")
                if stats["failure"] > 0:
                    parts.append(f"[red3]✗{stats['failure']}[/red3]")
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

    In non-interactive mode (CI, Claude Code, Colab, piped output),
    falls back to printing results one case at a time as they complete.
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        num_workers: int = 1,
        force_interactive: bool | None = None,
    ):
        """
        Initialize progress display for a given execution plan.

        Args:
            plan: Execution plan with tasks to track
            num_workers: Number of parallel workers
            force_interactive: Override interactive detection (True=live dashboard,
                               False=simple output, None=auto-detect)
        """
        self.plan = plan
        self.console = Console()
        self.num_workers = num_workers
        self.running_tasks = 0  # Track currently running tasks

        # Determine if we should use interactive mode
        if force_interactive is not None:
            self._interactive = force_interactive
        elif _is_headless_environment():
            self._interactive = False
        else:
            self._interactive = self.console.is_interactive

        # Number of cases to show per page (shared with runner.py via get_cases_per_page())
        self.cases_per_page = get_cases_per_page()

        # Per-benchmark and per-case statistics
        self.benchmark_stats = defaultdict(lambda: {
            "success": 0,
            "failure": 0,
            "error": 0,  # Hard errors (exceptions/crashes)
            "timeout": 0,
        })
        self.case_stats = defaultdict(lambda: {
            "success": 0,
            "failure": 0,
            "error": 0,  # Hard errors (exceptions/crashes)
            "timeout": 0,
            "wall_time_sum": 0.0,
            "wall_time_count": 0,
            "plan_cost_sum": 0.0,
            "plan_cost_count": 0,
        })

        # Track tasks with hard errors for stderr printing
        self.error_tasks = []  # List of tasks with actual errors

        # Create overall progress bar
        self.overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            StatusBarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            CompactTimeRemainingColumn(),
        )
        self.overall_task = self.overall_progress.add_task(
            "Overall",
            total=plan.total_tasks,
        )

        # Get benchmark descriptions from metadata
        benchmark_descriptions = plan.metadata.get("benchmark_descriptions", {})

        # Get tags for each benchmark (from first task)
        self.benchmark_tags = {}
        for benchmark_name, tasks in plan.group_by_benchmark().items():
            if tasks:
                self.benchmark_tags[benchmark_name] = tasks[0].tags

        # Create separate Progress objects for each benchmark
        self.benchmark_progresses = {}  # benchmark_name -> Progress object
        self.benchmark_tasks = {}  # benchmark_name -> task_id in its Progress
        self.case_tasks = {}  # (benchmark_name, case_idx) -> task_id

        # Track pages for benchmarks with many cases
        self.benchmark_pages = {}  # benchmark_name -> list of case_idx lists (pages)
        self.benchmark_current_page = {}  # benchmark_name -> current page index
        self.benchmark_case_to_page = {}  # benchmark_name -> {case_idx: page_idx}

        # Track max case index for formatting alignment
        self.benchmark_max_case_idx = {}  # benchmark_name -> max case index

        for benchmark_name, tasks in plan.group_by_benchmark().items():
            # Group by case
            cases = defaultdict(list)
            for task in tasks:
                cases[task.case_idx].append(task)

            sorted_case_indices = sorted(cases.keys())

            # Track max case index for formatting
            self.benchmark_max_case_idx[benchmark_name] = max(sorted_case_indices) if sorted_case_indices else 0

            # Split cases into pages if there are too many
            pages = []
            for i in range(0, len(sorted_case_indices), self.cases_per_page):
                page_cases = sorted_case_indices[i:i + self.cases_per_page]
                pages.append(page_cases)

            self.benchmark_pages[benchmark_name] = pages
            self.benchmark_current_page[benchmark_name] = 0

            # Map each case to its page
            case_to_page = {}
            for page_idx, page_cases in enumerate(pages):
                for case_idx in page_cases:
                    case_to_page[case_idx] = page_idx
            self.benchmark_case_to_page[benchmark_name] = case_to_page

            # Create a dedicated Progress for this benchmark
            task_id_to_key = {}
            bench_progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                StatusBarColumn(),
                TextColumn("[progress.percentage]"),
                BenchmarkStatsColumn(self.benchmark_stats, self.case_stats, task_id_to_key),
                CompactTimeRemainingColumn(),
            )

            # Add benchmark-level task
            description = benchmark_descriptions.get(benchmark_name, "")
            tags = self.benchmark_tags.get(benchmark_name, [])

            # Format tags similar to dryrun.py
            tag_str = ""
            if tags:
                tag_parts = [f"[cyan]{tag}[/cyan]" for tag in tags]
                tag_str = " [" + ", ".join(tag_parts) + "]"

            if len(pages) > 1:
                # Show page info if split
                task_desc = f"  {benchmark_name}{tag_str} [dim](Page 1/{len(pages)})[/dim]"
                if description:
                    task_desc += f"\n    [italic dim]{description}[/italic dim]"
            else:
                if description:
                    task_desc = f"  {benchmark_name}{tag_str}\n    [italic dim]{description}[/italic dim]"
                else:
                    task_desc = f"  {benchmark_name}{tag_str}"

            benchmark_task_id = bench_progress.add_task(
                task_desc,
                total=len(tasks),
            )

            # Add ALL case-level progress bars (we'll filter in _make_layout)
            # Calculate width needed for case indices
            max_case_idx = self.benchmark_max_case_idx[benchmark_name]
            case_width = len(str(max_case_idx))

            for case_idx in sorted_case_indices:
                case_tasks = cases[case_idx]
                num_repeats = len(case_tasks)

                # Format case parameters
                params = case_tasks[0].params
                param_parts = []
                for k, v in params.items():
                    param_parts.append(f"[cyan]{k}[/cyan]=[yellow]{v}[/yellow]")
                param_str = ", ".join(param_parts)

                case_task_id = bench_progress.add_task(
                    f"    Case {case_idx:{case_width}d}: {param_str}",
                    total=num_repeats,
                    visible=False,  # Hidden by default, shown per page
                )
                case_key = (benchmark_name, case_idx)
                self.case_tasks[case_key] = case_task_id
                # Store mapping for stats lookup
                task_id_to_key[case_task_id] = case_key

            self.benchmark_progresses[benchmark_name] = bench_progress
            self.benchmark_tasks[benchmark_name] = benchmark_task_id

        # Statistics
        self.stats = {
            "success": 0,
            "failure": 0,
            "error": 0,  # Hard errors (exceptions/crashes)
            "timeout": 0,
        }

        # Track completed benchmarks to hide from live view
        self.completed_benchmarks = set()

        # Track total tasks per benchmark for completion detection
        self.benchmark_total_tasks = {}
        self.benchmark_completed_tasks = defaultdict(int)
        for benchmark_name, tasks in plan.group_by_benchmark().items():
            self.benchmark_total_tasks[benchmark_name] = len(tasks)

        # Track the currently active benchmark (the one being displayed)
        self.active_benchmark = None

        # Track pending tasks per benchmark to determine next active
        self.benchmark_pending_tasks = {}
        for benchmark_name, tasks in plan.group_by_benchmark().items():
            self.benchmark_pending_tasks[benchmark_name] = len(tasks)

        # Track completed tasks per page for page transition
        self.page_completed_tasks = defaultdict(lambda: defaultdict(int))  # {benchmark: {page_idx: count}}

        self.live = None

    @property
    def is_interactive(self) -> bool:
        """Return whether the display is running in interactive mode."""
        return self._interactive

    def __enter__(self) -> "ProgressDisplay":
        """Start live display (interactive mode only)."""
        if self._interactive:
            self.live = Live(
                self._make_layout(),
                console=self.console,
                refresh_per_second=4,
                vertical_overflow="visible"
            )
            self.live.__enter__()
        else:
            # Non-interactive mode: print a simple header
            self.live = None
            self.console.print("[bold]Benchmark runner starting...[/bold]")
            self.console.print(f"[dim]Running {self.plan.total_tasks} tasks with {self.num_workers} workers[/dim]")
            self.console.print()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop live display."""
        if self.live:
            self.live.__exit__(exc_type, exc_val, exc_tb)
            self.live = None

    def print_final_error_summary(self) -> None:
        """Print a final summary of all errors after benchmarks complete."""
        if not self.error_tasks:
            return

        # Print separator and header
        self.console.rule(f"\n[orange1 bold]Error Summary: errors in {len(self.error_tasks)} runs[/orange1 bold]\n")

        # Group errors by benchmark
        errors_by_benchmark = defaultdict(list)
        for task in self.error_tasks:
            errors_by_benchmark[task.benchmark_name].append(task)

        # Print errors grouped by benchmark
        for benchmark_name in sorted(errors_by_benchmark.keys()):
            tasks = errors_by_benchmark[benchmark_name]
            tags = self.benchmark_tags.get(benchmark_name, [])

            # Print benchmark header with tags
            header_text = Text()
            header_text.append(benchmark_name, style="bold")
            if tags:
                header_text.append(" [")
                for i, tag in enumerate(tags):
                    if i > 0:
                        header_text.append(", ")
                    header_text.append(tag, style="cyan")
                header_text.append("]")

            self.console.print(header_text)
            self.console.print()

            # Group tasks by identical error message and stderr
            error_groups = defaultdict(list)
            for task in tasks:
                # Use error message and stderr as key for grouping
                error_key = (task.error or "", task.stderr or "")
                error_groups[error_key].append(task)

            # Print each unique error with all affected cases
            for (error_msg, stderr), affected_tasks in error_groups.items():
                # Print affected cases/repeats
                self.console.print(f"  [orange1]⚠[/orange1] [bold]Affected {len(affected_tasks)} case(s):[/bold]")

                # Group by case and show repeats
                cases_affected = defaultdict(list)
                for task in affected_tasks:
                    cases_affected[task.case_idx].append(task)

                for case_idx in sorted(cases_affected.keys()):
                    case_tasks = cases_affected[case_idx]

                    # Get parameters from first task (all same case have same params)
                    params = case_tasks[0].params
                    param_parts = []
                    for k, v in params.items():
                        param_parts.append(f"[cyan]{k}[/cyan]=[yellow]{v}[/yellow]")
                    param_str = ", ".join(param_parts)

                    # Show which repeats failed
                    repeat_indices = sorted([t.repeat_idx for t in case_tasks])
                    if len(repeat_indices) == 1:
                        repeat_str = f"repeat {repeat_indices[0]}"
                    else:
                        repeat_str = f"repeats {', '.join(map(str, repeat_indices))}"

                    self.console.print(
                        f"    • Case {case_idx} ({repeat_str}): {param_str}"
                    )

                self.console.print()

                # Print error message
                if error_msg:
                    self.console.print(f"  [orange1]Error: {error_msg}[/orange1]")

                # Print stderr if available
                if stderr:
                    self.console.print("  [dim]stderr:[/dim]")
                    # Indent stderr lines
                    for line in stderr.strip().split('\n'):
                        self.console.print(f"    [dim]{line}[/dim]")

                self.console.print()

            self.console.print()

    def _make_layout(self) -> Panel:
        """Create layout with only active benchmark + current page + overall progress at bottom."""
        components = []

        # Show only the currently active benchmark
        if self.active_benchmark and self.active_benchmark not in self.completed_benchmarks:
            bench_progress = self.benchmark_progresses[self.active_benchmark]

            # Update visibility of case tasks based on current page
            current_page_idx = self.benchmark_current_page[self.active_benchmark]
            current_page_cases = self.benchmark_pages[self.active_benchmark][current_page_idx]

            for (bname, case_idx), case_task_id in self.case_tasks.items():
                if bname == self.active_benchmark:
                    # Show only cases in the current page
                    bench_progress.tasks[case_task_id].visible = (case_idx in current_page_cases)

            components.append(bench_progress)
            components.append("")

        # Stats table
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_row(
            f"[dodger_blue1]Success: {self.stats['success']}[/dodger_blue1]",
            f"[red3 bold]Errors: {self.stats['error']}[/red3 bold]",
            f"[red3]Failed: {self.stats['failure']}[/red3]",
            f"[yellow]Timeout: {self.stats['timeout']}[/yellow]",
            f"[dim]Workers: {self.running_tasks}/{self.num_workers}[/dim]",
        )
        components.append(stats_table)
        components.append("")

        # Overall progress bar at the bottom
        components.append(self.overall_progress)

        # Combine
        group = Group(*components)

        return Panel(group, title="Benchmark Progress", border_style="cyan", height=None)

    def _format_case_line(self, benchmark_name: str, case_idx: int, case_width: int) -> str:
        """
        Format a case summary line for printing.

        Args:
            benchmark_name: Name of the benchmark
            case_idx: Case index
            case_width: Width for case index formatting

        Returns:
            Formatted case line string with rich markup
        """
        case_key = (benchmark_name, case_idx)
        case_stats = self.case_stats[case_key]

        # Get case params
        params = {}
        for task in self.plan.tasks:
            if task.benchmark_name == benchmark_name and task.case_idx == case_idx:
                params = task.params
                break

        # Format params with rich highlighting
        param_parts = []
        for k, v in params.items():
            param_parts.append(f"[cyan]{k}[/cyan]=[yellow]{v}[/yellow]")
        param_str = ", ".join(param_parts)

        # Format stats
        result_parts = []
        if case_stats["success"] > 0:
            result_parts.append(f"[dodger_blue1]✓{case_stats['success']}[/dodger_blue1]")
        if case_stats["error"] > 0:
            result_parts.append(f"[red3 bold]⚠{case_stats['error']}[/red3 bold]")
        if case_stats["failure"] > 0:
            result_parts.append(f"[red3]✗{case_stats['failure']}[/red3]")
        if case_stats["timeout"] > 0:
            result_parts.append(f"[yellow]⏱{case_stats['timeout']}[/yellow]")

        result_str = "/".join(result_parts) if result_parts else "0"

        # Add metrics
        metrics = []
        if case_stats["wall_time_count"] > 0:
            avg_wall_time = case_stats["wall_time_sum"] / case_stats["wall_time_count"]
            metrics.append(f"[dim]t={avg_wall_time:.1f}s[/dim]")
        if case_stats["plan_cost_count"] > 0:
            avg_plan_cost = case_stats["plan_cost_sum"] / case_stats["plan_cost_count"]
            metrics.append(f"[dim]cost={avg_plan_cost:.1f}[/dim]")

        metrics_str = " " + " ".join(metrics) if metrics else ""

        return f"  Case {case_idx:{case_width}d}: {param_str}  {result_str}{metrics_str}"

    def _print_benchmark_header(self, benchmark_name: str) -> None:
        """
        Print benchmark header with name, tags, and description.

        Args:
            benchmark_name: Name of the benchmark
        """
        description = self.plan.metadata.get("benchmark_descriptions", {}).get(benchmark_name, "")
        tags = self.benchmark_tags.get(benchmark_name, [])

        # Build header with tags
        header_text = Text()
        header_text.append(benchmark_name, style="bold")
        if tags:
            header_text.append(" [")
            for i, tag in enumerate(tags):
                if i > 0:
                    header_text.append(", ")
                header_text.append(tag, style="cyan")
            header_text.append("]")

        self.console.print(header_text)

        if description:
            self.console.print(f"  [italic dim]{description}[/italic dim]")

    def _print_error_details(self, benchmark_name: str, case_indices: list) -> None:
        """
        Print stderr for error tasks in specified cases.

        Args:
            benchmark_name: Name of benchmark
            case_indices: List of case indices to check for errors
        """
        # Find error tasks for this benchmark and these cases
        error_tasks_to_print = [
            task for task in self.error_tasks
            if task.benchmark_name == benchmark_name and task.case_idx in case_indices
        ]

        if not error_tasks_to_print:
            return

        # Print header for errors
        self.console.print()
        self.console.print("[red3 bold]Errors:[/red3 bold]")

        for task in error_tasks_to_print:
            # Format task identifier
            param_parts = []
            for k, v in task.params.items():
                param_parts.append(f"[cyan]{k}[/cyan]=[yellow]{v}[/yellow]")
            param_str = ", ".join(param_parts)

            self.console.print(
                f"\n  [red3 bold]⚠[/red3 bold] Case {task.case_idx}, Repeat {task.repeat_idx}: {param_str}"
            )

            # Print error message
            if task.error:
                self.console.print(f"    [red3]Error: {task.error}[/red3]")

            # Print stderr if available
            if task.stderr:
                self.console.print("    [dim]stderr:[/dim]")
                # Indent stderr lines
                for line in task.stderr.strip().split('\n'):
                    self.console.print(f"      [dim]{line}[/dim]")

    def _print_page_summary(self, benchmark_name: str, page_idx: int) -> None:
        """
        Print completed page summary above the live view.

        In interactive mode, prints above the Live display.
        In non-interactive mode, prints directly to console.

        Args:
            benchmark_name: Name of benchmark
            page_idx: Index of completed page
        """
        page_cases = self.benchmark_pages[benchmark_name][page_idx]
        total_pages = len(self.benchmark_pages[benchmark_name])

        # Print header only on first page to avoid repetition
        if page_idx == 0:
            self._print_benchmark_header(benchmark_name)

        # Calculate width needed for case indices
        max_case_idx = self.benchmark_max_case_idx[benchmark_name]
        case_width = len(str(max_case_idx))

        # Print each case
        for case_idx in page_cases:
            self.console.print(self._format_case_line(benchmark_name, case_idx, case_width))

        # Print stderr for any error tasks in this page
        self._print_error_details(benchmark_name, page_cases)

        # Print empty line only after the last page
        if page_idx == total_pages - 1:
            self.console.print()

    def _print_benchmark_summary(self, benchmark_name: str) -> None:
        """
        Print completed benchmark summary above the live view.
        Only prints for single-page benchmarks (multi-page already printed per-page).

        In interactive mode, prints above the Live display.
        In non-interactive mode, prints directly to console.

        Args:
            benchmark_name: Name of completed benchmark
        """
        # Check if this is a multi-page benchmark
        total_pages = len(self.benchmark_pages[benchmark_name])

        # For multi-page benchmarks, we already printed each page summary
        if total_pages > 1:
            return

        # For single-page benchmarks, print the summary now
        self._print_benchmark_header(benchmark_name)

        # Calculate width needed for case indices
        max_case_idx = self.benchmark_max_case_idx[benchmark_name]
        case_width = len(str(max_case_idx))

        # Print all cases
        all_case_indices = []
        for (bname, case_idx), case_task_id in sorted(self.case_tasks.items()):
            if bname != benchmark_name:
                continue
            all_case_indices.append(case_idx)
            self.console.print(self._format_case_line(benchmark_name, case_idx, case_width))

        # Print stderr for any error tasks in this benchmark
        self._print_error_details(benchmark_name, all_case_indices)

        self.console.print()

    @staticmethod
    def _sync_segment_counts(progress: Progress, task_id, stats: dict) -> None:
        """Push per-status counts onto a progress task for segmented rendering."""
        progress.update(
            task_id,
            success_count=stats.get("success", 0),
            failure_count=stats.get("failure", 0),
            error_count=stats.get("error", 0),
            timeout_count=stats.get("timeout", 0),
        )

    def mark_task_started(self, task: Task) -> None:
        """
        Mark a task as started.

        Args:
            task: Task that just started
        """
        # Increment running task count
        self.running_tasks += 1

        # Set this benchmark as active if not already set
        if self.active_benchmark is None:
            self.active_benchmark = task.benchmark_name

        # Refresh layout to show the active benchmark (interactive mode only)
        if self._interactive and self.live:
            self.live.update(self._make_layout())

    def update_task(self, task: Task) -> None:
        """
        Update progress after task completion.

        In interactive mode, updates the live display.
        In non-interactive mode, prints case results as pages complete.

        Args:
            task: Completed task
        """
        # Decrement running task count
        self.running_tasks = max(0, self.running_tasks - 1)

        # Update overall
        self.overall_progress.update(self.overall_task, advance=1)

        # Update benchmark-specific
        benchmark_name = task.benchmark_name
        benchmark_progress = self.benchmark_progresses.get(benchmark_name)
        benchmark_task_id = self.benchmark_tasks.get(benchmark_name)
        if benchmark_progress and benchmark_task_id is not None:
            benchmark_progress.update(benchmark_task_id, advance=1)

        # Update case-specific
        case_key = (benchmark_name, task.case_idx)
        case_task_id = self.case_tasks.get(case_key)
        if benchmark_progress and case_task_id is not None:
            benchmark_progress.update(case_task_id, advance=1)

        # Update stats (overall, per-benchmark, and per-case)
        if task.status == TaskStatus.SUCCESS:
            self.stats["success"] += 1
            self.benchmark_stats[benchmark_name]["success"] += 1
            self.case_stats[case_key]["success"] += 1
        elif task.status == TaskStatus.FAILURE:
            # Differentiate between hard errors (exceptions/crashes) and soft failures (success=False)
            is_hard_error = task.error != "Benchmark reported success=False"

            if is_hard_error:
                # Hard error: exception or crash with stderr
                self.stats["error"] += 1
                self.benchmark_stats[benchmark_name]["error"] += 1
                self.case_stats[case_key]["error"] += 1
                self.error_tasks.append(task)
            else:
                # Soft failure: success=False
                self.stats["failure"] += 1
                self.benchmark_stats[benchmark_name]["failure"] += 1
                self.case_stats[case_key]["failure"] += 1
        elif task.status == TaskStatus.TIMEOUT:
            self.stats["timeout"] += 1
            self.benchmark_stats[benchmark_name]["timeout"] += 1
            self.case_stats[case_key]["timeout"] += 1

        # Push counts to each bar so StatusBarColumn can render its segments.
        self._sync_segment_counts(
            self.overall_progress, self.overall_task, self.stats
        )
        if benchmark_progress is not None and benchmark_task_id is not None:
            self._sync_segment_counts(
                benchmark_progress,
                benchmark_task_id,
                self.benchmark_stats[benchmark_name],
            )
        if benchmark_progress is not None and case_task_id is not None:
            self._sync_segment_counts(
                benchmark_progress, case_task_id, self.case_stats[case_key]
            )

        # Track aggregate metrics for cases (wall_time and plan_cost)
        if task.wall_time is not None:
            self.case_stats[case_key]["wall_time_sum"] += task.wall_time
            self.case_stats[case_key]["wall_time_count"] += 1

        if task.result and isinstance(task.result, dict) and "plan_cost" in task.result:
            self.case_stats[case_key]["plan_cost_sum"] += float(task.result["plan_cost"])
            self.case_stats[case_key]["plan_cost_count"] += 1

        # Update pending tasks count
        self.benchmark_pending_tasks[benchmark_name] -= 1

        # Track page completion
        case_page = self.benchmark_case_to_page[benchmark_name].get(task.case_idx, 0)
        self.page_completed_tasks[benchmark_name][case_page] += 1

        # Check if current page is complete
        current_page_idx = self.benchmark_current_page[benchmark_name]
        current_page_cases = self.benchmark_pages[benchmark_name][current_page_idx]
        total_pages = len(self.benchmark_pages[benchmark_name])

        # Count actual tasks in current page from the plan
        expected_tasks_in_page = sum(
            1 for t in self.plan.tasks
            if t.benchmark_name == benchmark_name and t.case_idx in current_page_cases
        )
        completed_in_page = self.page_completed_tasks[benchmark_name][current_page_idx]

        if completed_in_page >= expected_tasks_in_page:
            # Page complete
            if total_pages > 1:
                self._print_page_summary(benchmark_name, current_page_idx)

            # Move to next page if available
            if current_page_idx + 1 < total_pages:
                self.benchmark_current_page[benchmark_name] += 1
                # Update benchmark title to show new page
                new_page_idx = self.benchmark_current_page[benchmark_name]
                bench_progress = self.benchmark_progresses[benchmark_name]
                benchmark_task_id = self.benchmark_tasks[benchmark_name]

                description = self.plan.metadata.get("benchmark_descriptions", {}).get(benchmark_name, "")
                tags = self.benchmark_tags.get(benchmark_name, [])

                # Format tags
                tag_str = ""
                if tags:
                    tag_parts = [f"[cyan]{tag}[/cyan]" for tag in tags]
                    tag_str = " [" + ", ".join(tag_parts) + "]"

                task_desc = f"  {benchmark_name}{tag_str} [dim](Page {new_page_idx + 1}/{total_pages})[/dim]"
                if description:
                    task_desc += f"\n    [italic dim]{description}[/italic dim]"

                bench_progress.tasks[benchmark_task_id].description = task_desc

        # Check if benchmark is now complete
        self.benchmark_completed_tasks[benchmark_name] += 1
        if self.benchmark_completed_tasks[benchmark_name] >= self.benchmark_total_tasks[benchmark_name]:
            self._print_benchmark_summary(benchmark_name)
            self.completed_benchmarks.add(benchmark_name)

            # Move to the next benchmark with pending tasks (in plan order, not alphabetical)
            if self.active_benchmark == benchmark_name:
                self.active_benchmark = None
                for bname in self.benchmark_progresses.keys():
                    if bname not in self.completed_benchmarks and self.benchmark_pending_tasks[bname] > 0:
                        self.active_benchmark = bname
                        break

        # Refresh layout (interactive mode only)
        if self._interactive and self.live:
            self.live.update(self._make_layout())

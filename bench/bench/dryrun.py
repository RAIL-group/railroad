"""
Dry-run formatter for displaying execution plans.

Provides Rich-based tree visualization of what will be executed.
"""

from rich.console import Console
from rich.tree import Tree
from rich.table import Table
from datetime import datetime
from collections import defaultdict
from .plan import ExecutionPlan


def format_dry_run(plan: ExecutionPlan):
    """
    Format and display execution plan as rich tree.

    Args:
        plan: Execution plan to display
    """
    console = Console()

    # Header
    console.print("\n[bold cyan]Benchmark Execution Plan[/bold cyan]")
    console.print()

    # Metadata table
    meta_table = Table.grid(padding=(0, 2))
    meta_table.add_column(style="bold")
    meta_table.add_column()

    # Format timestamp
    timestamp = plan.metadata.get('timestamp')
    if timestamp:
        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timestamp_str = "unknown"

    meta_table.add_row("Total tasks:", str(plan.total_tasks))
    meta_table.add_row("Estimated time:", f"{plan.estimated_time:.1f}s (sequential)")
    meta_table.add_row("Benchmarks:", str(plan.metadata.get('num_benchmarks', 'unknown')))
    meta_table.add_row("Repeats:", str(plan.metadata.get('num_repeats', 'unknown')))
    meta_table.add_row("Parallel workers:", str(plan.metadata.get('parallel_workers', 1)))
    meta_table.add_row("Timestamp:", timestamp_str)
    meta_table.add_row("Hostname:", plan.metadata.get('hostname', 'unknown'))
    meta_table.add_row("Git hash:", plan.metadata.get('git_hash', 'unknown')[:8])
    meta_table.add_row("Git dirty:", str(plan.metadata.get('git_dirty', True)))

    console.print(meta_table)
    console.print()

    # Tree view of benchmarks
    tree = Tree("[bold]Benchmarks[/bold]", guide_style="dim")

    # Group tasks by benchmark and case
    for benchmark_name, tasks in plan.group_by_benchmark().items():
        # Count cases and repeats
        num_tasks = len(tasks)
        num_cases = len(set(t.case_idx for t in tasks))
        num_repeats = len(set(t.repeat_idx for t in tasks))

        # Get tags from first task
        tags = tasks[0].tags
        tags_display = f", tags={{{', '.join(tags)}}}" if tags else ""

        benchmark_node = tree.add(
            f"[yellow]{benchmark_name}[/yellow] "
            f"({num_cases} cases × {num_repeats} repeats = {num_tasks} tasks{tags_display})"
        )

        # Group by case
        by_case = defaultdict(list)
        for task in tasks:
            by_case[task.case_idx].append(task)

        for case_idx in sorted(by_case.keys()):
            case_tasks = by_case[case_idx]

            # Get params from first task (all tasks in same case have same params)
            params = case_tasks[0].params
            timeout = case_tasks[0].timeout

            param_str = ", ".join(f"{k}={v}" for k, v in params.items())

            case_node = benchmark_node.add(
                f"Case {case_idx}: [dim]{param_str}[/dim] "
                f"(timeout={timeout}s, repeats={len(case_tasks)})"
            )

    console.print(tree)
    console.print()

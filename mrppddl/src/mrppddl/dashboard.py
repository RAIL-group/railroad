from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.rule import Rule
from rich.text import Text

import re
from time import sleep, perf_counter
from typing import List, Dict, Set, Union

from mrppddl._bindings import (
    Goal,
    LiteralGoal,
    AndGoal,
    OrGoal,
    GoalType,
    goal_from_fluent_set,
    Fluent,
)


def split_markdown_flat(text: str) -> List[Dict[str, str]]:
    """
    Split markdown into a flat list of items:
    - {'type': 'h1', 'text': 'Heading 1'}
    - {'type': 'h2', 'text': 'Heading 2'}
    - {'type': 'text', 'text': 'multi-line text block'}

    Only first- and second-level headings (#, ##) are treated specially.
    Everything else is captured as text blocks between headings, preserving order.
    """
    pattern = re.compile(r'^(#{1,2})\s+(.*)$', re.MULTILINE)
    items: List[Dict[str, str]] = []

    last_pos = 0
    for m in pattern.finditer(text):
        # Text block before this heading
        if m.start() > last_pos:
            block = text[last_pos:m.start()].strip("\n")
            if block.strip():
                items.append(["text", block])

        hashes, heading_text = m.group(1), m.group(2).strip()
        level = len(hashes)

        if level == 1:
            items.append(["h1", heading_text])
        elif level == 2:
            items.append(["h2", heading_text])

        last_pos = m.end()

    # Trailing text after the last heading
    if last_pos < len(text):
        block = text[last_pos:].strip("\n")
        if block.strip():
            items.append(["text", block])

    return items


class PlannerDashboard:
    """
    Rich-based dashboard for the robot task planner.

    Layout:
      ┌──────────────────────────────────────────┐
      │  progress bar (goals achieved / total)   │
      ├──────────────────────────────────────────┤
      │  Left: fluents & goals   │ Right: trace  │
      └──────────────────────────┴───────────────┘
    """

    def __init__(
        self,
        goal: Union[Set[Fluent], Goal],
        initial_heuristic=None,
        console=None,
    ):
        """Initialize the planner dashboard.

        Args:
            goal: Either a set of goal fluents (for backward compatibility)
                  or a Goal object (AndGoal, OrGoal, LiteralGoal, etc.)
            initial_heuristic: Initial heuristic value for progress tracking
            console: Optional Rich console for output
        """
        if console:
            self.console = console
        else:
            self.console = Console(force_terminal=True, record=True)

        # Normalize goal input to a Goal object
        if isinstance(goal, set):
            # Convert fluent set to Goal object for uniform handling
            self.goal = goal_from_fluent_set(goal)
            self.goal_fluents = list(goal)
        else:
            self.goal = goal
            self.goal_fluents = list(goal.get_all_literals())

        self.num_goals = len(self.goal_fluents)
        self.initial_heuristic = initial_heuristic
        self._start_time = perf_counter()
        self.actions_taken = list()

        self.history: list[dict] = []
        # Root layout
        self.layout = self._make_layout()

        # Goal progress bar
        self.progress = Progress(
            TextColumn("[bold blue]{task.description:<10}"),
            BarColumn(),
            TextColumn("{task.completed:.2f}/{task.total:.2f}", justify="right"),
            TextColumn("{task.fields[extra]:>15}", justify="right"),
            expand=True,
        )
        self.goal_task_id = self.progress.add_task(
            "Goals",
            total=float(self.num_goals),
            completed=0.0,
            extra="",  # no extra text initially
        )

        # Optional heuristic task
        self.heuristic_task_id = None
        if self.initial_heuristic is not None:
            self.heuristic_task_id = self.progress.add_task(
                "Heuristic",
                total=self.initial_heuristic,   # treat "total" as initial value
                completed=0.0,                 # improvement from initial
                extra=f"h={self.initial_heuristic:.2f}",
            )

        # Initial panels
        self.layout["progress"].update(self._build_progress_panel())
        self.layout["status"].update(
            Panel("Initializing... running first planning step.", title="State", border_style="green")
        )
        self.layout["debug"].update(
            Panel("No trace yet.", title="MCTS Trace", border_style="magenta")
        )

    def __enter__(self):
        self._live = Live(
            self.renderable,
            console=Console(force_terminal=True),
            refresh_per_second=100,
            screen=True,
            auto_refresh=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if hasattr(self, "_live"):
            self._live.__exit__(exc_type, exc, tb)
            self._live = None

    def start(self):
        if not hasattr(self, "_live"):
            self.__enter__()
        return self

    def stop(self):
        if hasattr(self, "_live"):
            self.__exit__(None, None, None)

    # ------------------------------------------------------------------ #
    # Layout helpers
    # ------------------------------------------------------------------ #
    def _make_layout(self) -> Layout:
        layout = Layout(name="root")

        layout.split(
            Layout(name="progress", size=4),
            Layout(name="body", ratio=1),
        )

        layout["body"].split_row(
            Layout(name="status", ratio=2),
            Layout(name="debug", ratio=3),
        )

        return layout

    # ------------------------------------------------------------------ #
    # Panel builders
    # ------------------------------------------------------------------ #
    def _build_state_panel(
        self,
        sim_state,
        relevant_fluents,
        step_index: int | None,
        last_action_name: str | None,
    ) -> Panel:
        # Top metadata table (step, time, last action)
        meta = Table.grid(padding=(0, 1))
        meta.add_row("Cost", f"{sim_state.time:.1f}")
        meta.add_row("Elapsed Time", f"{perf_counter() - self._start_time:,.2f}")
        meta.add_row(
            "Goals reached",
            f"{self._count_achieved_goals(sim_state)}/{self.num_goals}",
        )
        # Show overall goal satisfaction for complex goals
        goal_satisfied = self._is_goal_satisfied(sim_state)
        status_str = "[green]✓ SATISFIED[/green]" if goal_satisfied else "[yellow]IN PROGRESS[/yellow]"
        meta.add_row("Goal Status", status_str)
        if step_index is not None:
            meta.add_row("Step", str(step_index))
        if last_action_name is not None:
            meta.add_row("Last action", f"[bold]{last_action_name}[/]")

        # Active fluents
        active_table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            pad_edge=False,
        )
        active_table.add_column("Selected Active Fluents")

        for f in sorted(relevant_fluents, key=lambda x: x.name):
            if f in self.goal_fluents:
                active_table.add_row(f"[green]{f}[/green]")
            else:
                active_table.add_row(str(f))

        # Goal fluents with status
        goals_table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            pad_edge=False,
        )
        goals_table.add_column("Goal Fluent")
        goals_table.add_column("Status", justify="center")

        for g in self.goal_fluents:
            achieved = g in sim_state.fluents
            if achieved:
                goals_table.add_row(f"[green]{g}[/green]", "[green]✓[/green]")
            else:
                goals_table.add_row(f"[red]{g}[/red]", "[red]✗[/red]")

        # Display the most recent actions
        actions_table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            pad_edge=False,
        )
        actions_table.add_column("#", justify="right")
        actions_table.add_column("Recently Selected Actions")
        if self.actions_taken:
            for num, action in reversed(list(enumerate(self.actions_taken))):
                actions_table.add_row(str(num), action)
            pass
        else:
            actions_table.add_row("---", "[italic]No actions selected yet.[/]")
            pass
        
        # Combine meta + subtables into one grid
        container = Table.grid(padding=1)
        container.add_row(meta)
        container.add_row(goals_table)
        container.add_row(active_table)
        container.add_row(actions_table)

        return Panel(
            container,
            title="State",
            border_style="green",
        )

    def _build_trace_panel(self, sim_state, trace_text: str) -> Panel:
        def highlighted(text: str) -> Text:
            t = Text(text)
            self.console.highlighter.highlight(t)
            return t
        content = Group(
            highlighted(str(sim_state)),
            Rule(style="dim"),
            highlighted(trace_text),
        )
        return Panel(
            content,
            title="MCTS Tree Trace",
            border_style="magenta",
            highlight=True,
        )

    def _build_progress_panel(self) -> Panel:
        # Just the single shared Progress instance
        return Panel(self.progress, title="Planner Progress", border_style="cyan")

    @property
    def renderable(self):
        """Expose the root layout so it can be passed to Live()."""
        return self.layout

    def _count_achieved_goals(self, sim_state) -> int:
        """Count how many goal literals are achieved."""
        return self.goal.goal_count(sim_state.fluents)

    def _is_goal_satisfied(self, sim_state) -> bool:
        """Check if the overall goal is satisfied."""
        return self.goal.evaluate(sim_state.fluents)

    def _get_best_branch_progress(self, sim_state) -> tuple[int, int, str]:
        """Get progress for the best satisfying path through the goal.

        For OR goals: shows progress on the branch closest to completion.
        For AND goals: sums progress across children, using best branch for each OR child.
        For AND(OR, OR, ...): shows combined progress on the optimal path.

        Returns:
            (achieved, total, label) where label describes the branch type
        """
        achieved, total = self._compute_best_path_progress(self.goal, sim_state.fluents)

        # Generate a descriptive label
        goal_type = self.goal.get_type()
        if goal_type == GoalType.OR:
            label = "Best branch"
        elif goal_type == GoalType.AND and self._has_or_children(self.goal):
            label = "Best path"
        else:
            label = "All goals"

        return achieved, total, label

    def _compute_best_path_progress(self, goal, fluents) -> tuple[int, int]:
        """Recursively compute best-path progress through a goal tree.

        For OR: returns progress of best child (highest completion ratio)
        For AND: sums best-path progress of all children
        For LITERAL: returns (1, 1) if satisfied, (0, 1) if not
        """
        goal_type = goal.get_type()

        if goal_type == GoalType.LITERAL:
            satisfied = goal.evaluate(fluents)
            return (1, 1) if satisfied else (0, 1)

        elif goal_type == GoalType.TRUE_GOAL:
            return (1, 1)

        elif goal_type == GoalType.FALSE_GOAL:
            return (0, 1)

        elif goal_type == GoalType.OR:
            # Find the child with best completion ratio
            best_achieved, best_total = 0, 1

            for child in goal.children():
                achieved, total = self._compute_best_path_progress(child, fluents)
                if total > 0 and (achieved / total) > (best_achieved / best_total):
                    best_achieved, best_total = achieved, total

            return best_achieved, best_total

        elif goal_type == GoalType.AND:
            # Sum progress across all children (using best path for each)
            total_achieved, total_count = 0, 0

            for child in goal.children():
                achieved, total = self._compute_best_path_progress(child, fluents)
                total_achieved += achieved
                total_count += total

            return total_achieved, total_count

        else:
            # Unknown goal type - fall back to literal count
            return goal.goal_count(fluents), len(goal.get_all_literals())

    def _has_or_children(self, goal) -> bool:
        """Check if a goal has any OR children (for labeling purposes)."""
        goal_type = goal.get_type()
        if goal_type == GoalType.OR:
            return True
        elif goal_type == GoalType.AND:
            return any(
                child.get_type() == GoalType.OR
                for child in goal.children()
            )
        return False

    def _update_heuristic(self, heuristic_value: float | None):
        if self.heuristic_task_id is None:
            return
        if heuristic_value is None:
            return

        h0 = self.initial_heuristic
        h_now = max(0.0, float(heuristic_value))
        # improvement is initial - current; clamp to [0, h0]
        improvement = max(0.0, min(h0, h0 - h_now))

        self.progress.update(
            self.heuristic_task_id,
            completed=improvement,
            extra=f"h={h_now:.2f} Δh={improvement:.2f}",
        )

    def _record_history_entry(
        self,
        sim_state,
        relevant_fluents,
        tree_trace: str,
        step_index: int | None,
        last_action_name: str | None,
        heuristic_value: float | None,
    ):
        # Store a lightweight, serializable snapshot
        entry = {
            "step": step_index,
            "time": float(sim_state.time),
            "last_action": last_action_name,
            "heuristic": float(heuristic_value) if heuristic_value is not None else None,
            "relevant_fluents": [str(f) for f in sorted(relevant_fluents, key=lambda x: x.name)],
            "goals": {
                str(g): bool(g in sim_state.fluents) for g in self.goal_fluents
            },
            "goal_satisfied": self._is_goal_satisfied(sim_state),
            "tree_trace": tree_trace,
        }
        self.history.append(entry)

    def format_history_as_text(self) -> str:
        """Return the full dashboard history as a multi-line string."""
        lines: list[str] = []
        for entry in self.history:
            step = entry["step"]
            t = entry["time"]
            action = entry["last_action"]

            lines.append(f"# Step {step} | t = {t:.2f}s | action = {action}")
            lines.append(f"Selected Action: {action}\n")
            if entry["tree_trace"]:
                lines.append("\n## MCTS Tree Trace:")
                # indent the trace for readability
                for line in entry["tree_trace"].splitlines():
                    lines.append(f"    {line}")
            lines.append("\n## Selected Active Fluents:")
            for f in entry["relevant_fluents"]:
                lines.append(f"    {f}")
            lines.append("## Goal Fluents:")
            for g, ok in entry["goals"].items():
                mark = "✓" if ok else "✗"
                lines.append(f"   {mark} {g}")
            # Show overall goal satisfaction
            goal_status = entry.get("goal_satisfied", False)
            status_mark = "✓ SATISFIED" if goal_status else "✗ NOT SATISFIED"
            lines.append(f"## Overall Goal: {status_mark}")
            lines.append("\n")  # blank line between steps
        
        return "\n".join(lines)

    def print_history(self, final_state=None, actions_taken=None):
        """Pretty-print the full history using Rich."""
        local_console = self.console

        if not self.history:
            local_console.print("[yellow]No history recorded.[/yellow]")
            return

        split_text = split_markdown_flat(self.format_history_as_text())
        for text_type, text in split_text:
            if text_type == 'text':
                local_console.print(text)
            elif text_type == 'h1':
                local_console.print()
                local_console.rule(text)
            elif text_type == 'h2':
                local_console.print(f"[bold red]{text}[/]")

        if final_state and actions_taken:
            if self._is_goal_satisfied(final_state):
                local_console.rule("[bold green]Success!! :: Execution Summary[/]")
            else:
                local_console.rule("[bold red]Task Not Completed :: Execution Summary[/]")
                
            local_console.print(f"[bold red]Actions Taken (Num Actions: {len(actions_taken)})[/]", highlight=True)
            for i, action in enumerate(actions_taken, 1):
                local_console.print(f"  {i}. {action}", highlight=True)
            # Check which goals were achieved
            local_console.print("\n[bold red]Goal status:[/]")
            for goal in self.goal_fluents:
                achieved = goal in final_state.fluents
                status = "✓" if achieved else "✗"
                local_console.print(f"  {status} {goal}")
            local_console.print(f"\n[bold red]Total cost (time): {final_state.time:.1f} seconds [/]")
                

    def save_history(self, path: str):
        """Save the history to a plain-text log file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.format_history_as_text())

    def update(
        self,
        sim_state,
        relevant_fluents: Set = set(),
        tree_trace: str | None = None,
        step_index: int | None = None,
        last_action_name: str | None = None,
        heuristic_value: float | None = None,
    ):
        """
        Update the whole dashboard:

        - progress bar (based on goal fluents achieved)
        - left panel: active + goal fluents
        - right panel: MCTS trace
        """
        # Use best branch progress for OR goals
        achieved, total, label = self._get_best_branch_progress(sim_state)
        self.progress.update(
            self.goal_task_id,
            completed=achieved,
            total=total,
            extra=f"{int(achieved)}/{total} ({label})",
        )

        if last_action_name:
            self.actions_taken.append(last_action_name)

        # Heuristic (optional)
        self._update_heuristic(heuristic_value)

        self.layout["progress"].update(self._build_progress_panel())
        self.layout["status"].update(
            self._build_state_panel(
                sim_state=sim_state,
                relevant_fluents=relevant_fluents,
                step_index=step_index,
                last_action_name=last_action_name,
            )
        )
        if tree_trace:
            self.layout["debug"].update(self._build_trace_panel(sim_state, tree_trace))

        sleep(0.01)

        self._record_history_entry(
            sim_state=sim_state,
            relevant_fluents=relevant_fluents,
            tree_trace=str(sim_state) + "\n\n" + tree_trace if tree_trace else None,
            step_index=step_index,
            last_action_name=last_action_name,
            heuristic_value=heuristic_value,
        )

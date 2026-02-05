from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.rule import Rule
from rich.text import Text

import os
import re
from time import sleep, perf_counter
from typing import List, Dict, Set, Union, Tuple, Optional


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

from railroad._bindings import (
    Goal,
    LiteralGoal,
    AndGoal,
    OrGoal,
    GoalType,
    Fluent,
    State,
)
from railroad.helper import format_goal, get_best_branch, get_satisfied_branch


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
                items.append({"type": "text", "text": block})

        hashes, heading_text = m.group(1), m.group(2).strip()
        level = len(hashes)

        if level == 1:
            items.append({"type": "h1", "text": heading_text})
        elif level == 2:
            items.append({"type": "h2", "text": heading_text})

        last_pos = m.end()

    # Trailing text after the last heading
    if last_pos < len(text):
        block = text[last_pos:].strip("\n")
        if block.strip():
            items.append({"type": "text", "text": block})

    return items


def action_color(action: str) -> str:
    """Return Rich color name based on action type."""
    act = action.split()[0] if action else ""
    if act == "move":
        return "blue"
    elif act in ("pick", "place"):
        return "green"
    elif act == "search":
        return "yellow"
    elif act == "no-op":
        return "gray"
    return "white"


def _shorten_name(name: str) -> str:
    """Shorten a name by keeping first letter of each word (camelCase) and trailing numbers.

    Examples:
        "crawler" -> "c"
        "robot1" -> "r1"
        "BigRedRobot" -> "BRR"
        "myRobot3" -> "mR3"
    """
    import re
    # Extract trailing digits
    match = re.match(r'^(.*?)(\d*)$', name)
    base, digits = match.groups() if match else (name, '')

    # Find camelCase word boundaries: start + uppercase letters
    initials = []
    if base:
        initials.append(base[0])
        for c in base[1:]:
            if c.isupper():
                initials.append(c)

    return ''.join(initials) + digits


def render_timeline(actions: List[Tuple[str, float]], robots: Set[str],
                    width: int = 50, end_time: Optional[float] = None) -> str:
    """Render Braille timeline. Each robot uses 2 vertical dots; 2 robots per row."""
    if not actions or not robots:
        return ""
    L, R, B = [0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80], 0x2800  # braille dots
    actions_list = list(actions)  # for indexing by action index

    # Build events: (robot, time, index) for each robot in each action
    events = []
    for i, (act, t) in enumerate(actions):
        parts = act.split()
        involved = [r for r in robots if r in parts]
        if not involved and len(parts) >= 2 and parts[1].startswith("robot"):
            involved = [parts[1]]
        events.extend((r, t, i + 1) for r in involved)
    if not events:
        return ""

    min_t, max_t = 0.0, end_time if end_time else max(e[1] for e in events)
    if max_t <= min_t:
        max_t = min_t + 1.0
    pos = lambda t: int((t - min_t) / (max_t - min_t) * (width * 2 - 1))

    robots_list = sorted(robots)
    # Build short name mapping
    short_names = {r: _shorten_name(r) for r in robots_list}

    # Calculate name width from shortened names (individual and paired) and full names for label rows
    individual_nw = max(len(short_names[r]) for r in robots_list)
    paired_nw = max(
        len(','.join(short_names[r] for r in robots_list[i:i + 2]))
        for i in range(0, len(robots_list), 2)
    )
    full_nw = max(len(r) for r in robots_list)  # full names used in label rows
    nw = max(individual_nw, paired_nw, full_nw)  # name width
    lines = [f"{' ' * nw} |{min_t:.1f}{' ' * (width - len(f'{min_t:.1f}') - len(f'{max_t:.1f}'))}{max_t:.1f}|"]

    # Braille rows (2 robots per row)
    for i in range(0, len(robots_list), 2):
        chunk = robots_list[i:i + 2]
        chars = [B] * width
        for ri, robot in enumerate(chunk):
            slot = ri * 2
            for r, t, _ in events:
                if r == robot:
                    ci, sub = pos(t) // 2, pos(t) % 2
                    if 0 <= ci < width:
                        chars[ci] |= (L if sub == 0 else R)[slot] | (L if sub == 0 else R)[slot + 1]
        lines.append(f"{','.join(short_names[r] for r in chunk):>{nw}} |{''.join(chr(c) for c in chars)}|")

    # Label rows (with color coding)
    for robot in robots_list:
        label_parts = []
        counts = {}
        for r, t, idx in events:
            if r == robot:
                ci = pos(t) // 2
                if 0 <= ci < width:
                    counts.setdefault(ci, []).append(idx)
        last_ci = -1
        for ci in sorted(counts.keys()):
            label_parts.append(" " * (ci - last_ci - 1))  # spaces before
            idxs = counts[ci]
            idx = idxs[0]
            color = action_color(actions_list[idx - 1][0])
            char = str(idx % 10) if len(idxs) == 1 else "+"
            label_parts.append(f"[{color}]{char}[/]")
            last_ci = ci
        label_parts.append(" " * (width - last_ci - 1))  # trailing spaces
        lines.append(f"{robot:>{nw}}  {''.join(label_parts)} ")

    return "\n".join(lines)


class PlannerDashboard:
    """
    Rich-based dashboard for the robot task planner.

    Layout (interactive mode):
      ┌──────────────────────────────────────────┐
      │  progress bar (goals achieved / total)   │
      ├──────────────────────────────────────────┤
      │  Left: fluents & goals   │ Right: trace  │
      └──────────────────────────┴───────────────┘

    In non-interactive mode (CI, Claude Code, Colab, piped output),
    falls back to simple text progress updates.
    """

    def __init__(
        self,
        goal: Union[Goal, Fluent],
        initial_heuristic=None,
        console=None,
        force_interactive: bool | None = None,
    ):
        """Initialize the planner dashboard.

        Args:
            goal: A Goal object (AndGoal, OrGoal, LiteralGoal, etc.),
                  or a bare Fluent (which will be wrapped in LiteralGoal)
            initial_heuristic: Initial heuristic value for progress tracking
            console: Optional Rich console for output
            force_interactive: Override interactive detection (True=live dashboard,
                               False=simple output, None=auto-detect)
        """
        if console:
            self.console = console
        else:
            self.console = Console(record=True)

        # Determine if we should use interactive mode
        # Check for explicit override first, then headless environments, then console detection
        if force_interactive is not None:
            self._interactive = force_interactive
        elif _is_headless_environment():
            self._interactive = False
        else:
            self._interactive = self.console.is_interactive

        # Normalize goal input to a Goal object
        # Wrap bare Fluent with LiteralGoal for better user experience
        if isinstance(goal, Fluent):
            goal = LiteralGoal(goal)
        self.goal = goal
        self.goal_fluents = list(goal.get_all_literals())

        self.num_goals = len(self.goal_fluents)
        self.initial_heuristic = initial_heuristic
        self._start_time = perf_counter()

        # Actions stored as (action_name, start_time) tuples
        self.actions_taken: List[Tuple[str, float]] = []
        self._last_state_time: float = 0.0  # Track previous state time for action start

        # Track known robots from (free NAME) fluents
        self.known_robots: Set[str] = set()

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
            completed=0,
            extra="",  # no extra text initially
        )

        # Optional heuristic task
        self.heuristic_task_id = None
        if self.initial_heuristic is not None:
            self.heuristic_task_id = self.progress.add_task(
                "Heuristic",
                total=self.initial_heuristic,   # treat "total" as initial value
                completed=0,                   # improvement from initial
                extra=f"h={self.initial_heuristic:.2f}",
            )

        # Initial panels (only needed for interactive mode)
        if self._interactive:
            self.layout["progress"].update(self._build_progress_panel())
            self.layout["status"].update(
                Panel("Initializing... running first planning step.", title=Text("State", style="bold"), border_style="dark_red")
            )
            self.layout["debug"].update(
                Panel("No trace yet.", title=Text("MCTS Trace", style="bold default"), border_style="dark_red")
            )

    @property
    def is_interactive(self) -> bool:
        """Return whether the dashboard is running in interactive mode."""
        return self._interactive

    def __enter__(self) -> "PlannerDashboard":
        if self._interactive:
            self._live = Live(
                self.renderable,
                console=Console(force_terminal=True),
                refresh_per_second=100,
                screen=True,
                auto_refresh=True,
            )
            self._live.__enter__()
        else:
            # Non-interactive mode: just print initial message
            self._live = None
            self.console.print("[bold]Planner starting...[/bold]")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._live is not None:
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

        # Show overall goal satisfaction for complex goals
        if step_index is not None:
            meta.add_row("Step", str(step_index))
        if last_action_name is not None:
            meta.add_row("Last action", f"[bold]{last_action_name}[/]")

        # Active fluents
        active_table = Table(
            show_header=True,
            header_style="bold blue",
            box=None,
            pad_edge=False,
        )
        active_table.add_column("Selected Active Fluents")

        for f in sorted(relevant_fluents, key=lambda x: x.name):
            if f in self.goal_fluents:
                active_table.add_row(f"[green]{f}[/green]")
            else:
                active_table.add_row(str(f))

        # Goal structure display
        goals_table = Table(
            show_header=True,
            header_style="bold blue",
            box=None,
            pad_edge=False,
        )
        goals_table.add_column("Goal Structure")

        # Show full goal with colored fluents
        full_goal_str = format_goal(self.goal, compact=True)
        for line in full_goal_str.split('\n'):
            goals_table.add_row(self._colorize_goal_line(line, sim_state.fluents))

        # Show best branch being pursued (if different from full goal)
        best_branch = get_best_branch(self.goal, sim_state.fluents)
        if best_branch != self.goal:
            best_branch_str = format_goal(best_branch, compact=True)
            goals_table.add_row("")  # blank line
            goals_table.add_row("[italic]Best Path Through Goal:[/italic]")
            for line in best_branch_str.split('\n'):
                goals_table.add_row(self._colorize_goal_line(line, sim_state.fluents))

        # Actions section: header, timeline, then action list
        actions_table = Table(
            show_header=True,
            header_style="bold blue",
            box=None,
            pad_edge=False,
        )
        actions_table.add_column("Actions Taken")

        # Timeline visualization (no separate header)
        max_by_actions = 6 * max(2, len(self.actions_taken))
        timeline_width = min(max_by_actions, max(20, (self.console.width * 2 // 5) - 15))
        timeline_str = render_timeline(
            self.actions_taken, self.known_robots,
            width=timeline_width, end_time=sim_state.time
        )
        if timeline_str:
            for line in timeline_str.split('\n'):
                actions_table.add_row(line)
            actions_table.add_row("")  # blank line before action list

        # Action list (with color coding)
        if self.actions_taken:
            for num, (action, time) in reversed(list(enumerate(self.actions_taken, 1))):
                color = action_color(action)
                actions_table.add_row(f"[{color}]{num}[/]: {action}")
        else:
            actions_table.add_row("[italic]No actions selected yet.[/]")

        # Combine meta + subtables into one grid
        container = Table.grid(padding=1)
        container.add_row(meta)
        container.add_row(goals_table)
        container.add_row(active_table)
        container.add_row(actions_table)

        return Panel(
            container,
            title=Text("State", style="bold default"),
            border_style="dark_red",
        )

    def _build_trace_panel(self, sim_state, trace_text: str) -> Panel:
        def highlighted(text: str) -> Text:
            t = Text(text)
            highlighter = self.console.highlighter
            if highlighter is not None and hasattr(highlighter, "highlight"):
                highlighter.highlight(t)  # type: ignore[union-attr]
            return t
        content = Group(
            highlighted(str(sim_state)),
            Rule(style="dim"),
            highlighted(trace_text),
        )
        return Panel(
            content,
            title=Text("MCTS Tree Trace", style="bold default"),
            border_style="dark_red",
            highlight=True,
        )

    def _build_progress_panel(self) -> Panel:
        # Just the single shared Progress instance
        return Panel(self.progress, title=Text("Planner Progress", style="bold default"), border_style="dark_red")

    @property
    def renderable(self):
        """Expose the root layout so it can be passed to Live()."""
        return self.layout

    def _colorize_goal_line(self, line: str, fluents) -> str:
        """Colorize a goal line based on whether literals are satisfied.

        Uses both color AND a checkmark (✓) for accessibility:
        - ✓ (green) for satisfied goals
        - (red) for unsatisfied goals
        """
        # Build a set of fluent strings for faster lookup
        fluent_strs = {str(f) for f in fluents}

        def colorize_match(match):
            fluent_str = match.group(0)

            # Check if this is a negated fluent like "(not at Book table)"
            if fluent_str.startswith("(not "):
                # Extract the positive fluent: "(not at Book table)" -> "(at Book table)"
                positive_fluent_str = "(" + fluent_str[5:]
                # Negative goal is satisfied if the positive fluent is NOT in the state
                if positive_fluent_str not in fluent_strs:
                    return f"[green]✓{fluent_str}[/green]"
                else:
                    return f"[red] {fluent_str}[/red]"
            else:
                # Positive fluent: satisfied if it IS in the state
                if fluent_str in fluent_strs:
                    return f"[green]✓{fluent_str}[/green]"
                else:
                    return f"[red] {fluent_str}[/red]"

        # Pattern to match fluents: (name args...) or (not name args...)
        pattern = r'\([^()]+\)'
        return re.sub(pattern, colorize_match, line)

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
        if self.initial_heuristic is None:
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
        state: State,
        relevant_fluents,
        tree_trace: str | None,
        step_index: int | None,
        last_action_name: str | None,
        heuristic_value: float | None,
    ):
        # Store a lightweight, serializable snapshot
        entry = {
            "step": step_index,
            "time": float(state.time),
            "last_action": last_action_name,
            "heuristic": float(heuristic_value) if heuristic_value is not None else None,
            "relevant_fluents": [str(f) for f in sorted(relevant_fluents, key=lambda x: x.name)],
            "goals": {
                str(g): bool(g in state.fluents) for g in self.goal_fluents
            },
            "goal_satisfied": self._is_goal_satisfied(state),
            "tree_trace": tree_trace,
        }
        self.history.append(entry)

    def _format_single_entry(self, entry: dict) -> str:
        """Format a single history entry as markdown text."""
        lines: list[str] = []
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

        return "\n".join(lines)

    def _print_entry(self, entry: dict):
        """Pretty-print a single history entry using Rich."""
        text = self._format_single_entry(entry)
        split_text = split_markdown_flat(text)
        for item in split_text:
            text_type = item['type']
            content = item['text']
            if text_type == 'text':
                self.console.print(content)
            elif text_type == 'h1':
                self.console.print()
                self.console.rule(content)
            elif text_type == 'h2':
                self.console.print(f"[bold red]{content}[/]")

    def format_history_as_text(self) -> str:
        """Return the full dashboard history as a multi-line string."""
        lines: list[str] = []
        for entry in self.history:
            lines.append(self._format_single_entry(entry))
            lines.append("\n")  # blank line between steps

        return "\n".join(lines)

    def print_history(self, final_state=None, actions_taken=None):
        """Pretty-print the full history using Rich.

        In non-interactive mode, skips the step-by-step history (already printed
        during update calls) and only prints the final summary.
        """
        local_console = self.console

        if not self.history:
            local_console.print("[yellow]No history recorded.[/yellow]")
            return

        # In interactive mode (or when there's no final_state), print full history
        # In non-interactive mode, history was already printed during updates
        if self._interactive or not final_state:
            split_text = split_markdown_flat(self.format_history_as_text())
            for item in split_text:
                text_type = item['type']
                text = item['text']
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

            # Actions section: header, timeline, then action list
            local_console.print(f"[bold blue]Actions Taken ({len(actions_taken)})[/]")

            # Timeline (no separate label)
            max_by_actions = 6 * max(2, len(self.actions_taken))
            timeline_width = min(max_by_actions, max(20, local_console.width - 15))
            timeline_str = render_timeline(
                self.actions_taken, self.known_robots,
                width=timeline_width, end_time=final_state.time
            )
            if timeline_str:
                for line in timeline_str.split('\n'):
                    local_console.print(f"  {line}")
                local_console.print()  # blank line

            # Action list (with color coding)
            for i, action in enumerate(actions_taken, 1):
                color = action_color(action)
                local_console.print(f"  [{color}]{i}[/]. {action}")

            # Show full goal structure
            local_console.print("\n[bold blue]Full Goal:[/]")
            full_goal_str = format_goal(self.goal, compact=True)
            for line in full_goal_str.split('\n'):
                local_console.print(f"  {self._colorize_goal_line(line, final_state.fluents)}")

            # Show satisfied branch (if goal was met)
            if self._is_goal_satisfied(final_state):
                satisfied_branch = get_satisfied_branch(self.goal, final_state.fluents)
                if satisfied_branch:
                    local_console.print("\n[bold green]Satisfied Branch:[/]")
                    satisfied_str = format_goal(satisfied_branch, compact=True)
                    for line in satisfied_str.split('\n'):
                        local_console.print(f"  {self._colorize_goal_line(line, final_state.fluents)}")
            else:
                # Show best branch (closest to completion)
                best_branch = get_best_branch(self.goal, final_state.fluents)
                local_console.print("\n[bold yellow]Best Branch (incomplete):[/]")
                best_str = format_goal(best_branch, compact=True)
                for line in best_str.split('\n'):
                    local_console.print(f"  {self._colorize_goal_line(line, final_state.fluents)}")

            local_console.print(f"\n[bold red]Total cost (time): {final_state.time:.1f} seconds [/]")


    def save_history(self, path: str):
        """Save the history to a plain-text log file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.format_history_as_text())

    def update(
        self,
        state: State,
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
        # Extract robots from (free NAME) fluents
        for fluent in state.fluents:
            fluent_str = str(fluent)
            if fluent_str.startswith("(free ") and fluent_str.endswith(")"):
                robot_name = fluent_str[6:-1].strip()
                self.known_robots.add(robot_name)

        # Use best branch progress for OR goals
        achieved, total, label = self._get_best_branch_progress(state)

        if last_action_name:
            # Record action with its START time (before execution)
            self.actions_taken.append((last_action_name, self._last_state_time))

        # Update last state time for next action's start time
        self._last_state_time = state.time

        if self._interactive:
            # Interactive mode: update live dashboard
            self.progress.update(
                self.goal_task_id,
                completed=achieved,
                total=total,
                extra=f"{int(achieved)}/{total} ({label})",
            )

            # Heuristic (optional)
            self._update_heuristic(heuristic_value)

            self.layout["progress"].update(self._build_progress_panel())
            self.layout["status"].update(
                self._build_state_panel(
                    sim_state=state,
                    relevant_fluents=relevant_fluents,
                    step_index=step_index,
                    last_action_name=last_action_name,
                )
            )
            if tree_trace:
                self.layout["debug"].update(self._build_trace_panel(state, tree_trace))

            sleep(0.01)

        # Record history entry (used by both modes)
        self._record_history_entry(
            state=state,
            relevant_fluents=relevant_fluents,
            tree_trace=str(state) + "\n\n" + tree_trace if tree_trace else None,
            step_index=step_index,
            last_action_name=last_action_name,
            heuristic_value=heuristic_value,
        )

        if not self._interactive:
            # Non-interactive mode: print full entry details
            self._print_entry(self.history[-1])

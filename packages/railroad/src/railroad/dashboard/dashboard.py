from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.rule import Rule
from rich.text import Text

from math import isinf
import re
from time import sleep, perf_counter
from typing import Any, List, Set, Union, Tuple, Callable

from railroad._bindings import (
    Action,
    Goal,
    GoalType,
    Fluent,
    LiteralGoal,
    State,
)

from ._protocols import DashboardEnvironment, DashboardPlanner
from ._goals import format_goal, get_satisfied_branch, get_best_branch
from ._tui import (
    _is_headless_environment,
    split_markdown_flat,
    action_color,
    render_timeline,
)
from .plotting import _PlottingMixin


class PlannerDashboard(_PlottingMixin):
    """
    Rich-based dashboard for the robot task planner.

    Layout (interactive mode):
      +------------------------------------------+
      |  progress bar (goals achieved / total)   |
      +------------------------------------------+
      |  Left: fluents & goals   | Right: trace  |
      +--------------------------+---------------+

    In non-interactive mode (CI, Claude Code, Colab, piped output),
    falls back to simple text progress updates.
    """

    def __init__(
        self,
        goal: Union[Goal, Fluent],
        env: DashboardEnvironment,
        *,
        fluent_filter: Callable[[Fluent], bool] | None = None,
        planner_factory: Callable[[list[Action]], DashboardPlanner] | None = None,
        print_on_exit: bool = True,
        console: Console | None = None,
        force_interactive: bool | None = None,
    ):
        """Initialize the planner dashboard.

        Args:
            goal: A Goal object (AndGoal, OrGoal, LiteralGoal, etc.),
                  or a bare Fluent (which will be wrapped in LiteralGoal)
            env: Environment providing state and actions
            fluent_filter: Optional filter for selecting relevant fluents to display
            planner_factory: Factory to create a planner from actions (default: MCTSPlanner)
            print_on_exit: Whether to automatically call print_history() on __exit__
            console: Optional Rich console for output
            force_interactive: Override interactive detection (True=live dashboard,
                               False=simple output, None=auto-detect)
        """
        if console:
            self.console = console
        else:
            self.console = Console(record=True)

        self._env = env
        self._fluent_filter = fluent_filter
        self._print_on_exit = print_on_exit
        self._step_index = 0

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
        self._start_time = perf_counter()

        # Compute initial heuristic
        from railroad.planner import MCTSPlanner
        factory = planner_factory or MCTSPlanner
        planner = factory(env.get_actions())
        self.initial_heuristic = planner.heuristic(env.state, self.goal)
        if isinf(self.initial_heuristic):
            raise ValueError(
                "Initial heuristic is inf; provide a reachable goal or a planner heuristic with a finite fallback."
            )

        # Actions stored as (action_name, start_time) tuples
        self.actions_taken: List[Tuple[str, float]] = []
        self._last_state_time: float = 0.0  # Track previous state time for action start

        # Track known robots from (free NAME) fluents
        self.known_robots: Set[str] = set()

        self.history: list[dict] = []

        # Trajectory tracking: entity_name -> [(time, location_name, (x,y) or None), ...]
        self._entity_positions: dict[str, list[tuple[float, str, tuple[float, float] | None]]] = {}
        self._goal_time: float | None = None
        self._nav_grid_snapshots: list[tuple[float, Any]] = []
        self._nav_continuous_positions: dict[str, list[tuple[float, float, float]]] = {}

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

    def _compute_relevant_fluents(self, state: State) -> set[Fluent]:
        if self._fluent_filter is not None:
            return {f for f in state.fluents if self._fluent_filter(f)}
        return set(state.fluents)

    def _get_location_coords(self) -> dict[str, tuple[float, float]]:
        """Get location coordinates from the environment.

        Combines coordinates from all available sources:
        1. ProcTHOR scene: env.scene.locations (static map locations)
        2. LocationRegistry: env.location_registry (dynamically discovered)
        Registry entries take priority for overlapping names.
        """
        env = self._env
        coords: dict[str, tuple[float, float]] = {}

        # ProcTHOR scene locations
        scene = getattr(env, "scene", None)
        if scene is not None and hasattr(scene, "locations"):
            for name, xy in scene.locations.items():
                coords[name] = (float(xy[0]), float(xy[1]))

        # LocationRegistry (may add dynamically discovered locations)
        registry = getattr(env, "location_registry", None)
        if registry is not None:
            all_locs: set[str] = set()
            for positions in self._entity_positions.values():
                for _, loc_name, _ in positions:
                    all_locs.add(loc_name)
            for loc in all_locs:
                c = registry.get(loc)
                if c is not None:
                    coords[loc] = (float(c[0]), float(c[1]))

        return coords

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
        # Auto-call initial update
        self._do_update(
            state=self._env.state,
            relevant_fluents=self._compute_relevant_fluents(self._env.state),
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)
            self._live = None
        self.finalize_trajectories()
        if self._print_on_exit:
            self.print_history()

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

    def _colorize_goal_line(self, line: str, fluents, *, rich: bool = True) -> str:
        """Annotate a goal line based on whether literals are satisfied.

        Inserts checkmarks for satisfied goals. When rich=True, also wraps
        in Rich color markup (green/red).
        """
        # Build a set of fluent strings for faster lookup
        fluent_strs = {str(f) for f in fluents}

        def colorize_match(match):
            fluent_str = match.group(0)

            # Check if this is a negated fluent like "(not at Book table)"
            if fluent_str.startswith("(not "):
                positive_fluent_str = "(" + fluent_str[5:]
                satisfied = positive_fluent_str not in fluent_strs
            else:
                satisfied = fluent_str in fluent_strs

            mark = "\u2713" if satisfied else " "
            text = f"{mark}{fluent_str}"
            if rich:
                color = "green" if satisfied else "red"
                return f"[{color}]{text}[/{color}]"
            return text

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
            extra=f"h={h_now:.2f} \u0394h={improvement:.2f}",
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
        entry = {
            "step": step_index,
            "time": float(state.time),
            "last_action": last_action_name,
            "heuristic": float(heuristic_value) if heuristic_value is not None else None,
            "relevant_fluents": [str(f) for f in sorted(relevant_fluents, key=lambda x: x.name)],
            "fluents": set(state.fluents),
            "goals": {
                str(g): LiteralGoal(g).evaluate(state.fluents) for g in self.goal_fluents
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

        return "\n".join(lines)

    def _print_entry(self, entry: dict) -> None:
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

        # Goal with colorized satisfaction status
        self.console.print("[bold red]Goal:[/]")
        fluents = entry["fluents"]
        for line in format_goal(self.goal, compact=True).split('\n'):
            self.console.print(f"  {self._colorize_goal_line(line, fluents)}")

        goal_satisfied = entry.get("goal_satisfied", False)
        status = "[green]\u2713 SATISFIED[/]" if goal_satisfied else "[red]\u2717 NOT SATISFIED[/]"
        self.console.print(f"[bold red]Overall Goal:[/] {status}")

    def format_history_as_text(self) -> str:
        """Return the full dashboard history as a multi-line string."""
        lines: list[str] = []
        for entry in self.history:
            lines.append(self._format_single_entry(entry))
            fluents = entry["fluents"]
            lines.append("Goal:")
            for goal_line in format_goal(self.goal, compact=True).split('\n'):
                lines.append(f"    {self._colorize_goal_line(goal_line, fluents, rich=False)}")
            goal_status = entry.get("goal_satisfied", False)
            status_mark = "\u2713 SATISFIED" if goal_status else "\u2717 NOT SATISFIED"
            lines.append(f"Overall Goal: {status_mark}")
            lines.append("\n")  # blank line between steps

        return "\n".join(lines)

    def print_history(self):
        """Pretty-print the full history using Rich.

        In non-interactive mode, skips the step-by-step history (already printed
        during update calls) and only prints the final summary.
        """
        local_console = self.console
        final_state = self._env.state
        actions_taken = [name for name, _ in self.actions_taken]

        if not self.history:
            local_console.print("[yellow]No history recorded.[/yellow]")
            return

        # In interactive mode, print full history
        # In non-interactive mode, history was already printed during updates
        if self._interactive:
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

    def finalize_trajectories(self) -> None:
        """Record pending robot positions from upcoming effects without mutating the environment.

        When the planning loop ends (goal reached), some robots may still be
        mid-action with pending effects. This method scans upcoming effects for
        future (at entity location) fluents and records them so that
        plot_trajectories() can show complete paths.

        Sets _goal_time so that plotted trajectories are truncated at goal time.
        """
        state = self._env.state
        self._goal_time = state.time

        coord_lookup = self._get_location_coords()
        for effect_time, effect in state.upcoming_effects:
            for fluent in effect.resulting_fluents:
                if fluent.name == "at" and len(fluent.args) >= 2:
                    entity_name = fluent.args[0]
                    location_name = fluent.args[1]
                    coords = coord_lookup.get(location_name)
                    if entity_name not in self._entity_positions:
                        self._entity_positions[entity_name] = []
                    positions = self._entity_positions[entity_name]
                    if not positions or positions[-1][1] != location_name or positions[-1][0] != effect_time:
                        positions.append((effect_time, location_name, coords))

    def update(
        self,
        mcts: DashboardPlanner,
        action_name: str,
    ):
        """Update the dashboard after an action has been executed.

        Args:
            mcts: The planner used for this step (provides heuristic and trace)
            action_name: The name of the action that was just executed
        """
        state = self._env.state
        relevant_fluents = self._compute_relevant_fluents(state)
        tree_trace = mcts.get_trace_from_last_mcts_tree()
        heuristic_value = mcts.heuristic(state, self.goal)
        step_index = self._step_index
        self._step_index += 1

        self._do_update(
            state=state,
            relevant_fluents=relevant_fluents,
            tree_trace=tree_trace,
            step_index=step_index,
            last_action_name=action_name,
            heuristic_value=heuristic_value,
        )

    def _record_entity_positions(self, state: State) -> None:
        """Extract and record entity positions from (at entity location) fluents."""
        coord_lookup = self._get_location_coords()
        for fluent in state.fluents:
            if fluent.name == "at" and len(fluent.args) >= 2:
                entity_name = fluent.args[0]
                location_name = fluent.args[1]
                coords = coord_lookup.get(location_name)
                if entity_name not in self._entity_positions:
                    self._entity_positions[entity_name] = []
                positions = self._entity_positions[entity_name]
                # Record when location changes OR when time advances at the
                # same location (e.g. pick/place actions).  The duplicate
                # waypoint creates a "hold" segment so interpolation shows
                # the entity as stationary during non-movement actions.
                if not positions or positions[-1][1] != location_name or positions[-1][0] != state.time:
                    positions.append((state.time, location_name, coords))

    def make_act_callback(self) -> Callable[[], None]:
        """Create a callback for ``env.act()`` that captures intermediate state.

        For navigation environments, this records robot poses (for smooth,
        obstacle-respecting trajectories) and observed-grid snapshots (for
        animated grid evolution in videos) at every act-loop time step.

        Returns a reusable callback — create it once and pass it to every
        ``env.act(action, loop_callback_fn=callback)`` call.
        """
        last_snapshot_time: list[float] = [float('-inf')]
        min_snapshot_interval = 0.5  # seconds of sim-time between grid copies

        def _callback() -> None:
            env = self._env
            t = env.time  # type: ignore[union-attr]

            # Capture robot poses (cheap — just a few tuples per call)
            robot_poses = getattr(env, 'robot_poses', None)
            if robot_poses is not None:
                for robot, pose in robot_poses.items():
                    if robot not in self._nav_continuous_positions:
                        self._nav_continuous_positions[robot] = []
                    self._nav_continuous_positions[robot].append(
                        (t, pose.x, pose.y)
                    )

            # Capture grid snapshot (throttled to limit memory)
            observed_grid = getattr(env, 'observed_grid', None)
            if observed_grid is not None and (t - last_snapshot_time[0]) >= min_snapshot_interval:
                self._nav_grid_snapshots.append((t, observed_grid.copy()))
                last_snapshot_time[0] = t

        return _callback

    def _do_update(
        self,
        state: State,
        relevant_fluents: Set = set(),
        tree_trace: str | None = None,
        step_index: int | None = None,
        last_action_name: str | None = None,
        heuristic_value: float | None = None,
    ):
        """Internal update implementation shared by update() and __enter__."""
        # Extract robots from (free NAME) fluents
        for fluent in state.fluents:
            fluent_str = str(fluent)
            if fluent_str.startswith("(free ") and fluent_str.endswith(")"):
                robot_name = fluent_str[6:-1].strip()
                self.known_robots.add(robot_name)

        self._record_entity_positions(state)

        # Capture navigation state (robot poses + grid) for plotting.
        # The act callback captures intermediate states; this captures
        # the initial state (from __enter__) and post-action states.
        robot_poses = getattr(self._env, 'robot_poses', None)
        if robot_poses is not None:
            for robot, pose in robot_poses.items():
                if robot not in self._nav_continuous_positions:
                    self._nav_continuous_positions[robot] = []
                self._nav_continuous_positions[robot].append(
                    (state.time, pose.x, pose.y)
                )

        observed_grid = getattr(self._env, 'observed_grid', None)
        if observed_grid is not None:
            self._nav_grid_snapshots.append((state.time, observed_grid.copy()))

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

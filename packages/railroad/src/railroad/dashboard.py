from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.rule import Rule
from rich.text import Text

import math
import os
import re
from time import sleep, perf_counter
from typing import Any, Collection, List, Dict, Set, Union, Tuple, Optional, Callable, runtime_checkable, Protocol

from railroad._bindings import (
    Action,
    Goal,
    GroundedEffect,
    LiteralGoal,
    AndGoal,
    GoalType,
    Fluent,
    State,
)


@runtime_checkable
class DashboardEnvironment(Protocol):
    @property
    def state(self) -> State: ...
    def get_actions(self) -> list[Action]: ...


class DashboardPlanner(Protocol):
    def heuristic(self, state: State, goal: Union[Goal, Fluent]) -> float: ...
    def get_trace_from_last_mcts_tree(self) -> str: ...


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


# =============================================================================
# Goal formatting and analysis functions
# =============================================================================


def format_goal(goal: Goal, indent: int = 0, compact: bool = False) -> str:
    """Format a goal as a readable string.

    Args:
        goal: A Goal object (LiteralGoal, AndGoal, OrGoal, etc.)
        indent: Current indentation level (for nested goals)
        compact: If True, use single-line format for simple goals

    Returns:
        A formatted string representation of the goal.

    Example output:
        AND(
          (at r1 kitchen)
          OR(
            (holding r1 cup)
            (holding r1 plate)
          )
        )
    """
    goal_type = goal.get_type()
    prefix = "  " * indent

    if goal_type == GoalType.LITERAL:
        assert isinstance(goal, LiteralGoal)
        return f"{prefix}{goal.fluent()}"

    elif goal_type == GoalType.TRUE_GOAL:
        return f"{prefix}TRUE"

    elif goal_type == GoalType.FALSE_GOAL:
        return f"{prefix}FALSE"

    elif goal_type == GoalType.AND:
        children = list(goal.children())
        if compact and len(children) <= 2 and all(
            c.get_type() == GoalType.LITERAL for c in children
        ):
            child_strs = [str(c.fluent()) for c in children if isinstance(c, LiteralGoal)]
            return f"{prefix}AND({', '.join(child_strs)})"
        else:
            lines = [f"{prefix}AND("]
            for child in children:
                lines.append(format_goal(child, indent + 1, compact))
            lines.append(f"{prefix})")
            return "\n".join(lines)

    elif goal_type == GoalType.OR:
        children = list(goal.children())
        if compact and len(children) <= 2 and all(
            c.get_type() == GoalType.LITERAL for c in children
        ):
            child_strs = [str(c.fluent()) for c in children if isinstance(c, LiteralGoal)]
            return f"{prefix}OR({', '.join(child_strs)})"
        else:
            lines = [f"{prefix}OR("]
            for child in children:
                lines.append(format_goal(child, indent + 1, compact))
            lines.append(f"{prefix})")
            return "\n".join(lines)

    else:
        return f"{prefix}<unknown goal type>"


def get_satisfied_branch(goal: Goal, fluents: Set[Fluent]) -> Union[Goal, None]:
    """Find the minimal satisfied branch of a goal.

    For OR goals, returns the first satisfied child.
    For AND goals, returns an AND of all satisfied children's branches.
    For literals, returns the literal if satisfied, None otherwise.

    Args:
        goal: A Goal object
        fluents: Set of fluents representing current state

    Returns:
        A Goal representing the satisfied portion, or None if not satisfied.
    """
    if not goal.evaluate(fluents):
        return None

    goal_type = goal.get_type()

    if goal_type == GoalType.LITERAL:
        return goal

    elif goal_type == GoalType.TRUE_GOAL:
        return goal

    elif goal_type == GoalType.FALSE_GOAL:
        return None

    elif goal_type == GoalType.OR:
        # Return the first satisfied branch
        for child in goal.children():
            if child.evaluate(fluents):
                return get_satisfied_branch(child, fluents)
        return None

    elif goal_type == GoalType.AND:
        # Return AND of all satisfied branches
        satisfied_children = []
        for child in goal.children():
            branch = get_satisfied_branch(child, fluents)
            if branch is not None:
                satisfied_children.append(branch)
        if len(satisfied_children) == 1:
            return satisfied_children[0]
        elif len(satisfied_children) > 1:
            return AndGoal(satisfied_children)
        return None

    return None


def get_best_branch(goal: Goal, fluents: Set[Fluent]) -> Goal:
    """Find the most promising branch of a goal (highest completion ratio).

    For OR goals, returns the child with highest completion ratio.
    For AND goals, returns an AND of the best branches from each child.
    For literals, returns the literal itself.

    Args:
        goal: A Goal object
        fluents: Set of fluents representing current state

    Returns:
        A Goal representing the best branch to pursue.
    """
    goal_type = goal.get_type()

    if goal_type == GoalType.LITERAL:
        return goal

    elif goal_type in (GoalType.TRUE_GOAL, GoalType.FALSE_GOAL):
        return goal

    elif goal_type == GoalType.OR:
        # Find child with best completion ratio
        best_child = None
        best_ratio = -1.0

        for child in goal.children():
            child_literals = child.get_all_literals()
            total = len(child_literals)
            achieved = child.goal_count(fluents)
            ratio = achieved / total if total > 0 else 0.0

            if ratio > best_ratio:
                best_ratio = ratio
                best_child = child

        if best_child is not None:
            return get_best_branch(best_child, fluents)
        return goal

    elif goal_type == GoalType.AND:
        # Return AND of best branches from each child
        best_children = []
        for child in goal.children():
            best_children.append(get_best_branch(child, fluents))

        if len(best_children) == 1:
            return best_children[0]
        elif len(best_children) > 1:
            return AndGoal(best_children)
        return goal

    return goal


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
    def pos(t):
        return int((t - min_t) / (max_t - min_t) * (width * 2 - 1))

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


def _generate_coordinates(location_names: Collection[str]) -> dict[str, tuple[float, float]]:
    """Generate circular layout coordinates for locations lacking real coordinates.

    Places locations evenly around a unit circle. This is a simple placeholder
    layout — callers with real coordinates should provide them instead.
    """
    names = sorted(location_names)
    n = len(names)
    if n == 0:
        return {}
    if n == 1:
        return {names[0]: (0.0, 0.0)}
    coords: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(names):
        angle = 2 * math.pi * i / n
        coords[name] = (math.cos(angle), math.sin(angle))
    return coords



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

        # Actions stored as (action_name, start_time) tuples
        self.actions_taken: List[Tuple[str, float]] = []
        self._last_state_time: float = 0.0  # Track previous state time for action start

        # Track known robots from (free NAME) fluents
        self.known_robots: Set[str] = set()

        self.history: list[dict] = []

        # Trajectory tracking: entity_name -> [(time, location_name, (x,y) or None), ...]
        self._entity_positions: dict[str, list[tuple[float, str, tuple[float, float] | None]]] = {}
        self._goal_time: float | None = None

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

    def _get_location_coords(self) -> tuple[
        dict[str, tuple[float, float]],  # location_name -> (x, y)
        Any,                              # occupancy grid or None
        Any,                              # scene graph or None
    ]:
        """Get location coordinates from the environment.

        Three-tier detection:
        1. ProcTHOR: env.scene with .locations, .grid, .scene_graph
        2. LocationRegistry: env.location_registry with .get()
        3. Pure symbolic: empty dict (coordinates deferred to plot time)
        """
        env = self._env
        coords: dict[str, tuple[float, float]] = {}
        grid = None
        graph = None

        # Tier 1: ProcTHOR environment
        scene = getattr(env, "scene", None)
        if scene is not None and hasattr(scene, "locations"):
            for name, xy in scene.locations.items():
                coords[name] = (float(xy[0]), float(xy[1]))
            grid = getattr(scene, "grid", None)
            graph = getattr(scene, "scene_graph", None)
            return coords, grid, graph

        # Tier 2: LocationRegistry
        registry = getattr(env, "location_registry", None)
        if registry is not None:
            # Collect all location names from current entity positions
            all_locs: set[str] = set()
            for positions in self._entity_positions.values():
                for _, loc_name, _ in positions:
                    all_locs.add(loc_name)
            for loc in all_locs:
                c = registry.get(loc)
                if c is not None:
                    coords[loc] = (float(c[0]), float(c[1]))
            return coords, None, None

        # Tier 3: Pure symbolic — no coordinates available
        return coords, None, None

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

        coord_lookup, _, _ = self._get_location_coords()
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

    def _build_entity_trajectories(
        self,
        *,
        location_coords: dict[str, tuple[float, float]] | None = None,
        include_objects: bool = False,
    ) -> tuple[
        dict[str, tuple[list[tuple[float, float]], list[float]]],  # entity -> (waypoints, times)
        dict[str, tuple[float, float]],  # resolved env_coords
        Any,  # grid (for plot_grid background)
    ]:
        """Build resolved and expanded entity trajectories.

        Handles: environment coordinate lookup, ``location_coords`` conflict
        check, missing-coordinate generation, robot/object separation, and
        scene ``get_trajectory`` expansion.

        Args:
            location_coords: Optional explicit location->(x,y) mapping.
                Raises ValueError if any stored positions already have
                coordinates from the environment.
            include_objects: If True, also include non-robot entities.

        Returns:
            (trajectories, env_coords, grid) where *trajectories* maps entity
            names to (waypoints, times) lists with scene-expanded paths.
        """
        # Get environment coordinates + grid
        env_coords, grid, _graph = self._get_location_coords()

        # Handle explicit location_coords
        if location_coords is not None:
            for entity, positions in self._entity_positions.items():
                for _, loc_name, stored_coords in positions:
                    if stored_coords is not None:
                        raise ValueError(
                            f"Cannot pass location_coords when positions already "
                            f"have coordinates from the environment "
                            f"(entity={entity!r}, location={loc_name!r}). "
                            f"Use location_coords only when the environment does "
                            f"not provide coordinates."
                        )
            env_coords.update(location_coords)

        # Generate coordinates for locations that still lack them
        all_location_names: set[str] = set()
        for positions in self._entity_positions.values():
            for _, loc_name, _ in positions:
                all_location_names.add(loc_name)
        missing = all_location_names - set(env_coords.keys())
        if missing:
            env_coords.update(_generate_coordinates(missing))

        # Separate entities into robots and objects
        robot_entities: dict[str, list[tuple[float, str, tuple[float, float] | None]]] = {}
        object_entities: dict[str, list[tuple[float, str, tuple[float, float] | None]]] = {}
        for entity, positions in self._entity_positions.items():
            if entity in self.known_robots:
                robot_entities[entity] = positions
            else:
                object_entities[entity] = positions

        # Scene-aware trajectory expansion
        scene = getattr(self._env, "scene", None)
        get_trajectory_fn = getattr(scene, "get_trajectory", None)

        trajectories: dict[str, tuple[list[tuple[float, float]], list[float]]] = {}

        # Build robot trajectories (with scene expansion)
        for entity, positions in sorted(robot_entities.items()):
            waypoints: list[tuple[float, float]] = []
            times: list[float] = []
            for t, loc_name, stored_coords in positions:
                if stored_coords is not None:
                    waypoints.append(stored_coords)
                    times.append(t)
                elif loc_name in env_coords:
                    waypoints.append(env_coords[loc_name])
                    times.append(t)

            if len(waypoints) < 2:
                trajectories[entity] = (waypoints, times)
                continue

            # Expand through scene trajectory if available
            if get_trajectory_fn is not None:
                import numpy as np
                expanded_wps: list[tuple[float, float]] = []
                expanded_ts: list[float] = []
                for seg_i in range(len(waypoints) - 1):
                    path = get_trajectory_fn([waypoints[seg_i], waypoints[seg_i + 1]])
                    if not path or len(path) < 2:
                        path = [waypoints[seg_i], waypoints[seg_i + 1]]
                    pts = np.array(path)
                    cum_dist = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
                    t0, t1 = times[seg_i], times[seg_i + 1]
                    seg_times = (t0 + (t1 - t0) * cum_dist / max(cum_dist[-1], 1e-9)).tolist()
                    start = 1 if expanded_wps else 0
                    expanded_wps.extend(path[start:])
                    expanded_ts.extend(seg_times[start:])
                waypoints = expanded_wps
                times = expanded_ts

            trajectories[entity] = (waypoints, times)

        # Build object trajectories (no scene expansion)
        if include_objects:
            for entity, positions in sorted(object_entities.items()):
                waypoints = []
                times = []
                for t, loc_name, stored_coords in positions:
                    if stored_coords is not None:
                        waypoints.append(stored_coords)
                        times.append(t)
                    elif loc_name in env_coords:
                        waypoints.append(env_coords[loc_name])
                        times.append(t)
                trajectories[entity] = (waypoints, times)

        return trajectories, env_coords, grid

    def plot_trajectories(
        self,
        ax: Any = None,
        *,
        show_objects: bool = False,
        location_coords: dict[str, tuple[float, float]] | None = None,
    ) -> Any:
        """Plot entity trajectories collected during planning.

        Works across all environment types:
        - ProcTHOR: uses occupancy grid background and obstacle-respecting paths
        - LocationRegistry: uses Euclidean coordinates with gradient-colored lines
        - Pure symbolic: auto-generates circular layout coordinates

        Args:
            ax: Matplotlib axes. If None, a new figure/axes is created.
            show_objects: If True, also plot object trajectories as dashed lines.
            location_coords: Optional explicit location->(x,y) mapping used to
                resolve positions that lack coordinates (stored as None). If
                provided, any positions that already have non-None coordinates
                from the environment will raise a ValueError to prevent
                conflicting coordinate sources.

        Returns:
            The matplotlib axes with the plotted trajectories.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Trajectory plotting requires matplotlib: pip install matplotlib"
            )

        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Build trajectories using shared helper
        trajectories, env_coords, grid = self._build_entity_trajectories(
            location_coords=location_coords,
            include_objects=show_objects,
        )

        # Plot grid background if available (ProcTHOR tier)
        if grid is not None:
            try:
                from railroad.environment.procthor.plotting import plot_grid
                plot_grid(ax, grid)
            except ImportError:
                pass

        # Colormaps for distinguishing robots
        colormaps = [
            "viridis", "plasma", "inferno", "magma", "cividis",
            "spring", "summer", "autumn", "winter", "cool",
        ]

        # Dense interpolation for smooth scatter trails
        import numpy as np
        t_end = self._goal_time or 0.0
        for _entity, (_wps, traj_times) in trajectories.items():
            if traj_times:
                t_end = max(t_end, max(traj_times))
        dense_times = np.linspace(0.0, t_end, 2000) if t_end > 0 else np.array([0.0])
        dense_positions = self.get_entity_positions_at_times(
            dense_times, location_coords=location_coords,
        )

        # Plot robot trajectories as interpolated scatter colored by time
        robot_id = 0
        for entity in sorted(self.known_robots & set(trajectories.keys())):
            waypoints, times = trajectories[entity]
            if len(waypoints) < 2:
                continue

            cmap_name = colormaps[robot_id % len(colormaps)]
            robot_id += 1

            # Get original positions for location markers/labels
            positions = self._entity_positions[entity]
            location_waypoints: list[tuple[float, float]] = []
            for _, loc_name, stored_coords in positions:
                if stored_coords is not None:
                    location_waypoints.append(stored_coords)
                elif loc_name in env_coords:
                    location_waypoints.append(env_coords[loc_name])

            if entity in dense_positions:
                pts = dense_positions[entity]
                ax.scatter(pts[:, 0], pts[:, 1], c=dense_times, cmap=cmap_name, s=4, zorder=5, alpha=0.7)

            # Plot location markers and labels
            wx = [p[0] for p in location_waypoints]
            wy = [p[1] for p in location_waypoints]
            ax.scatter(wx, wy, s=20, zorder=6, color="black")

            ax.annotate(
                entity, (wx[0], wy[0]),
                fontsize=7, fontweight="bold", color="brown",
            )

            for i, (_, loc_name, _) in enumerate(positions):
                if i < len(location_waypoints):
                    ax.annotate(
                        loc_name, location_waypoints[i],
                        fontsize=5, color="brown",
                        xytext=(3, 3), textcoords="offset points",
                    )

        # Optionally plot object trajectories
        if show_objects:
            object_entities = {e: trajectories[e] for e in trajectories if e not in self.known_robots}
            for entity in sorted(object_entities):
                waypoints, _times = object_entities[entity]
                if len(waypoints) < 2:
                    continue

                x = [p[0] for p in waypoints]
                y = [p[1] for p in waypoints]
                ax.plot(x, y, linestyle="--", linewidth=1, alpha=0.6, label=entity)
                ax.scatter(x, y, s=10, zorder=5, alpha=0.6)

            if object_entities:
                ax.legend(fontsize=6)

        # Auto-scale for non-grid plots
        if grid is None:
            ax.autoscale()
            ax.set_aspect("equal", adjustable="datalim")

        ax.set_title("Entity Trajectories")
        return ax

    def get_entity_positions_at_times(
        self,
        times: Any,
        *,
        location_coords: dict[str, tuple[float, float]] | None = None,
        include_objects: bool = False,
    ) -> dict[str, Any]:
        """Interpolate entity positions at arbitrary query times.

        Uses numpy.interp on x and y independently. Clamps to first position
        before trajectory start and last position after trajectory end.

        Args:
            times: Query times as a numpy array or list of floats.
            location_coords: Optional explicit location->(x,y) mapping.
            include_objects: If True, include non-robot entities.

        Returns:
            ``{entity_name: (N, 2) ndarray}`` of interpolated positions.
        """
        import numpy as np

        query_times = np.asarray(times, dtype=float)
        trajectories, _env_coords, _grid = self._build_entity_trajectories(
            location_coords=location_coords,
            include_objects=include_objects,
        )

        result: dict[str, Any] = {}
        for entity, (waypoints, traj_times) in trajectories.items():
            if len(waypoints) < 2:
                continue
            wp_arr = np.array(waypoints)
            t_arr = np.array(traj_times)
            x_interp = np.interp(query_times, t_arr, wp_arr[:, 0])
            y_interp = np.interp(query_times, t_arr, wp_arr[:, 1])
            result[entity] = np.column_stack([x_interp, y_interp])

        return result

    def save_trajectory_video(
        self,
        path: str,
        *,
        location_coords: dict[str, tuple[float, float]] | None = None,
        fps: int = 60,
        duration: float = 10.0,
        figsize: tuple[float, float] = (12.8, 7.2),
    ) -> None:
        """Save an animated trajectory video/GIF.

        Args:
            path: Output file path. Extension determines writer:
                ``.gif`` uses PillowWriter, ``.mp4``/``.avi`` uses FFMpegWriter.
            location_coords: Optional explicit location->(x,y) mapping.
            fps: Frames per second.
            duration: Total animation duration in seconds.
            figsize: Figure size in inches.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        trajectories, env_coords, grid = self._build_entity_trajectories(
            location_coords=location_coords,
            include_objects=False,
        )

        # Determine time range
        t_end = self._goal_time or 0.0
        for _entity, (_wps, traj_times) in trajectories.items():
            if traj_times:
                t_end = max(t_end, max(traj_times))
        if t_end <= 0.0:
            return

        n_frames = int(fps * duration)
        frame_times = np.linspace(0.0, t_end, n_frames)

        # Pre-compute dense trail positions (same resolution as static plot)
        dense_trail_times = np.linspace(0.0, t_end, 2000)
        dense_positions = self.get_entity_positions_at_times(
            dense_trail_times, location_coords=location_coords,
        )
        if not dense_positions:
            return

        # Pre-compute low-res positions for the moving marker
        marker_positions = self.get_entity_positions_at_times(
            frame_times, location_coords=location_coords,
        )
        entity_names = sorted(dense_positions.keys())

        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import Normalize

        # Check if ProcTHOR scene provides a top-down image
        scene = getattr(self._env, "scene", None)
        get_top_down = getattr(scene, "get_top_down_image", None)
        has_overhead = get_top_down is not None

        fig = plt.figure(figsize=figsize)
        if has_overhead:
            gs = GridSpec(1, 3, width_ratios=[1, 2, 1], figure=fig)
            overhead_ax = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot(gs[0, 1])
            sidebar_ax = fig.add_subplot(gs[0, 2])
            # Render the overhead map
            assert get_top_down is not None
            top_down_image = get_top_down(orthographic=True)
            overhead_ax.imshow(top_down_image)
            overhead_ax.axis("off")
            overhead_ax.set_title("Top-down View", fontsize=8)
        else:
            gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
            ax = fig.add_subplot(gs[0, 0])
            sidebar_ax = fig.add_subplot(gs[0, 1])
        sidebar_ax.set_axis_off()

        # Static background: grid
        if grid is not None:
            try:
                from railroad.environment.procthor.plotting import plot_grid
                plot_grid(ax, grid)
            except ImportError:
                pass

        # Static background: location markers + labels
        plotted_locs: set[str] = set()
        for positions in self._entity_positions.values():
            for _, loc_name, stored_coords in positions:
                if loc_name in plotted_locs:
                    continue
                plotted_locs.add(loc_name)
                coord = stored_coords if stored_coords is not None else env_coords.get(loc_name)
                if coord is not None:
                    ax.plot(coord[0], coord[1], "ks", markersize=4, zorder=4)
                    ax.annotate(
                        loc_name, coord,
                        fontsize=5, color="brown",
                        xytext=(3, 3), textcoords="offset points",
                    )

        colormaps = [
            "viridis", "plasma", "inferno", "magma", "cividis",
            "spring", "summer", "autumn", "winter", "cool",
        ]

        # Distinct marker colors per entity
        marker_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange",
                         "tab:purple", "tab:cyan", "tab:pink", "tab:olive"]

        # --- Sidebar layout: compute goal section height first, then colorbars fill above ---
        n_entities = len(entity_names)
        cbar_width = 0.06  # width of each colorbar strip in axes fraction
        cbar_gap = 0.03    # gap between strips
        cbar_left = 0.05   # left margin

        # Goal section: measure how tall it needs to be
        goal_str = format_goal(self.goal, compact=False)
        goal_lines = goal_str.split("\n")
        goal_line_spacing = 0.035
        goal_section_height = (len(goal_lines) + 1) * goal_line_spacing + 0.04  # +1 for header
        goal_section_bottom = 0.03
        goal_section_top = goal_section_bottom + goal_section_height

        # Colorbars fill the space above the goal section
        cbar_bottom = goal_section_top + 0.04
        cbar_top = 0.88  # leave room for marker + label at top
        cbar_height = cbar_top - cbar_bottom

        # Total width used by colorbar strips
        cbar_total_width = n_entities * cbar_width + (n_entities - 1) * cbar_gap
        actions_left = cbar_left + cbar_total_width + 0.08  # action list starts here

        for idx, entity in enumerate(entity_names):
            cmap_name = colormaps[idx % len(colormaps)]
            mcolor = marker_colors[idx % len(marker_colors)]
            x0 = cbar_left + idx * (cbar_width + cbar_gap)

            # Create an inset axes for each colorbar strip
            cbar_ax = sidebar_ax.inset_axes((x0, cbar_bottom, cbar_width, cbar_height))
            gradient = np.linspace(0, 1, 256).reshape(-1, 1)
            cbar_ax.imshow(gradient, aspect="auto", cmap=cmap_name, origin="lower",
                           extent=(0, 1, 0, t_end))
            cbar_ax.set_xlim(0, 1)
            cbar_ax.set_ylim(t_end, 0)  # invert: t=0 at top, t_end at bottom
            cbar_ax.set_xticks([])
            if idx == 0:
                cbar_ax.set_ylabel("time", fontsize=6)
                cbar_ax.tick_params(axis="y", labelsize=5)
            else:
                cbar_ax.set_yticks([])

            # Robot marker at top of the strip
            sidebar_ax.plot(
                x0 + cbar_width / 2, cbar_top + 0.04, "o",
                color=mcolor, markeredgecolor="black", markeredgewidth=0.8,
                markersize=8, transform=sidebar_ax.transAxes, clip_on=False,
            )
            sidebar_ax.text(
                x0 + cbar_width / 2, cbar_top + 0.08, entity,
                fontsize=5, fontfamily="monospace", fontweight="bold",
                ha="center", va="bottom",
                transform=sidebar_ax.transAxes,
            )

        # --- Sidebar: goal progress section ---
        # Pre-compute goal snapshots: sorted list of (time, {literal_str: bool})
        goal_snapshots: list[tuple[float, dict[str, bool]]] = []
        for entry in self.history:
            goal_snapshots.append((entry["time"], entry["goals"]))

        # Render goal lines as text artists
        sidebar_ax.text(
            cbar_left, goal_section_top - 0.01, "Goal:",
            fontsize=7, fontfamily="monospace", fontweight="bold",
            ha="left", va="top", transform=sidebar_ax.transAxes,
        )
        goal_text_artists: list[tuple[Any, str | None]] = []
        for i, line in enumerate(goal_lines):
            y_pos = goal_section_top - 0.01 - (i + 1) * goal_line_spacing
            # Detect if this line contains a literal: matches (name args...)
            stripped = line.strip()
            literal_str: str | None = None
            if stripped.startswith("(") and stripped.endswith(")") and "(" not in stripped[1:]:
                literal_str = stripped
            txt = sidebar_ax.text(
                cbar_left, y_pos, line,
                fontsize=6, fontfamily="monospace",
                color="red" if literal_str else "gray",
                ha="left", va="top", transform=sidebar_ax.transAxes,
            )
            goal_text_artists.append((txt, literal_str))

        # --- Sidebar: action list (text artists, initially hidden) ---
        # Position actions proportionally by start time, with a minimum
        # gap so labels don't overlap.
        actions = self.actions_taken  # List[(action_name, start_time)]
        n_actions = len(actions)
        action_y_top = cbar_top
        action_y_bottom = cbar_bottom
        min_gap = 0.025  # minimum vertical spacing in axes fraction

        # Compute ideal y positions proportional to time (top = t=0)
        action_y_positions: list[float] = []
        if n_actions > 0:
            y_range_avail = action_y_top - action_y_bottom
            for _act_name, act_time in actions:
                frac = act_time / t_end if t_end > 0 else 0.0
                action_y_positions.append(action_y_top - frac * y_range_avail)
            # Enforce minimum gap: walk top-to-bottom pushing overlaps down
            for i in range(1, n_actions):
                if action_y_positions[i] > action_y_positions[i - 1] - min_gap:
                    action_y_positions[i] = action_y_positions[i - 1] - min_gap

        action_texts = []
        for i, (act_name, act_time) in enumerate(actions):
            y_pos = action_y_positions[i]
            txt = sidebar_ax.text(
                actions_left, y_pos, f"{i+1}. {act_name}",
                fontsize=5, fontfamily="monospace",
                ha="left", va="top",
                transform=sidebar_ax.transAxes,
                alpha=0.0,  # hidden initially
            )
            action_texts.append((txt, act_time))

        # --- Main plot: entity artists ---
        markers = []
        labels = []
        trails = []
        for idx, entity in enumerate(entity_names):
            pos0 = marker_positions[entity][0]
            (marker,) = ax.plot(
                [pos0[0]], [pos0[1]], "o",
                color=marker_colors[idx % len(marker_colors)],
                markeredgecolor="black", markeredgewidth=1.0,
                markersize=11, zorder=10, label=entity,
            )
            label = ax.text(
                pos0[0], pos0[1], entity,
                fontsize=6, fontfamily="monospace", fontweight="bold",
                ha="center", va="bottom",
                zorder=11,
            )
            trail = ax.scatter([], [], s=4, zorder=5, alpha=0.7)
            trail.set_cmap(colormaps[idx % len(colormaps)])
            trail.set_clim(0.0, t_end)
            markers.append(marker)
            labels.append(label)
            trails.append(trail)

        ax.legend(fontsize=7, loc="upper right")
        ax.set_title("Entity Trajectories")

        if grid is None:
            ax.autoscale()
            ax.set_aspect("equal", adjustable="datalim")

        # Compute label offset in data coords (fraction of y-axis range)
        y_range = ax.get_ylim()
        label_offset = (y_range[1] - y_range[0]) * 0.02

        def _update(frame: int):
            current_time = frame_times[frame]
            for idx, entity in enumerate(entity_names):
                # Update current position marker
                pos = marker_positions[entity]
                markers[idx].set_data([pos[frame, 0]], [pos[frame, 1]])
                # Update label position (just above the marker)
                labels[idx].set_position((pos[frame, 0], pos[frame, 1] + label_offset))
                # Update trail from the dense pre-computed data,
                # sliced up to the current frame time
                if entity in dense_positions:
                    mask = dense_trail_times <= current_time
                    trails[idx].set_offsets(dense_positions[entity][mask])
                    trails[idx].set_array(dense_trail_times[mask])
            # Reveal actions whose start time has passed
            for txt, act_time in action_texts:
                if current_time >= act_time:
                    txt.set_alpha(1.0)
            # Update goal literal colors based on latest snapshot <= current_time
            current_goals: dict[str, bool] = {}
            for snap_time, snap_goals in goal_snapshots:
                if snap_time <= current_time:
                    current_goals = snap_goals
                else:
                    break
            for txt, literal_str in goal_text_artists:
                if literal_str is not None:
                    satisfied = current_goals.get(literal_str, False)
                    txt.set_color("green" if satisfied else "red")
            return (markers + labels + trails
                    + [txt for txt, _ in action_texts]
                    + [txt for txt, _ in goal_text_artists])

        anim = FuncAnimation(fig, _update, frames=n_frames, blit=False, interval=1000 / fps)

        # Select writer by extension
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext == "gif":
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
        else:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps)

        anim.save(path, writer=writer, dpi=150)
        plt.close(fig)

    def show_plots(
        self,
        *,
        save_plot: str | None = None,
        show_plot: bool = False,
        save_video: str | None = None,
        location_coords: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Convenience method that handles plot/video output based on CLI flags.

        Args:
            save_plot: If set, save the trajectory plot to this file path.
            show_plot: If True, display the trajectory plot interactively.
            save_video: If set, save a trajectory animation to this file path.
            location_coords: Optional explicit location->(x,y) mapping.
        """
        if not save_plot and not show_plot and not save_video:
            return

        if save_plot or show_plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            self.plot_trajectories(ax=ax, location_coords=location_coords)
            if save_plot:
                fig.savefig(save_plot, dpi=300)
                self.console.print(f"Saved plot to [yellow]{save_plot}[/yellow]")
            if show_plot:
                plt.show()
            else:
                plt.close(fig)

        if save_video:
            self.save_trajectory_video(save_video, location_coords=location_coords)
            self.console.print(f"Saved video to [yellow]{save_video}[/yellow]")

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
        coord_lookup, _, _ = self._get_location_coords()
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

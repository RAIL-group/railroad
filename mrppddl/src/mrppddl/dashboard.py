from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.pretty import Pretty
from rich.syntax import Syntax

from mrppddl._bindings import ff_heuristic

console = Console()


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

    def __init__(self, goal_fluents, initial_heuristic=None):
        self.goal_fluents = list(goal_fluents)
        self.num_goals = len(self.goal_fluents)
        self.initial_heuristic = initial_heuristic

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
            Panel("Waiting for first step…", title="State", border_style="green")
        )
        self.layout["debug"].update(
            Panel("No trace yet.", title="MCTS Trace", border_style="magenta")
        )

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
        meta.add_row("Time", f"{sim_state.time:.1f} s")
        meta.add_row(
            "Goals reached",
            f"{self._count_achieved_goals(sim_state)}/{self.num_goals}",
        )
        if step_index is not None:
            meta.add_row("Step", str(step_index))
        if last_action_name is not None:
            meta.add_row("Last action", f"[bold]{last_action_name}[/]")

        # Active fluents
        active_table = Table(
            show_header=True,
            header_style="bold magenta",
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

        # Combine meta + subtables into one grid
        container = Table.grid(padding=1)
        container.add_row(meta)
        container.add_row(goals_table)
        container.add_row(active_table)

        return Panel(
            container,
            title="State",
            border_style="green",
        )

    def _build_trace_panel(self, trace_text: str) -> Panel:
        # text = Text.from_ansi(trace_text) if "\x1b[" in trace_text else Text(trace_text)
        return Panel(
            trace_text,
            title="MCTS Tree Trace",
            border_style="magenta",
            highlight=True,
        )

    def _build_progress_panel(self) -> Panel:
        # Just the single shared Progress instance
        return Panel(self.progress, title="Planner Progress", border_style="cyan")

    # # Stack one or two Progress bars in a small grid
    #     grid = Table.grid(expand=True)
    #     grid.add_row(self.goal_progress)
    #     if self.heuristic_progress is not None:
    #         grid.add_row(self.heuristic_progress)
    #     return Panel(
    #         grid,
    #         title="Planner Progress",
    #         border_style="cyan",
    #     )

    @property
    def renderable(self):
        """Expose the root layout so it can be passed to Live()."""
        return self.layout

    def _count_achieved_goals(self, sim_state) -> int:
        return sum(1 for g in self.goal_fluents if g in sim_state.fluents)

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

    def update(
        self,
        sim_state,
        relevant_fluents,
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
        achieved = self._count_achieved_goals(sim_state)
        self.progress.update(
            self.goal_task_id,
            completed=achieved,
            extra=f"{int(achieved)}/{self.num_goals} goals",
        )

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
            self.layout["debug"].update(self._build_trace_panel(tree_trace))

        from time import sleep
        sleep(0.01)

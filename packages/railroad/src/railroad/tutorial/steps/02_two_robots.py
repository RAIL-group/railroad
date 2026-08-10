"""Step 02 -- add a second robot.

No operator changes at all. A robot is an object of type `robot` plus two
initial fluents, and concurrency falls out of the state semantics: time only
advances when nobody is free, so a second robot simply fills the gap.

Watch the Braille timeline in the summary -- the two rows overlap -- and the
total cost against step 01.
"""

import time
from functools import reduce
from operator import and_

import numpy as np
from rich.console import Console

from railroad import operators
from railroad.bench import BenchmarkCase, benchmark
from railroad.core import Effect, Fluent as F, Operator, State, get_action_by_name
from railroad.dashboard import PlannerDashboard
from railroad.environment import SymbolicEnvironment
from railroad.planner import MCTSPlanner
from railroad.tutorial import report

LOCATIONS = {
    "living_room": (0.0, 0.0),
    "kitchen": (5.0, 0.0),
    "table": (2.0, 3.0),
    "shelf": (8.0, 3.0),
}
OBJECTS = ["Book", "Mug", "Vase"]
NUM_ROBOTS = 2

ROBOT_VELOCITY = 1.0
PICK_TIME = 5.0
PLACE_TIME = 5.0
MAX_STEPS = 40


def move_time(robot: str, loc_from: str, loc_to: str) -> float:
    """Straight-line travel time. Any callable of the parameters will do."""
    a, b = np.array(LOCATIONS[loc_from]), np.array(LOCATIONS[loc_to])
    return float(np.linalg.norm(a - b)) / ROBOT_VELOCITY


class ClearTable(SymbolicEnvironment):
    """Move, pick, place. Nothing is hidden and nothing is uncertain yet."""

    def define_operators(self):
        # Written out rather than taken from railroad.operators so that both
        # halves of a durative action are visible: the robot stops being free
        # and stops being anywhere at t=0, and both facts are restored at the
        # destination when the move lands.
        move = Operator(
            name="move",
            parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
            preconditions=[F("at ?r ?from"), F("free ?r")],
            effects=[
                Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
                Effect(
                    time=(move_time, ["?r", "?from", "?to"]),
                    resulting_fluents={F("free ?r"), F("at ?r ?to")},
                ),
            ],
        )
        return [
            move,
            operators.construct_pick_operator_blocking(PICK_TIME),
            operators.construct_place_operator_blocking(PLACE_TIME),
        ]


def build(num_robots: int = NUM_ROBOTS):
    """The problem: where things are, who is around, and what counts as done."""
    robots = [f"robot{i + 1}" for i in range(num_robots)]
    fluents = {F(f"at {obj} table") for obj in OBJECTS}
    for robot in robots:
        fluents |= {F(f"free {robot}"), F(f"at {robot} living_room")}

    # "None of these objects is on the table" -- a conjunction of negations.
    goal = reduce(and_, [~F(f"at {obj} table") for obj in OBJECTS])

    env = ClearTable(
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": set(LOCATIONS),
            "object": set(OBJECTS),
        },
        seed=0,
    )
    return env, goal


def solve(env, goal, dashboard, *, iterations: int, c: float) -> bool:
    """Replan every time a robot frees up; return whether the goal was met."""
    for _ in range(MAX_STEPS):
        if goal.evaluate(env.state.fluents):
            return True
        actions = env.get_actions()
        planner = MCTSPlanner(actions)
        name = planner(env.state, goal, max_iterations=iterations, c=c, max_depth=20)
        if name == "NONE":
            dashboard.console.print("[yellow]Planner returned NONE.[/yellow]")
            return False
        env.act(get_action_by_name(actions, name))
        dashboard.update(planner, name)
    return goal.evaluate(env.state.fluents)


def relevant(fluent) -> bool:
    return any(word in fluent.name for word in ("at", "holding"))


def demo() -> None:
    env, goal = build()
    with PlannerDashboard(goal, env, fluent_filter=relevant) as dashboard:
        solve(env, goal, dashboard, iterations=4000, c=300)
    report(dashboard)


if __name__ == "__main__":
    demo()


# ---- sweep: press b ---------------------------------------------------------
# Does the second robot pay for itself, and does a third? With three objects and
# three robots the answer collapses: they all drive to the table and each picks
# one. The goal only asks that nothing *remain* on the table, so the run ends at
# the last pick -- one trip plus one pick, and nothing is ever put away.

@benchmark(
    name="s02_two_robots",
    description="Clear a table with 1-3 robots; sweep team size and MCTS iterations.",
    tags=["tutorial"],
    repeat=4,
    timeout=60.0,
)
def bench_two_robots(case: BenchmarkCase) -> dict:
    env, goal = build(case.num_robots)
    console = Console(record=True, force_terminal=True, width=120)
    dashboard = PlannerDashboard(
        goal, env, fluent_filter=relevant, print_on_exit=False, console=console
    )
    started = time.perf_counter()
    solve(env, goal, dashboard, iterations=case.mcts.iterations, c=case.mcts.c)
    dashboard.print_history()
    return {
        "success": goal.evaluate(env.state.fluents),
        "plan_cost": float(env.state.time),
        "wall_time": time.perf_counter() - started,
        "actions_count": len(dashboard.actions_taken),
        "log_html": console.export_html(inline_styles=True),
    }


bench_two_robots.add_cases([
    {"num_robots": num_robots, "mcts.iterations": iterations, "mcts.c": 300}
    for num_robots in (1, 2, 3)
    for iterations in (400, 1000, 4000)
])

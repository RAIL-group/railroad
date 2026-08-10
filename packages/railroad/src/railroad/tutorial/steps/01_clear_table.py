"""Step 01 -- clear the table.

A whole problem: a typed object universe, three operators, a goal made of
negated literals, and the plan-act loop that drives it.

The loop is the thing to look at. There is no plan to execute -- `solve` asks
for one action, hands it to the environment, and asks again once a robot is
free. Replanning is not error recovery here, it is the control structure.
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
ROBOTS = ["robot1"]

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


def build():
    """The problem: where things are, who is around, and what counts as done."""
    fluents = {F(f"at {obj} table") for obj in OBJECTS}
    for robot in ROBOTS:
        fluents |= {F(f"free {robot}"), F(f"at {robot} living_room")}

    # "None of these objects is on the table" -- a conjunction of negations.
    goal = reduce(and_, [~F(f"at {obj} table") for obj in OBJECTS])

    env = ClearTable(
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(ROBOTS),
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
    report(dashboard, step="01")


if __name__ == "__main__":
    demo()


# ---- sweep: press b ---------------------------------------------------------
# How much search does this problem actually need? There is a floor: below a few
# tens of iterations the run fails outright, and where that floor sits depends on
# the exploration constant -- a larger c spends more of a small budget looking
# around. Above ~100 iterations the extra search buys nothing here.

@benchmark(
    name="s01_clear_table",
    description="Clear a table with one robot; sweep MCTS iterations and c.",
    tags=["tutorial"],
    repeat=4,
    timeout=60.0,
)
def bench_clear_table(case: BenchmarkCase) -> dict:
    env, goal = build()
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


bench_clear_table.add_cases([
    {"mcts.iterations": iterations, "mcts.c": c}
    for c in (100, 300)
    for iterations in (10, 25, 50, 100, 400, 1000, 4000)
])

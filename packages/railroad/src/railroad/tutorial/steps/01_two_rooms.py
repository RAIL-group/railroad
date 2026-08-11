"""Step 01 -- one robot, two rooms.

The smallest thing that is still a whole problem: a typed universe, three
operators written out by hand, a goal, and the plan-act loop that drives it.
Then the experiment -- `h_mult=0` takes the value function away.
"""

from functools import reduce
from operator import and_

import numpy as np

from railroad import tutorial
from railroad.bench import BenchmarkCase, benchmark
from railroad.core import Effect, Fluent as F, Operator, State, get_action_by_name
from railroad.environment import SymbolicEnvironment
from railroad.planner import MCTSPlanner

LOCATIONS = {
    "table": (2.0, 1.0),
    "shelf": (5.6, 5.8),
}
OBJECTS = ["Mug", "Book"]

ROBOT_VELOCITY = 1.0
PICK_TIME = 5.0
PLACE_TIME = 5.0
MAX_STEPS = 40


def move_time(robot: str, loc_from: str, loc_to: str) -> float:
    """Straight-line travel time. `@numeric` lets `move_time + 0.1` compose."""
    a, b = np.array(LOCATIONS[loc_from]), np.array(LOCATIONS[loc_to])
    return float(np.linalg.norm(a - b)) / ROBOT_VELOCITY


class TwoRooms(SymbolicEnvironment):
    """Move, pick, place. Nothing is hidden and nothing is uncertain yet."""

    def define_operators(self):
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
        pick = Operator(
            name="pick",
            parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
            preconditions=[
                F("at ?r ?loc"), F("free ?r"), F("at ?obj ?loc"),
                ~F("hand-full ?r"),
            ],
            effects=[
                Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?obj ?loc")}),
                Effect(time=PICK_TIME, resulting_fluents={
                    F("free ?r"), F("holding ?r ?obj"), F("hand-full ?r"),
                }),
            ],
        )
        place = Operator(
            name="place",
            parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
            preconditions=[
                F("at ?r ?loc"), F("free ?r"), F("holding ?r ?obj"),
                F("hand-full ?r"),
            ],
            effects=[
                Effect(time=0,
                       resulting_fluents={~F("free ?r"), ~F("holding ?r ?obj")}),
                Effect(time=PLACE_TIME, resulting_fluents={
                    F("free ?r"), F("at ?obj ?loc"), ~F("hand-full ?r"),
                }),
            ],
        )
        return [move, pick, place]


def build():
    """The problem: where things are, who is around, and what counts as done."""
    fluents = {F(f"at {obj} table") for obj in OBJECTS}
    fluents |= {F("free robot1"), F("at robot1 table")}

    goal = reduce(and_, [F(f"at {obj} shelf") for obj in OBJECTS])

    env = TwoRooms(
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": {"robot1"},
            "location": set(LOCATIONS),
            "object": set(OBJECTS),
        },
        seed=0,
    )
    return env, goal


def solve(env, goal, view, *, iterations: int, c: float, h_mult: float) -> bool:
    """Replan every time a robot frees up; return whether the goal was met."""
    for _ in range(MAX_STEPS):
        if goal.evaluate(env.state.fluents):
            return True
        actions = env.get_actions()
        planner = MCTSPlanner(actions)
        name = planner(env.state, goal, max_iterations=iterations, c=c, max_depth=20,
                       heuristic_multiplier=h_mult)
        if name == "NONE":
            view.console.print("[yellow]Planner returned NONE.[/yellow]")
            return False
        env.act(get_action_by_name(actions, name))
        view.update(planner, name)
    return goal.evaluate(env.state.fluents)


def relevant(fluent) -> bool:
    return any(word in fluent.name for word in ("at", "holding"))


@benchmark(
    name="s01_two_rooms",
    description="Move two objects to the shelf, with and without the value function.",
    tags=["tutorial"],
    repeat=4,
    timeout=60.0,
)
def run(case: BenchmarkCase) -> dict:
    env, goal = build()
    with tutorial.dashboard(case, goal, env, fluent_filter=relevant) as view:
        solve(env, goal, view, iterations=case.mcts.iterations, c=case.mcts.c,
              h_mult=case.mcts.h_mult)
    return tutorial.finish(view, case, location_coords=LOCATIONS)


run.add_cases([
    {"mcts.h_mult": h_mult, "mcts.iterations": iterations, "mcts.c": 300}
    for h_mult in (5.0, 0.0)
    for iterations in (400, 1000, 4000)
])


if __name__ == "__main__":
    tutorial.main(run)

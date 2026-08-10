"""Step 01 -- clear the table.

A whole problem: a typed object universe, three operators, a goal made of
negated literals, and the plan-act loop that drives it.

The operators are written out rather than taken from `railroad.operators`,
because they are the point. Each one is a list of effects at times: what stops
being true when the action starts, and what becomes true when it lands.

The loop is the other thing to look at. There is no plan to execute -- `solve`
asks for one action, hands it to the environment, and asks again once a robot
is free. Replanning is not error recovery here, it is the control structure.
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
    "living_room": (0.0, 0.0),
    "kitchen": (5.0, 0.0),
    "table": (2.0, 3.0),
    "shelf": (8.0, 3.0),
}
OBJECTS = ["Book", "Mug", "Vase"]
NUM_ROBOTS = 1

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
        # Both halves of a durative action are visible here: the robot stops
        # being free and stops being anywhere at t=0, and both facts are
        # restored at the destination when the move lands. Nothing else in the
        # system needs to know that a move "takes time".
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
        # `just-picked` is set when the pick lands and expires a tenth of a
        # second later -- a fluent with a lifetime, which is all it takes to
        # stop a robot putting down what it has only just picked up.
        pick = Operator(
            name="pick",
            parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
            preconditions=[
                F("at ?r ?loc"), F("free ?r"), F("at ?obj ?loc"),
                ~F("hand-full ?r"), ~F("just-placed ?r ?obj"),
            ],
            effects=[
                Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?obj ?loc")}),
                Effect(time=PICK_TIME, resulting_fluents={
                    F("free ?r"), F("holding ?r ?obj"), F("hand-full ?r"),
                    F("just-picked ?r ?obj"),
                }),
                Effect(time=PICK_TIME + 0.1,
                       resulting_fluents={~F("just-picked ?r ?obj")}),
            ],
        )
        place = Operator(
            name="place",
            parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
            preconditions=[
                F("at ?r ?loc"), F("free ?r"), F("holding ?r ?obj"),
                F("hand-full ?r"), ~F("just-picked ?r ?obj"),
            ],
            effects=[
                Effect(time=0,
                       resulting_fluents={~F("free ?r"), ~F("holding ?r ?obj")}),
                Effect(time=PLACE_TIME, resulting_fluents={
                    F("free ?r"), F("at ?obj ?loc"), ~F("hand-full ?r"),
                    F("just-placed ?r ?obj"),
                }),
                Effect(time=PLACE_TIME + 0.1,
                       resulting_fluents={~F("just-placed ?r ?obj")}),
            ],
        )
        return [move, pick, place]


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


def solve(env, goal, view, *, iterations: int, c: float) -> bool:
    """Replan every time a robot frees up; return whether the goal was met."""
    for _ in range(MAX_STEPS):
        if goal.evaluate(env.state.fluents):
            return True
        actions = env.get_actions()
        planner = MCTSPlanner(actions)
        name = planner(env.state, goal, max_iterations=iterations, c=c, max_depth=20)
        if name == "NONE":
            view.console.print("[yellow]Planner returned NONE.[/yellow]")
            return False
        env.act(get_action_by_name(actions, name))
        view.update(planner, name)
    return goal.evaluate(env.state.fluents)


def relevant(fluent) -> bool:
    return any(word in fluent.name for word in ("at", "holding"))


# ---- one function, two ways to run it ---------------------------------------
# `python demo.py` runs case 0 of the sweep below, live, with the dashboard.
# `railroad benchmarks run -i demo.py --tags tutorial` runs all of them, many
# times over, in parallel. Same code either way, so a value you tune here is
# the value the sweep measures.
#
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
def run(case: BenchmarkCase) -> dict:
    env, goal = build()
    with tutorial.dashboard(case, goal, env, fluent_filter=relevant) as view:
        solve(env, goal, view, iterations=case.mcts.iterations, c=case.mcts.c)
    return tutorial.result(view)


run.add_cases([
    # Case 0 is what `python demo.py` runs, so the grid starts at the
    # configuration you would actually use and works down from there.
    {"mcts.iterations": iterations, "mcts.c": c}
    for c in (300, 100)
    for iterations in (4000, 1000, 400, 100, 50, 25, 10)
])


if __name__ == "__main__":
    tutorial.main(run)

"""Step 04 -- add a second robot.

No operator changes at all. A robot is an object of type `robot` plus two
initial fluents, and concurrency falls out of the state semantics: time only
advances when nobody is free, so a second robot simply fills the gap.

Watch the Braille timeline in the summary -- the two rows overlap -- and the
total cost against step 03.
"""

from functools import reduce
from operator import and_

import numpy as np

from railroad import tutorial
from railroad.bench import BenchmarkCase, benchmark
from railroad.core import Effect, Fluent as F, Operator, State, get_action_by_name
from railroad.environment import SymbolicEnvironment
from railroad.operators import numeric
from railroad.planner import MCTSPlanner

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
NO_OP_TIME = 5.0
MAX_STEPS = 40


@numeric
def move_time(robot: str, loc_from: str, loc_to: str) -> float:
    """Straight-line travel time.

    `@numeric` lets it compose like a number, so the `just-moved` expiry below
    is `move_time + 0.1` rather than a second function that only adds to this
    one.
    """
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
            preconditions=[F("at ?r ?from"), F("free ?r"), ~F("just-moved ?r")],
            effects=[
                Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
                Effect(
                    time=(move_time, ["?r", "?from", "?to"]),
                    resulting_fluents={F("free ?r"), F("at ?r ?to"),
                                       F("just-moved ?r")},
                ),
                Effect(time=(move_time + 0.1, ["?r", "?from", "?to"]),
                       resulting_fluents={~F("just-moved ?r")}),
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
        # Step 02's escape hatch, and `just-moved` is why it has to be here: a
        # robot that arrives somewhere with nothing to do there would otherwise
        # have no legal action at all. extra_cost is charged in the objective on
        # top of elapsed time, so waiting stays a last resort.
        no_op = Operator(
            name="no_op",
            parameters=[("?r", "robot")],
            preconditions=[F("free ?r")],
            effects=[
                Effect(time=0, resulting_fluents={~F("free ?r")}),
                Effect(time=NO_OP_TIME, resulting_fluents={F("free ?r")}),
            ],
            extra_cost=100.0,
        )
        return [move, pick, place, no_op]


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
# `uv run python demo.py` runs case 0 of the sweep below, live, with the dashboard.
# `uv run railroad benchmarks run -i demo.py --tags tutorial` runs all of them, many
# times over, in parallel. Same code either way.
#
# Does the second robot pay for itself, and does a third? Measured over 8
# repeats: 44.3 -> 22.1 -> 8.6 seconds, so the second halves it and the third
# more than halves it again. With three objects and three robots the answer
# collapses: they all drive to the table and each picks one, in 6 actions. The
# goal only asks that nothing *remain* on the table, so the run ends at the last
# pick -- one trip plus one pick, and nothing is ever put away.

@benchmark(
    name="s04_two_robots",
    description="Clear a table with 1-3 robots; sweep team size and search budget.",
    tags=["tutorial"],
    repeat=4,
    timeout=60.0,
)
def run(case: BenchmarkCase) -> dict:
    env, goal = build(case.num_robots)
    with tutorial.dashboard(case, goal, env, fluent_filter=relevant) as view:
        solve(env, goal, view, iterations=case.mcts.iterations, c=case.mcts.c)
    # The metrics, plus this run's trajectory as a plot.jpg artifact so the
    # results dashboard shows a picture of every run in the sweep, not just a
    # row of numbers. --save-plot and --save-video are the live equivalents.
    # LOCATIONS puts the rooms where we said.
    return tutorial.finish(view, case, location_coords=LOCATIONS)


run.add_cases([
    # Case 0 is what `uv run python demo.py` runs, so the two-robot team comes first.
    {"num_robots": num_robots, "mcts.iterations": iterations, "mcts.c": 300}
    for num_robots in (2, 1, 3)
    for iterations in (4000, 1000, 400)
])


if __name__ == "__main__":
    tutorial.main(run)

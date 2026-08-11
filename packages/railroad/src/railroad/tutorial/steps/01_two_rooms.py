"""Step 01 -- one robot, two rooms.

The smallest thing that is still a whole problem: two rooms, two objects on the
table, and a robot asked to put them on the shelf. A typed object universe,
three operators, a goal, and the plan-act loop that drives it.

The operators are written out rather than taken from `railroad.operators`,
because they are the point. Each one is a list of effects at times: what stops
being true when the action starts, and what becomes true when it lands.

The loop is the other thing to look at. There is no plan to execute -- `solve`
asks for one action, hands it to the environment, and asks again once the robot
is free. Replanning is not error recovery here, it is the control structure.

And then the experiment: `h_mult`. MCTS scores a leaf by elapsed time plus
`h_mult` times an estimate of the work remaining. Set it to 0 and the estimate
stops counting, so the search picks whatever is cheapest right now. It does not
merely get slower -- it never finishes, at any budget. Watch what it does
instead, because the shape of that failure is the whole of the next step.
"""

from functools import reduce
from operator import and_

import numpy as np

from railroad import tutorial
from railroad.bench import BenchmarkCase, benchmark
from railroad.core import Effect, Fluent as F, Operator, State, get_action_by_name
from railroad.environment import SymbolicEnvironment
from railroad.planner import MCTSPlanner

# Placed on a 3-4-5 triangle: exactly 6.0 apart, and not on a shared axis, so
# the trajectory plot is a diagonal across its frame rather than a flat line
# through the middle of an empty one.
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
    """Straight-line travel time. Any callable of the parameters will do."""
    a, b = np.array(LOCATIONS[loc_from]), np.array(LOCATIONS[loc_to])
    return float(np.linalg.norm(a - b)) / ROBOT_VELOCITY


class TwoRooms(SymbolicEnvironment):
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
        # Same shape. The object stops being on the table the moment the pick
        # starts and is in the hand when it lands: nothing is ever in both
        # places, and nothing is in neither.
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

    # "Both of these are on the shelf" -- a conjunction of literals.
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
    """Replan every time the robot frees up; return whether the goal was met."""
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


# ---- one function, two ways to run it ---------------------------------------
# `uv run python demo.py` runs case 0 of the sweep below, live, with the dashboard.
# `uv run railroad benchmarks run -i demo.py --tags tutorial` runs all of them, many
# times over, in parallel. Same code either way, so a value you tune here is
# the value the sweep measures.
#
# The experiment is h_mult, and the answer is not a slowdown. At the default 5,
# every budget in the grid finds the same 7-action, 38-second plan. At 0,
# nothing finishes -- not at 400 iterations, not at 4000. This is not a search
# that needs more time; it is a search with no gradient to climb.

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
    # The metrics, plus this run's trajectory as a plot.jpg artifact so the
    # results dashboard shows a picture of every run in the sweep, not just a
    # row of numbers. --save-plot and --save-video are the live equivalents.
    # LOCATIONS puts the rooms where we said.
    return tutorial.finish(view, case, location_coords=LOCATIONS)


run.add_cases([
    # Case 0 is what `uv run python demo.py` runs, so the grid starts at the
    # configuration you would actually use. The h_mult=0 half starts at case 3,
    # and `tutorial run --case 3` is how you watch it fail live.
    {"mcts.h_mult": h_mult, "mcts.iterations": iterations, "mcts.c": 300}
    for h_mult in (5.0, 0.0)
    for iterations in (400, 1000, 4000)
])


if __name__ == "__main__":
    tutorial.main(run)

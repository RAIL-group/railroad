"""Step 02 -- stop the robot undoing itself.

One guard per action, each the same three lines: a precondition, a flag set
when the action lands, and an expiry a tenth of a second later. You may not put
down what you have only just picked up, pick up what you have only just put
down, or move twice without doing anything in between.

Why it is needed. Step 01 at h_mult=0 did not wander -- it did something much
more specific. It picked the mug up, put it straight back down, and repeated
that until the step limit, never once leaving the table. Nothing in the problem
forbade it, and with the estimate of work-remaining switched off the planner
scored every leaf by elapsed time alone and took the cheapest action available.
Picking is cheap. Walking to the shelf is not.

`just-moved` closes the other half of that. Guard pick and place alone and the
robot stops churning, but only starts pacing between the two rooms instead --
also cheap, also going nowhere. Guard all three and every remaining legal
action is one that makes progress.

It needs `no_op` to be safe, which is the part worth dwelling on: a guard can
leave a robot with nothing legal to do at all, and a search with no value
function does not stumble into that state, it *aims* for it. A dead end is
where the clock stops.

The sweep is unchanged, so run the same case again -- `tutorial run --case 3`,
still h_mult=0 -- and the search that could not finish walks itself to the goal
in the optimal seven actions.
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
NO_OP_TIME = 5.0
MAX_STEPS = 40


@numeric
def move_time(robot: str, loc_from: str, loc_to: str) -> float:
    """Straight-line travel time. Any callable of the parameters will do.

    `@numeric` is the only new import: it makes the function compose like a
    number, so `move_time + 0.1` below is itself a callable of the same
    parameters. Without it the expiry would need a second function that did
    nothing but call this one and add to it.
    """
    a, b = np.array(LOCATIONS[loc_from]), np.array(LOCATIONS[loc_to])
    return float(np.linalg.norm(a - b)) / ROBOT_VELOCITY


class TwoRooms(SymbolicEnvironment):
    """Move, pick, place. Nothing is hidden and nothing is uncertain yet."""

    def define_operators(self):
        # `just-moved` is the third of the pair-per-action, and the one that
        # does the most work here: it stops a robot chaining move after move
        # without ever doing anything at the far end. Pacing between two rooms
        # is exactly as cheap as pacing back, so nothing else rules it out.
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
        # `just-picked` is set when the pick lands and cleared a tenth of a
        # second later -- a fluent with a lifetime, which is the whole of the
        # mechanism. Nothing needs a timer or a special case: the expiry is an
        # ordinary effect at an ordinary time, and while it is pending the
        # matching `place` has an unsatisfied precondition.
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
        # And the escape hatch `just-moved` makes necessary. Walk to the shelf
        # empty-handed and there is nothing to pick and no second move allowed:
        # without somewhere to put the time that state has no legal action at
        # all, and a search with no value function *prefers* it, because a dead
        # end is the cheapest place to stop the clock. extra_cost is charged in
        # the objective on top of elapsed time, so waiting stays a last resort.
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
# The same grid as step 01, which is the point: compare the two sweeps in the
# dashboard. The h_mult=5 half is unchanged, 38 seconds either way -- a decent
# estimate of work-remaining already knew that undoing yourself is not progress.
# The h_mult=0 half goes from failing at every budget to succeeding at every
# budget, on the optimal plan. Not "needs less search", either: the floor is
# not merely lower than step 01's, it is below any budget worth running. The
# guards did not help the planner look harder, they removed everywhere wrong
# to look.

@benchmark(
    name="s02_action_blocking",
    description="The same two objects, with a guard against undoing the last action.",
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
    # and `tutorial run --case 3` is the one that failed a moment ago.
    {"mcts.h_mult": h_mult, "mcts.iterations": iterations, "mcts.c": 300}
    for h_mult in (5.0, 0.0)
    for iterations in (400, 1000, 4000)
])


if __name__ == "__main__":
    tutorial.main(run)

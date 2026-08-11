"""Step 05 -- hide the objects.

The objects are no longer in the initial state; the robots have to look. That
buys one new operator, and with it a probabilistic effect and a prior over where
things are -- flat here, because we genuinely have no idea. (Step 07 replaces it
with a learned one.)

Two conventions of ObjectSearchEnvironment are doing work now. Search outcomes
resolve against ground truth rather than sampling, so a bad prior costs time but
never invents a discovery. And searching a location *reveals* it: everything
actually there becomes known, and the location can never be searched again.

Which is exactly what the lock is for. `searched ?loc ?obj` only lands when the
search *completes*, so nothing stops two robots holding a search of the same
room at once -- and the planner has every reason to do it, because it reads the
two probabilistic branches as independent draws. Double-searching looks like two
tickets on one lottery rather than one ticket twice. Here it is worse than that:
one search reveals the whole room, so the second robot was never going to learn
anything.

Three lines fix it -- a `lock-search ?loc` precondition, the lock taken when the
search starts and released when it finishes -- and the flag is here so the sweep
can run it both ways. With one robot expect nothing to change: a robot cannot
contend with itself.
"""

from functools import reduce
from operator import and_

import numpy as np

from railroad import tutorial
from railroad.bench import BenchmarkCase, benchmark
from railroad.core import Effect, Fluent as F, Operator, State, get_action_by_name
from railroad.environment import ObjectSearchEnvironment
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
USE_SEARCH_LOCK = True

# Ground truth, known to the environment and not to the planner.
TRUE_LOCATIONS = {
    "table": {"Book"},
    "kitchen": {"Mug"},
    "shelf": {"Vase"},
}

ROBOT_VELOCITY = 1.0
SEARCH_TIME = 5.0
FIND_PROB = 0.5
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


class HouseSearch(ObjectSearchEnvironment):
    """Move and search. Where things are is now the whole problem."""

    def __init__(self, *args, use_search_lock: bool = USE_SEARCH_LOCK, **kwargs):
        # define_operators() runs inside the base __init__, so anything it reads
        # has to be set before that call.
        self.use_search_lock = use_search_lock
        super().__init__(*args, **kwargs)

    def define_operators(self):
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
        # The lock makes searching a room exclusive for as long as it takes.
        locked = [~F("lock-search ?loc")] if self.use_search_lock else []
        starting = {~F("free ?r")}
        finishing = {F("free ?r"), F("searched ?loc ?obj")}
        if self.use_search_lock:
            starting.add(F("lock-search ?loc"))
            finishing.add(~F("lock-search ?loc"))

        # Look for one object in one place. `prob_effects` is the whole of the
        # uncertainty: two branches, weighted, and whichever fires is applied
        # after this effect's own fluents. The planner reasons over both; the
        # environment picks the real one from TRUE_LOCATIONS.
        search = Operator(
            name="search",
            parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
            preconditions=[
                F("at ?r ?loc"),
                F("free ?r"),
                ~F("revealed ?loc"),
                ~F("searched ?loc ?obj"),
                ~F("found ?obj"),
                *locked,
            ],
            effects=[
                Effect(time=0, resulting_fluents=starting),
                Effect(
                    time=SEARCH_TIME,
                    resulting_fluents=finishing,
                    prob_effects=[
                        (FIND_PROB, [Effect(
                            time=0,
                            resulting_fluents={F("found ?obj"), F("at ?obj ?loc")},
                        )]),
                        (1 - FIND_PROB, []),
                    ],
                ),
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
        return [move, search, no_op]


def build(num_robots: int = NUM_ROBOTS,
          use_search_lock: bool = USE_SEARCH_LOCK):
    """The problem: who is around, what we are after, and what we already know."""
    robots = [f"robot{i + 1}" for i in range(num_robots)]
    # 'revealed' means "nothing left to learn here", so the start counts as done.
    fluents = {F("revealed living_room")}
    for robot in robots:
        fluents |= {F(f"free {robot}"), F(f"at {robot} living_room")}

    goal = reduce(and_, [F(f"found {obj}") for obj in OBJECTS])

    env = HouseSearch(
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": set(LOCATIONS),
            "object": set(OBJECTS),
        },
        true_object_locations=TRUE_LOCATIONS,
        use_search_lock=use_search_lock,
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
    return any(word in fluent.name for word in ("at", "found", "searched"))


# ---- one function, two ways to run it ---------------------------------------
# `uv run python demo.py` runs case 0 of the sweep below, live, with the dashboard.
# `uv run railroad benchmarks run -i demo.py --tags tutorial` runs all of them, many
# times over, in parallel. Same code either way.
#
# The A/B, measured over 8 repeats. One robot cannot contend with itself and the
# numbers say so: 28.3 seconds either way. Two robots 19.2 with the lock against
# 22-23 without; three robots 19.2 -> 13.5.
#
# Read the spread as well as the mean. With the lock the cost is the same every
# time (sd 0.0 at two and three robots); without it the same configuration
# scatters over about five seconds, because which room gets double-searched is
# up to the sampling. And 'searches' is the cleanest measure of all, since three
# rooms hold something and so 3 is the floor: a flat 3.0 with the lock at any
# team size, against 3.0, 3.8, 5.8 without. The lock does not make robots
# faster. It stops them buying the same information twice.

@benchmark(
    name="s05_hidden_objects",
    description="Find three hidden objects, with and without the per-room search lock.",
    tags=["tutorial"],
    repeat=8,
    timeout=60.0,
)
def run(case: BenchmarkCase) -> dict:
    env, goal = build(case.num_robots, use_search_lock=case.use_search_lock)
    with tutorial.dashboard(case, goal, env, fluent_filter=relevant) as view:
        solve(env, goal, view, iterations=case.mcts.iterations, c=case.mcts.c)
    # The metrics, plus this run's trajectory as a plot.jpg artifact so the
    # results dashboard shows a picture of every run in the sweep, not just a
    # row of numbers. --save-plot and --save-video are the live equivalents.
    # LOCATIONS puts the rooms where we said.
    return tutorial.finish(view, case, location_coords=LOCATIONS)


run.add_cases([
    # Case 0 is what `uv run python demo.py` runs: two robots, lock on.
    {"num_robots": num_robots, "use_search_lock": use_search_lock,
     "mcts.iterations": 4000, "mcts.c": 300}
    for num_robots in (2, 1, 3)
    for use_search_lock in (True, False)
])


if __name__ == "__main__":
    tutorial.main(run)

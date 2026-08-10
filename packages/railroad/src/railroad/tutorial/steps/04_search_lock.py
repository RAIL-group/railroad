"""Step 04 -- stop two robots searching the same room.

Three lines: a `lock-search ?loc` precondition, the lock taken when the search
starts, and released when it finishes.

Why it is needed. `searched ?loc ?obj` only lands when the search *completes*,
so nothing in step 03 forbade two robots from holding a search of the same room
at once -- and the planner had every reason to do it. It reads the two
probabilistic branches as independent draws, so double-searching looks like two
tickets on one lottery instead of one. In this environment it is worse than
that: one search reveals the whole room, so the second robot was never going to
learn anything.

The flag is here so the sweep can run it both ways. With one robot expect
nothing to change: a robot cannot contend with itself.
"""

from functools import reduce
from operator import and_

import numpy as np

from railroad import tutorial
from railroad.bench import BenchmarkCase, benchmark
from railroad.core import Effect, Fluent as F, Operator, State, get_action_by_name
from railroad.environment import ObjectSearchEnvironment
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
MAX_STEPS = 40


def move_time(robot: str, loc_from: str, loc_to: str) -> float:
    """Straight-line travel time. Any callable of the parameters will do."""
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
            preconditions=[F("at ?r ?from"), F("free ?r")],
            effects=[
                Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
                Effect(
                    time=(move_time, ["?r", "?from", "?to"]),
                    resulting_fluents={F("free ?r"), F("at ?r ?to")},
                ),
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
        return [move, search]


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
# The A/B. Plan cost alone under-reports the fix: with two robots the wasted
# search overlaps a useful one, so the makespan barely moves (18.9 -> 19.1) even
# though a robot-search was thrown away. 'searches' is the honest measure --
# three rooms hold something, so 3 is the floor, and without the lock the count
# climbs with the team (3 / 4 / 5.6). With three robots the contention finally
# costs real time too: 18.7 -> 13.5.

@benchmark(
    name="s04_search_lock",
    description="Find three hidden objects, with and without the per-room search lock.",
    tags=["tutorial"],
    repeat=8,
    timeout=60.0,
)
def run(case: BenchmarkCase) -> dict:
    env, goal = build(case.num_robots, use_search_lock=case.use_search_lock)
    with tutorial.dashboard(case, goal, env, fluent_filter=relevant) as view:
        solve(env, goal, view, iterations=case.mcts.iterations, c=case.mcts.c)
    return tutorial.result(view)


run.add_cases([
    # Case 0 is what `uv run python demo.py` runs: two robots, lock on.
    {"num_robots": num_robots, "use_search_lock": use_search_lock,
     "mcts.iterations": 4000, "mcts.c": 300}
    for num_robots in (2, 1, 3)
    for use_search_lock in (True, False)
])


if __name__ == "__main__":
    tutorial.main(run)

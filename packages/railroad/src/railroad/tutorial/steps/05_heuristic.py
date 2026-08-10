"""Step 05 -- the value function.

Same problem, now with the heuristic exposed. MCTS scores a state with

    lambda_add * h_add + lambda_max * h_max + lambda_ff * h_ff

plus a probabilistic-retry delta, and `heuristic_multiplier` sets how loudly all
of that speaks relative to accumulated cost in the reward.

The retry delta is the part that matters here. h_add and h_ff both come from a
relaxation that assumes an action does what it says; under a 0.5 find
probability that is optimistic by a factor of two, and the delta is what pays
for the searches you expect to have to repeat. Turn the mix towards h_add
(lambda_add=1, lambda_ff=0) and you get the cheapest-achiever estimate; towards
h_ff and you get a relaxed-plan cost that counts shared work once.

Search budget is pinned low on purpose: at 4000 iterations MCTS papers over a
mediocre value function, and the differences vanish.
"""

from functools import reduce
from operator import and_

import numpy as np

from railroad import tutorial
from railroad.bench import BenchmarkCase, benchmark
from railroad.core import Effect, Fluent as F, Operator, State, get_action_by_name
from railroad.environment import ObjectSearchEnvironment
from railroad.planner import MCTSPlanner

# A bigger house, because the four-room version is too easy to tell value
# functions apart on: at any budget worth using, every mix finds the same plan.
LOCATIONS = {
    "living_room": (0.0, 0.0),
    "kitchen": (5.0, 0.0),
    "table": (2.0, 3.0),
    "shelf": (8.0, 3.0),
    "bedroom": (12.0, 0.0),
    "office": (12.0, 6.0),
    "garage": (0.0, 9.0),
    "attic": (7.0, 11.0),
}
OBJECTS = ["Book", "Mug", "Vase"]
NUM_ROBOTS = 2
USE_SEARCH_LOCK = True

# Ground truth, known to the environment and not to the planner.
TRUE_LOCATIONS = {
    "attic": {"Book"},
    "kitchen": {"Mug"},
    "office": {"Vase"},
}

ROBOT_VELOCITY = 1.0
SEARCH_TIME = 5.0
FIND_PROB = 0.5
MAX_STEPS = 40
SEARCH_BUDGET = 400  # small enough that the heuristic still decides things


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


def solve(
    env, goal, view, *,
    iterations: int,
    c: float,
    h_mult: float = 1.0,
    lambda_add: float = 0.5,
    lambda_ff: float = 0.5,
) -> bool:
    """Replan every time a robot frees up; return whether the goal was met."""
    for _ in range(MAX_STEPS):
        if goal.evaluate(env.state.fluents):
            return True
        actions = env.get_actions()
        # lambda_max is the third component (h_max) and defaults to 0; the
        # probabilistic-retry delta rides on top of whatever mix you pick.
        planner = MCTSPlanner(actions, lambda_add=lambda_add, lambda_ff=lambda_ff)
        name = planner(env.state, goal, max_iterations=iterations, c=c, max_depth=20,
                       heuristic_multiplier=h_mult)
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
# Three mixes of the relaxation against three multipliers, at a budget small
# enough that the value function still decides things. Two things to look at:
# h_mult=1 wins for every mix here (40.1 / 40.9 / 41.8, against 2-8s more at
# h_mult=5), so the multiplier matters more than the mix; and the violins are
# not the same width -- pure h_add has a standard deviation of 12-15 against 2.6
# for the balanced mix, so its mean flatters it.

@benchmark(
    name="s05_heuristic",
    description="Sweep the h_add/h_ff mix and the heuristic multiplier at a small "
                "search budget.",
    tags=["tutorial"],
    repeat=8,
    timeout=60.0,
)
def run(case: BenchmarkCase) -> dict:
    env, goal = build()
    with tutorial.dashboard(case, goal, env, fluent_filter=relevant) as view:
        solve(env, goal, view,
              iterations=case.mcts.iterations, c=case.mcts.c,
              h_mult=case.mcts.h_mult,
              lambda_add=case.lambda_add, lambda_ff=case.lambda_ff)
    # --save-plot and --save-video draw the trajectories and the action
    # list; the sweep skips them. LOCATIONS puts the rooms where we said.
    outcome = tutorial.result(view)
    tutorial.show_plots(view, case, location_coords=LOCATIONS)
    return outcome


run.add_cases([
    # Case 0 is what `uv run python demo.py` runs: the balanced mix at h_mult=1.
    {"lambda_add": lambda_add, "lambda_ff": lambda_ff,
     "mcts.h_mult": h_mult, "mcts.iterations": SEARCH_BUDGET, "mcts.c": 300}
    for lambda_add, lambda_ff in ((0.5, 0.5), (1.0, 0.0), (0.0, 1.0))
    for h_mult in (1.0, 2.0, 5.0)
])


if __name__ == "__main__":
    tutorial.main(run)

"""Step 03 -- hide the objects.

The objects are no longer in the initial state; the robots have to look. That
buys one new operator, and with it a probabilistic effect and a prior over
where things are -- flat here, because we genuinely have no idea. (Step 07
replaces it with a learned one.)

Two conventions of ObjectSearchEnvironment are doing work now. Search outcomes
resolve against ground truth rather than sampling, so a bad prior costs time but
never invents a discovery. And searching a location *reveals* it: everything
actually there becomes known, and the location can never be searched again.

Watch what two robots do with that.
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
            ],
            effects=[
                Effect(time=0, resulting_fluents={~F("free ?r")}),
                Effect(
                    time=SEARCH_TIME,
                    resulting_fluents={F("free ?r"), F("searched ?loc ?obj")},
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


def build(num_robots: int = NUM_ROBOTS):
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
# Adding robots has stopped paying. Look at the action list from a two- or
# three-robot run: nothing stops two of them searching the same room at the same
# time, and a flat prior makes that look like a good idea -- two draws at 0.5
# beat one. It is not. One search reveals everything in the room.

@benchmark(
    name="s03_hidden_objects",
    description="Find three hidden objects with 1-3 robots and a flat prior.",
    tags=["tutorial"],
    repeat=4,
    timeout=60.0,
)
def run(case: BenchmarkCase) -> dict:
    env, goal = build(case.num_robots)
    with tutorial.dashboard(case, goal, env, fluent_filter=relevant) as view:
        solve(env, goal, view, iterations=case.mcts.iterations, c=case.mcts.c)
    # --plot and --video draw the trajectories and the action list; the
    # sweep skips them. LOCATIONS is what puts the rooms where we said.
    outcome = tutorial.result(view)
    tutorial.show_plots(view, case, location_coords=LOCATIONS)
    return outcome


run.add_cases([
    # Case 0 is what `uv run python demo.py` runs, so the two-robot team comes first.
    {"num_robots": num_robots, "mcts.iterations": iterations, "mcts.c": 300}
    for num_robots in (2, 1, 3)
    for iterations in (4000, 1000, 400)
])


if __name__ == "__main__":
    tutorial.main(run)

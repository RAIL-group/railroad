"""Step 05 -- hide the objects.

The objects leave the initial state and the robots have to look, which buys a
probabilistic effect and a prior over where things are. `lock-search` stops two
robots buying the same information twice.
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
    """Straight-line travel time. `@numeric` lets `move_time + 0.1` compose."""
    a, b = np.array(LOCATIONS[loc_from]), np.array(LOCATIONS[loc_to])
    return float(np.linalg.norm(a - b)) / ROBOT_VELOCITY


class HouseSearch(ObjectSearchEnvironment):
    """Move and search. Where things are is now the whole problem."""

    def __init__(self, *args, use_search_lock: bool = USE_SEARCH_LOCK, **kwargs):
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
        locked = [~F("lock-search ?loc")] if self.use_search_lock else []
        starting = {~F("free ?r")}
        finishing = {F("free ?r"), F("searched ?loc ?obj")}
        if self.use_search_lock:
            starting.add(F("lock-search ?loc"))
            finishing.add(~F("lock-search ?loc"))

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
    return any(word in fluent.name for word in ("at", "found", "searched"))


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
        solve(env, goal, view, iterations=case.mcts.iterations, c=case.mcts.c,
              h_mult=case.mcts.h_mult)
    return tutorial.finish(view, case, location_coords=LOCATIONS)


run.add_cases([
    {"num_robots": num_robots, "use_search_lock": use_search_lock,
     "mcts.iterations": 4000, "mcts.h_mult": 5.0, "mcts.c": 300}
    for num_robots in (1, 2, 3)
    for use_search_lock in (True, False)
])


if __name__ == "__main__":
    tutorial.main(run, default_case=2)

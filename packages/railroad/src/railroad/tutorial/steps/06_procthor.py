"""Step 06 -- a real house.

The operators barely change. What changes is where their numbers come from:
move times are now Theta* paths over the scene's occupancy grid rather than
straight lines, and the locations are the containers of a ProcTHOR-generated
home rather than four names in a dict.

Two things worth saying out loud. The search operator below is the one from
step 04, lock and all -- and it is also, line for line, what
`railroad.operators.construct_search_operator` builds, so everything we wrote
by hand is what the library was doing anyway. And the find probability is now a
*callable* rather than the flat 0.5, which is the seam step 07 pulls on.

    uv run railroad tutorial run --save-video house.mp4

renders the run over the scene's top-down view, into media/.
"""

import random
from functools import reduce
from operator import and_

from railroad import tutorial
from railroad.bench import BenchmarkCase, benchmark
from railroad.core import Effect, Fluent as F, Operator, State, get_action_by_name
from railroad.environment.procthor import ProcTHOREnvironment
from railroad.planner import MCTSPlanner

SCENE_SEED = 8613  # cached in this checkout, so no Unity and no network
NUM_ROBOTS = 2
NUM_OBJECTS = 2

SEARCH_TIME = 10.0
PICK_TIME = 10.0
PLACE_TIME = 10.0
NO_OP_TIME = 5.0
# Generous: a run that stops here has genuinely wandered, rather than been cut
# off mid-search. The learned prior in step 07 needs the room.
MAX_STEPS = 100
# 8 containers and 13 locations ground out to ~440 actions with two robots, so
# this runs in well under a minute. Bigger seeds (8616, 8617) are prettier and
# several times slower; 8616 does not finish at all inside MAX_STEPS here.
SEARCH_BUDGET = 1000


class HouseSearch(ProcTHOREnvironment):
    """Search a ProcTHOR home and deliver what you find."""

    def move_time(self, robot: str, loc_from: str, loc_to: str) -> float:
        """Theta* over the occupancy grid, not a straight line any more."""
        return self.estimate_move_time(robot, loc_from, loc_to)

    def move_expiry(self, robot: str, loc_from: str, loc_to: str) -> float:
        return self.move_time(robot, loc_from, loc_to) + 0.1

    def find_prob(self, robot: str, location: str, obj: str) -> float:
        """A hand-tuned prior, and a generous one: it reads ground truth.

        0.8 where the object actually is, 0.1 everywhere else. Step 07 replaces
        it with a model that has never seen the answer.
        """
        del robot
        for loc, objects in self.scene.object_locations.items():
            if obj in objects:
                return 0.8 if loc == location else 0.1
        return 0.1

    def miss_prob(self, robot: str, location: str, obj: str) -> float:
        return 1.0 - self.find_prob(robot, location, obj)

    def define_operators(self):
        # Same shapes as steps 01 and 04. The `just-moved` pair keeps a robot
        # from chaining moves without ever doing anything at the far end.
        move = Operator(
            name="move",
            parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
            preconditions=[F("at ?r ?from"), F("free ?r"), ~F("just-moved ?r")],
            effects=[
                Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
                Effect(time=(self.move_time, ["?r", "?from", "?to"]),
                       resulting_fluents={F("free ?r"), F("at ?r ?to"),
                                          F("just-moved ?r")}),
                Effect(time=(self.move_expiry, ["?r", "?from", "?to"]),
                       resulting_fluents={~F("just-moved ?r")}),
            ],
        )
        # Step 04's search, with the constant probability replaced by a pair of
        # callables of (robot, location, object). Nothing else knows or cares
        # where those numbers come from.
        search = Operator(
            name="search",
            parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
            preconditions=[
                F("at ?r ?loc"), F("free ?r"),
                ~F("revealed ?loc"), ~F("searched ?loc ?obj"), ~F("found ?obj"),
                ~F("lock-search ?loc"),
            ],
            effects=[
                Effect(time=0, resulting_fluents={~F("free ?r"),
                                                  F("lock-search ?loc")}),
                Effect(
                    time=SEARCH_TIME,
                    resulting_fluents={F("free ?r"), F("searched ?loc ?obj"),
                                       ~F("lock-search ?loc")},
                    prob_effects=[
                        ((self.find_prob, ["?r", "?loc", "?obj"]),
                         [Effect(time=0, resulting_fluents={F("found ?obj"),
                                                            F("at ?obj ?loc")})]),
                        ((self.miss_prob, ["?r", "?loc", "?obj"]), []),
                    ],
                ),
            ],
        )
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
        # Waiting has to be an option, and has to be unattractive: extra_cost is
        # charged in the objective on top of elapsed time.
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
        return [move, search, pick, place, no_op]


def build(seed: int = SCENE_SEED, num_robots: int = NUM_ROBOTS,
          num_objects: int = NUM_OBJECTS):
    """Load the scene, pick targets, and say where they have to end up."""
    robots = [f"robot{i + 1}" for i in range(num_robots)]
    fluents = {F("revealed start_loc")}
    for robot in robots:
        fluents |= {F(f"free {robot}"), F(f"at {robot} start_loc")}

    # The scene lives inside the environment, so the object universe has to be
    # filled in afterwards -- and the grounding cache invalidated by hand,
    # because it cannot see that the operators now close over new targets.
    env = HouseSearch(
        seed=seed,
        state=State(0.0, fluents, []),
        objects_by_type={"robot": set(robots), "location": {"start_loc"}},
    )
    rng = random.Random(seed)
    everything = sorted({o for objs in env.scene.object_locations.values()
                         for o in objs})
    targets = rng.sample(everything, k=min(num_objects, len(everything)))
    destination = rng.choice(sorted(env.scene.object_locations))

    env.objects_by_type["location"] = set(env.scene.locations)
    env.objects_by_type["object"] = set(targets)
    env.invalidate_grounding()

    # 'found' is left implicit: the heuristic's at-implies-found augmentation
    # knows an object's location can only be established by finding it.
    goal = reduce(and_, [F(f"at {obj} {destination}") for obj in targets])
    return env, goal


def solve(env, goal, view, *, iterations: int, c: float,
          h_mult: float = 2.0) -> bool:
    """Replan every time a robot frees up; return whether the goal was met."""
    for _ in range(MAX_STEPS):
        if goal.evaluate(env.state.fluents):
            return True
        actions = env.get_actions()
        planner = MCTSPlanner(actions)
        name = planner(env.state, goal, max_iterations=iterations, c=c,
                       max_depth=20, heuristic_multiplier=h_mult)
        if name == "NONE":
            view.console.print("[yellow]Planner returned NONE.[/yellow]")
            return False
        env.act(get_action_by_name(actions, name))
        view.update(planner, name)
    return goal.evaluate(env.state.fluents)


def relevant(fluent) -> bool:
    return any(word in fluent.name for word in ("at", "holding", "found", "searched"))


# ---- one function, two ways to run it ---------------------------------------
# `uv run python demo.py` runs case 0 of the sweep below, live, with the dashboard.
# `uv run railroad benchmarks run -i demo.py --tags tutorial` runs all of them, many
# times over, in parallel. Same code either way.
#
# Two houses, one and two robots. Scenes differ enough in size and layout that
# the absolute costs are not comparable across seeds -- the question is whether
# the second robot still pays once travel is real geometry rather than a
# straight line.

@benchmark(
    name="s06_procthor",
    description="Search-and-deliver in ProcTHOR homes; sweep scene and team size.",
    tags=["tutorial"],
    repeat=3,
    timeout=300.0,
)
def run(case: BenchmarkCase) -> dict:
    env, goal = build(seed=case.scene_seed, num_robots=case.num_robots)
    print(f"scene {case.scene_seed}: {len(env.scene.object_locations)} containers, "
          f"{len(env.scene.objects)} objects")
    with tutorial.dashboard(case, goal, env, fluent_filter=relevant) as view:
        solve(env, goal, view, iterations=case.mcts.iterations, c=case.mcts.c)
    # Measure first, draw second: rendering an MP4 of a 30-action run takes
    # minutes, and it is not part of what the run cost.
    outcome = tutorial.result(view)
    tutorial.show_plots(view, case)
    return outcome


run.add_cases([
    # Case 0 is what `uv run python demo.py` runs: seed 8613 with two robots.
    {"scene_seed": scene_seed, "num_robots": num_robots,
     "mcts.iterations": SEARCH_BUDGET, "mcts.c": 300}
    for scene_seed in (8613, 8612)
    for num_robots in (2, 1)
])


if __name__ == "__main__":
    tutorial.main(run)

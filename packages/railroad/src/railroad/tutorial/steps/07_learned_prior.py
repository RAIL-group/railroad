"""Step 07 -- stop cheating.

The prior stops reading ground truth and becomes a small network over the
names of things. One function changes.
"""

import random
from functools import reduce
from operator import and_

from railroad import tutorial
from railroad.bench import BenchmarkCase, benchmark
from railroad.core import Effect, Fluent as F, Operator, State, get_action_by_name
from railroad.environment.procthor import ProcTHOREnvironment
from railroad.planner import MCTSPlanner

SCENE_SEED = 8612
NUM_ROBOTS = 2
NUM_OBJECTS = 2
USE_LEARNED_PRIOR = True

SEARCH_TIME = 10.0
PICK_TIME = 10.0
PLACE_TIME = 10.0
NO_OP_TIME = 5.0
MAX_STEPS = 100
SEARCH_BUDGET = 1000


class HouseSearch(ProcTHOREnvironment):
    """Search a ProcTHOR home and deliver what you find."""

    def move_time(self, robot: str, loc_from: str, loc_to: str) -> float:
        """Theta* over the occupancy grid, not a straight line any more."""
        return self.estimate_move_time(robot, loc_from, loc_to)

    def move_expiry(self, robot: str, loc_from: str, loc_to: str) -> float:
        return self.move_time(robot, loc_from, loc_to) + 0.1

    use_learned_prior: bool = USE_LEARNED_PRIOR

    def find_prob(self, robot: str, location: str, obj: str) -> float:
        """Either estimate, behind one signature the operator cannot tell apart."""
        if self.use_learned_prior:
            return self.learned(robot, location, obj)
        for loc, objects in self.scene.object_locations.items():
            if obj in objects:
                return 0.8 if loc == location else 0.1
        return 0.1

    @property
    def learned(self):
        """The packaged network, built once and cached per object internally."""
        if not hasattr(self, "_learned_fn"):
            from railroad.environment.procthor.learning.utils import (
                get_default_fcnn_model_path,
            )
            self._learned_fn = self.scene.get_object_find_prob_fn(
                nn_model_path=str(get_default_fcnn_model_path())
            )
        return self._learned_fn

    def miss_prob(self, robot: str, location: str, obj: str) -> float:
        return 1.0 - self.find_prob(robot, location, obj)

    def define_operators(self):
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
          num_objects: int = NUM_OBJECTS,
          use_learned_prior: bool = USE_LEARNED_PRIOR):
    """Load the scene, pick targets, and say where they have to end up."""
    robots = [f"robot{i + 1}" for i in range(num_robots)]
    fluents = {F("revealed start_loc")}
    for robot in robots:
        fluents |= {F(f"free {robot}"), F(f"at {robot} start_loc")}

    env = HouseSearch(
        seed=seed,
        state=State(0.0, fluents, []),
        objects_by_type={"robot": set(robots), "location": {"start_loc"}},
    )
    env.use_learned_prior = use_learned_prior
    rng = random.Random(seed)
    everything = sorted({o for objs in env.scene.object_locations.values()
                         for o in objs})
    targets = rng.sample(everything, k=min(num_objects, len(everything)))
    destination = rng.choice(sorted(env.scene.object_locations))

    env.objects_by_type["location"] = set(env.scene.locations)
    env.objects_by_type["object"] = set(targets)
    env.invalidate_grounding()

    goal = reduce(and_, [F(f"at {obj} {destination}") for obj in targets])
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
    return any(word in fluent.name for word in ("at", "holding", "found", "searched"))


def show_beliefs(env) -> None:
    """Rank every container for every target -- the model, before any planning."""
    truth = {obj: loc for loc, objs in env.scene.object_locations.items()
             for obj in objs}
    for obj in sorted(env.objects_by_type["object"]):
        scored = sorted(
            ((env.find_prob("robot1", loc, obj), loc)
             for loc in env.scene.object_locations),
            reverse=True,
        )
        print(f"\n  {obj}")
        for prob, loc in scored:
            here = "  <- actually here" if truth.get(obj) == loc else ""
            print(f"    {prob:.3f}  {loc}{here}")


@benchmark(
    name="s07_learned_prior",
    description="Search-and-deliver in ProcTHOR homes, ground-truth prior vs the "
                "packaged learned model.",
    tags=["tutorial"],
    repeat=8,
    timeout=300.0,
)
def run(case: BenchmarkCase) -> dict:
    env, goal = build(seed=case.scene_seed, num_robots=NUM_ROBOTS,
                      use_learned_prior=case.use_learned_prior)
    print(f"scene {case.scene_seed}: {len(env.scene.object_locations)} containers, "
          f"{len(env.scene.objects)} objects, "
          f"{'learned' if env.use_learned_prior else 'hand-tuned'} prior")
    show_beliefs(env)
    with tutorial.dashboard(case, goal, env, fluent_filter=relevant) as view:
        solve(env, goal, view, iterations=case.mcts.iterations, c=case.mcts.c,
              h_mult=case.mcts.h_mult)
    return tutorial.finish(view, case)


run.add_cases([
    {"scene_seed": scene_seed, "use_learned_prior": use_learned_prior,
     "mcts.iterations": SEARCH_BUDGET, "mcts.h_mult": 2.0, "mcts.c": 300}
    for scene_seed in (8612, 8613)
    for use_learned_prior in (True, False)
])


if __name__ == "__main__":
    tutorial.main(run)

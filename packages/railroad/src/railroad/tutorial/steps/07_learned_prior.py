"""Step 07 -- stop cheating.

The prior in step 06 read `scene.object_locations` -- it was an oracle wearing
a probability's clothes. This one has never seen the answer. It is a small
network over SBERT embeddings of the *names* (room, container, object), shipped
with the package at ~30MB, and it answers "would you expect a pillow on a bed?"

The demo prints the ranked containers before it plans anything. That table is
the step; the plan afterwards is the consequence. (It also warms the estimator's
per-object cache, so the first search does not pay for it mid-run.)

Nothing else changes. The find probability is a callable of (robot, location,
object) either way, which is the whole point: swapping a hand-written estimate
for a learned one is a change to one function, not to the domain.

    railroad tutorial run --video learned.mp4
"""

import random
import time
from functools import reduce
from operator import and_

from rich.console import Console

from railroad import operators
from railroad.bench import BenchmarkCase, benchmark
from railroad.core import Fluent as F, State, get_action_by_name
from railroad.dashboard import PlannerDashboard
from railroad.environment.procthor import ProcTHOREnvironment
from railroad.planner import MCTSPlanner
from railroad.tutorial import media_args, report

# A smaller house than step 06: this step is about the table, and 8612 both
# prints a cleaner one and finishes in a quarter of the time. The sweep
# below covers 8613 too, where the model has a harder time.
SCENE_SEED = 8612
NUM_ROBOTS = 2
NUM_OBJECTS = 2
USE_LEARNED_PRIOR = True

SEARCH_TIME = 10.0
PICK_TIME = 10.0
PLACE_TIME = 10.0
# Generous: a run that stops here has genuinely wandered, rather than
# been cut off mid-search. The learned prior in step 07 needs the room.
MAX_STEPS = 100
# 8 containers and 13 locations ground out to ~440 actions with two robots, so
# this runs in well under a minute. Bigger seeds (8616, 8617) are prettier and
# several times slower; 8616 does not finish at all inside MAX_STEPS here.
SEARCH_BUDGET = 1000


class HouseSearch(ProcTHOREnvironment):
    """Search a ProcTHOR home and deliver what you find."""

    use_learned_prior: bool = USE_LEARNED_PRIOR

    def find_prob(self, robot: str, location: str, obj: str) -> float:
        """Either estimate, behind one signature the operator cannot tell apart."""
        if self.use_learned_prior:
            return self._learned(robot, location, obj)
        # The step 06 prior: generous, and cheating -- it reads ground truth.
        del robot
        for loc, objects in self.scene.object_locations.items():
            if obj in objects:
                return 0.8 if loc == location else 0.1
        return 0.1

    @property
    def _learned(self):
        """The packaged network, built once and cached per object internally."""
        if not hasattr(self, "_learned_fn"):
            from railroad.environment.procthor.learning.utils import (
                get_default_fcnn_model_path,
            )
            self._learned_fn = self.scene.get_object_find_prob_fn(
                nn_model_path=str(get_default_fcnn_model_path())
            )
        return self._learned_fn

    def define_operators(self):
        return [
            # Travel time comes from the occupancy grid now, not from geometry
            # we made up. Everything else is the library's.
            operators.construct_move_operator_blocking(self.estimate_move_time),
            operators.construct_search_operator(self.find_prob, SEARCH_TIME),
            operators.construct_pick_operator_blocking(PICK_TIME),
            operators.construct_place_operator_blocking(PLACE_TIME),
            # Waiting has to be an option, and has to be unattractive.
            operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0),
        ]


def build(seed: int = SCENE_SEED, num_robots: int = NUM_ROBOTS,
          num_objects: int = NUM_OBJECTS,
          use_learned_prior: bool = USE_LEARNED_PRIOR):
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
    env.use_learned_prior = use_learned_prior
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


def solve(env, goal, dashboard, *, iterations: int, c: float,
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
            dashboard.console.print("[yellow]Planner returned NONE.[/yellow]")
            return False
        env.act(get_action_by_name(actions, name))
        dashboard.update(planner, name)
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
            # The marker is ground truth, shown for us and not for the planner.
            here = "  <- actually here" if truth.get(obj) == loc else ""
            print(f"    {prob:.3f}  {loc}{here}")


def demo() -> None:
    env, goal = build()
    print(f"scene {SCENE_SEED}: {len(env.scene.object_locations)} containers, "
          f"{len(env.scene.objects)} objects, "
          f"{'learned' if env.use_learned_prior else 'hand-tuned'} prior")
    show_beliefs(env)
    with PlannerDashboard(goal, env, fluent_filter=relevant) as dashboard:
        solve(env, goal, dashboard, iterations=SEARCH_BUDGET, c=300)
    report(dashboard, step="07")
    dashboard.show_plots(**media_args())


if __name__ == "__main__":
    demo()


# ---- sweep: press b ---------------------------------------------------------
# The oracle against the model, on two houses, and the model loses. Measured
# over 3 repeats: on 8612, 322.8 against 437.4; on 8613, 344.2 against 554.9,
# and the learned prior finishes only two runs in three. That gap is the price
# of not knowing -- the hand-tuned prior reads the answer off ground truth, so
# it is an upper bound nothing deployable can reach. What the sweep measures is
# how much the uncertainty costs, which is the number worth trying to shrink.

@benchmark(
    name="s07_learned_prior",
    description="Search-and-deliver in ProcTHOR homes, ground-truth prior vs the "
                "packaged learned model.",
    tags=["tutorial"],
    repeat=3,
    timeout=300.0,
)
def bench_learned_prior(case: BenchmarkCase) -> dict:
    env, goal = build(seed=case.scene_seed, num_robots=case.num_robots,
                      use_learned_prior=case.use_learned_prior)
    console = Console(record=True, force_terminal=True, width=120)
    dashboard = PlannerDashboard(
        goal, env, fluent_filter=relevant, print_on_exit=False, console=console
    )
    started = time.perf_counter()
    solve(env, goal, dashboard, iterations=case.mcts.iterations, c=case.mcts.c)
    dashboard.print_history()
    actions = [name for name, _ in dashboard.actions_taken]
    return {
        "success": goal.evaluate(env.state.fluents),
        "plan_cost": float(env.state.time),
        "wall_time": time.perf_counter() - started,
        "actions_count": len(actions),
        "searches": len([n for n in actions if n.startswith("search")]),
        "log_html": console.export_html(inline_styles=True),
    }


bench_learned_prior.add_cases([
    {"scene_seed": scene_seed, "use_learned_prior": use_learned_prior,
     "num_robots": 2, "mcts.iterations": 1000, "mcts.c": 300}
    for scene_seed in (8612, 8613)
    for use_learned_prior in (False, True)
])

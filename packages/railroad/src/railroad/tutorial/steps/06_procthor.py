"""Step 06 -- a real house.

The operators barely change. What changes is where their numbers come from:
move times are now Theta* paths over the scene's occupancy grid rather than
straight lines, and the locations are the containers of a ProcTHOR-generated
home rather than four names in a dict.

Note that `railroad.operators.construct_search_operator` already carries the
lock from step 04 -- that is the library version of the operator we wrote by
hand. The task grows a delivery half too, so pick and place are back.

    railroad tutorial run --video house.mp4

renders the run over the scene's top-down view.
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

SCENE_SEED = 8613  # cached in this checkout, so no Unity and no network
NUM_ROBOTS = 2
NUM_OBJECTS = 2

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


def demo() -> None:
    env, goal = build()
    print(f"scene {SCENE_SEED}: {len(env.scene.object_locations)} containers, "
          f"{len(env.scene.objects)} objects")
    with PlannerDashboard(goal, env, fluent_filter=relevant) as dashboard:
        solve(env, goal, dashboard, iterations=SEARCH_BUDGET, c=300)
    report(dashboard, step="06")
    dashboard.show_plots(**media_args())


if __name__ == "__main__":
    demo()


# ---- sweep: press b ---------------------------------------------------------
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
def bench_procthor(case: BenchmarkCase) -> dict:
    env, goal = build(seed=case.scene_seed, num_robots=case.num_robots)
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


bench_procthor.add_cases([
    {"scene_seed": scene_seed, "num_robots": num_robots,
     "mcts.iterations": 1000, "mcts.c": 300}
    for scene_seed in (8612, 8613)
    for num_robots in (1, 2)
])

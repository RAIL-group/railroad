"""
ProcTHOR Multi-Robot Search Benchmark

Wraps the procthor_search example as a benchmark case. Robots must search
a ProcTHOR-generated household scene to find target objects and bring them
to a designated room.

Modeled on movie_night.py for benchmark plumbing and on
railroad.examples.procthor_search for the environment / operator setup.
"""

import time
import itertools
import random
from functools import reduce
from operator import and_

from rich.console import Console

from railroad.core import Fluent as F, Operator, State, get_action_by_name
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.bench import benchmark, BenchmarkCase


def _sample_objects_and_location(scene, num_objects: int, seed: int | None):
    rng = random.Random(seed)
    all_objects = sorted({
        obj
        for objs in scene.object_locations.values()
        for obj in objs
    })
    all_locations = sorted(scene.object_locations.keys())
    return (
        rng.sample(all_objects, k=min(num_objects, len(all_objects))),
        rng.choice(all_locations),
    )


@benchmark(
    name="procthor_search",
    description="Multi-robot search in a ProcTHOR-generated household scene.",
    tags=["multi-agent", "search", "procthor"],
    timeout=600.0,
    repeat=15,
)
def bench_procthor_search(case: BenchmarkCase):
    from railroad.environment.procthor import ProcTHOREnvironment

    num_robots = case.params["num_robots"]
    num_objects = case.params["num_objects"]
    scene_seed = case.params["scene_seed"]
    sample_seed = case.params.get("sample_seed", scene_seed)

    class SearchProcTHOREnvironment(ProcTHOREnvironment):
        """Bench-local ProcTHOR env with internal operator construction."""

        def set_target_objects(self, target_objects: list[str]) -> None:
            self._target_objects_for_search = list(target_objects)
            self.objects_by_type["object"] = set(target_objects)
            self._operators = self.define_operators()

        def define_operators(self) -> list[Operator]:
            def object_find_prob_fn(robot: str, location: str, obj: str) -> float:
                del robot
                for loc, objs in self.scene.object_locations.items():
                    if obj in objs:
                        return 0.8 if loc == location else 0.1
                return 0.1

            move_op = operators.construct_move_operator_blocking(self.estimate_move_time)
            search_op = operators.construct_search_operator(object_find_prob_fn, 10.0)
            pick_op = operators.construct_pick_operator_blocking(10.0)
            place_op = operators.construct_place_operator_blocking(10.0)
            no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)
            return [no_op, pick_op, place_op, move_op, search_op]

    robot_names = [f"robot{i + 1}" for i in range(num_robots)]

    initial_fluents = {F("revealed start_loc")}
    for robot in robot_names:
        initial_fluents.add(F(f"at {robot} start_loc"))
        initial_fluents.add(F(f"free {robot}"))
    initial_state = State(0.0, initial_fluents, [])

    env = SearchProcTHOREnvironment(
        seed=scene_seed,
        state=initial_state,
        objects_by_type={
            "robot": set(robot_names),
            "location": {"start_loc"},
        },
    )
    env.objects_by_type["location"] = set(env.scene.locations.keys())

    target_objects, target_location = _sample_objects_and_location(
        env.scene,
        num_objects=num_objects,
        seed=sample_seed,
    )
    env.set_target_objects(target_objects)

    goal = reduce(and_, [
        F(f"at {obj} {target_location}") & F(f"found {obj}")
        for obj in target_objects
    ])

    max_iterations = 60

    recording_console = Console(record=True, force_terminal=True, width=120)

    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "holding", "found", "searched"])

    dashboard = PlannerDashboard(
        goal, env, fluent_filter=fluent_filter,
        print_on_exit=False, console=recording_console,
    )

    start_time = time.perf_counter()

    for _iteration in range(max_iterations):
        if goal.evaluate(env.state.fluents):
            break

        all_actions = env.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(
            env.state, goal,
            max_iterations=case.mcts.iterations,
            c=case.mcts.c,
            max_depth=20,
            heuristic_multiplier=case.mcts.h_mult,
        )

        if action_name == "NONE":
            dashboard.console.print("No more actions available. Goal may not be achievable.")
            break

        action = get_action_by_name(all_actions, action_name)
        env.act(action)
        dashboard.update(mcts, action_name)

    actions_taken = [name for name, _ in dashboard.actions_taken]
    dashboard.print_history()
    html_output = recording_console.export_html(inline_styles=True)

    result = {
        "success": goal.evaluate(env.state.fluents),
        "wall_time": time.perf_counter() - start_time,
        "plan_cost": float(env.state.time),
        "actions_count": len(actions_taken),
        "actions": actions_taken,
        "log_html": html_output,
    }

    try:
        location_coords = {
            name: (float(coord[0]), float(coord[1]))
            for name, coord in env.scene.locations.items()
        }
        plot_image = dashboard.get_plot_image(location_coords=location_coords)
        if plot_image is not None:
            result["log_plot"] = plot_image
    except Exception as e:
        print(f"Failed to render trajectory plot: {e}")

    return result


bench_procthor_search.add_cases([
    {
        "mcts.iterations": iterations,
        "mcts.c": c,
        "mcts.h_mult": h_mult,
        "num_robots": num_robots,
        "num_objects": num_objects,
        "scene_seed": scene_seed,
    }
    for c, num_robots, h_mult, iterations, num_objects, scene_seed in itertools.product(
        [400],               # mcts.c
        [1, 2, 3],           # num_robots
        [2, 4, 10],          # mcts.h_mult
        [4000, 10000],       # mcts.iterations
        [2],                 # num_objects
        list(range(8610, 8620)),  # scene_seed
    )
])

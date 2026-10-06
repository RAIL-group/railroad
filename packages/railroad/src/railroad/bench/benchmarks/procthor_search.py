"""
ProcTHOR Multi-Robot Search Benchmark

Wraps the procthor_search example as a benchmark case. Robots must search
a ProcTHOR-generated household scene to find target objects and bring them
to a designated room.

Registered once per planner configuration, so each benchmark's cases vary
only in the problem: `procthor_search` runs the planner's defaults (the
concurrent heuristic with MaxUCT backup), and `procthor_search_ff` the FF
heuristic with mean backup, the original setup.

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
from railroad.bench.benchmarks._helpers import capture_timeout_log


def _sample_objects_and_location(scene, num_objects: int, seed: int | None):
    """Draw the target objects and the location they must be brought to.

    Targets are nested across `num_objects` and share one location, so a sweep
    over the count changes nothing else. The location is drawn after the first
    two targets, which keeps the two-object problems as they were.
    """
    rng = random.Random(seed)
    all_objects = sorted({
        obj
        for objs in scene.object_locations.values()
        for obj in objs
    })
    all_locations = sorted(scene.object_locations.keys())
    objects = rng.sample(all_objects, k=min(2, len(all_objects)))
    location = rng.choice(all_locations)
    rest = [obj for obj in all_objects if obj not in objects]
    objects += rng.sample(rest, k=min(max(num_objects - 2, 0), len(rest)))
    return objects[:num_objects], location


def _run_procthor_search(case: BenchmarkCase, **planner_kwargs):
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
            if case.params.get("find_prob", "oracle") == "learned":
                # The packaged ProcTHOR model, as `railroad example procthor-search
                # --estimate-object-find-prob`; loaded once per environment.
                if not hasattr(self, "_learned_find_prob_fn"):
                    from railroad.environment.procthor.learning.utils import get_default_fcnn_model_path
                    self._learned_find_prob_fn = self.scene.get_object_find_prob_fn(
                        nn_model_path=str(get_default_fcnn_model_path()),
                    )
                object_find_prob_fn = self._learned_find_prob_fn
            else:
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
            no_op = operators.construct_no_op_operator(no_op_time=5.0)
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

    # `found {obj}` is intentionally left implicit: the heuristics'
    # "at implies found" augmentation infers that an object's location can
    # only be established by finding it.
    goal = reduce(and_, [
        F(f"at {obj} {target_location}")
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

    # If the harness timeout fires mid-loop, still log the in-progress dashboard.
    with capture_timeout_log(case, dashboard):
        for _iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions, **planner_kwargs)
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


@benchmark(
    name="procthor_search",
    description="Multi-robot search in a ProcTHOR-generated household scene.",
    tags=["multi-agent", "search", "procthor"],
    timeout=600.0,
    repeat=15,
)
def bench_procthor_search(case: BenchmarkCase):
    return _run_procthor_search(case)


@benchmark(
    name="procthor_search_ff",
    description="procthor_search planned with the FF heuristic and mean backup.",
    tags=["multi-agent", "search", "procthor"],
    timeout=600.0,
    repeat=15,
)
def bench_procthor_search_ff(case: BenchmarkCase):
    return _run_procthor_search(case, heuristic="ff", backup="mean")


def _cases(h_mult: float) -> list[dict]:
    return [
        {
            "mcts.iterations": iterations,
            "mcts.c": c,
            "mcts.h_mult": h_mult,
            "find_prob": find_prob,
            "num_robots": num_robots,
            "num_objects": num_objects,
            "scene_seed": scene_seed,
        }
        for scene_seed, c, num_robots, find_prob, iterations, num_objects in itertools.product(
            list(range(8610, 8620)),  # scene_seed
            [400],                    # mcts.c
            [1, 2, 3],                # num_robots
            ["oracle", "learned"],    # find_prob: ground-truth-backed 0.8/0.1, or the learned estimator
            [4000],                   # mcts.iterations
            [1, 2, 4],                # num_objects
        )
    ]


# MaxUCT wants multiplier 1; the FF heuristic with mean backup, 4.
bench_procthor_search.add_cases(_cases(h_mult=1))
bench_procthor_search_ff.add_cases(_cases(h_mult=4))

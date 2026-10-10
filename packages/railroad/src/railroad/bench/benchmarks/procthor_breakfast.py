"""
ProcTHOR Breakfast Benchmark

Robots serve breakfast at a bed in any one of three ways -- a boiled egg in a
bowl, a peeled apple, tomato or potato on a plate, or toast on a plate -- in a
ProcTHOR house where every dish can be made (see
railroad.environment.procthor.breakfast). One goal with several ways to reach
it and dishes whose parts several robots can share.

Registered once per planner configuration, like procthor_search:
`procthor_breakfast` runs the planner's defaults and `procthor_breakfast_ff`
the FF heuristic with mean backup.
"""

import itertools
import time

from rich.console import Console

from railroad.core import get_action_by_name
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad.bench import benchmark, BenchmarkCase
from railroad.bench.benchmarks._helpers import capture_timeout_log


def _served(env) -> str | None:
    """Which dish the final state serves, if any."""
    from railroad.environment.procthor.breakfast import DISHES
    fluents = {str(f).strip("()") for f in env.state.fluents}
    for food_role, done, _, dish_role in DISHES:
        for food in env.roles[food_role]:
            for dish in env.roles[dish_role]:
                if {f"{done} {food}", f"in {food} {dish}", f"at {dish} {env.destination}"} <= fluents:
                    return f"{done} {food} in {dish}"
    return None


def _run_breakfast(case: BenchmarkCase, **planner_kwargs):
    from railroad.environment.procthor.breakfast import BreakfastEnvironment

    env = BreakfastEnvironment(
        case.params["map"], case.params["num_robots"],
        find_prob=case.params.get("find_prob", "learned"),
    )
    goal = env.goal
    max_actions = case.params.get("max_actions", 80)

    recording_console = Console(record=True, force_terminal=True, width=120)

    def fluent_filter(f):
        return any(kw in f.name for kw in
                   ["at", "holding", "found", "searched", "boiled", "peeled", "toasted", "in"])

    dashboard = PlannerDashboard(
        goal, env, fluent_filter=fluent_filter,
        print_on_exit=False, console=recording_console,
    )

    start_time = time.perf_counter()
    with capture_timeout_log(case, dashboard):
        for _ in range(max_actions):
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
            env.act(get_action_by_name(all_actions, action_name))
            dashboard.update(mcts, action_name)

    actions_taken = [name for name, _ in dashboard.actions_taken]
    dashboard.print_history()
    result = {
        "success": goal.evaluate(env.state.fluents),
        "wall_time": time.perf_counter() - start_time,
        "plan_cost": float(env.state.time),
        "actions_count": len(actions_taken),
        "actions": actions_taken,
        "served": _served(env) or "",
        "log_html": recording_console.export_html(inline_styles=True),
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
    name="procthor_breakfast",
    description="Serve breakfast any of three ways in a ProcTHOR house.",
    tags=["multi-agent", "search", "procthor", "alternatives"],
    timeout=900.0,
    repeat=5,
)
def bench_procthor_breakfast(case: BenchmarkCase):
    return _run_breakfast(case)


@benchmark(
    name="procthor_breakfast_ff",
    description="procthor_breakfast planned with the FF heuristic and mean backup.",
    tags=["multi-agent", "search", "procthor", "alternatives"],
    timeout=900.0,
    repeat=5,
)
def bench_procthor_breakfast_ff(case: BenchmarkCase):
    return _run_breakfast(case, heuristic="ff", backup="mean")


def _cases(h_mult: float) -> list[dict]:
    from railroad.environment.procthor.breakfast import MAPS
    return [
        {
            "mcts.iterations": 4000,
            "mcts.c": 400,
            "mcts.h_mult": h_mult,
            "map": map_name,
            "num_robots": num_robots,
            "find_prob": find_prob,
        }
        for map_name, num_robots, find_prob in itertools.product(
            sorted(MAPS), [1, 2, 3], ["oracle", "learned"],
        )
    ]


bench_procthor_breakfast.add_cases(_cases(h_mult=1))
bench_procthor_breakfast_ff.add_cases(_cases(h_mult=4))

"""
Benchmark: Multi-robot planning in ProcTHOR using learned object properties.
"""

import time
import itertools
import random
from functools import reduce
from operator import and_

from railroad.core import Fluent as F, State, get_action_by_name
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.bench import benchmark, BenchmarkCase
from rich.console import Console
from typing import List

SEED_RANGE = (1000, 1050)


def get_goal(objects: List[str], locations: List[str]):
    """Generate diverse goals for ProcTHOR object search and delivery."""
    # single object search
    goal1 = F(f"found {objects[0]}")
    # multi object search
    goal2 = reduce(and_, [F(f"found {obj}") for obj in objects])
    # multi object search and deliver to one location
    goal3 = reduce(and_, [F(f"at {obj} {locations[0]}") for obj in objects])
    # multi object search and deliver to different locations
    goal4 = reduce(and_, [F(f"at {obj} {locations[i % len(locations)]}")
                          for i, obj in enumerate(objects)])
    # object search either objA or objB and deliver to one location
    goal5 = F(f"at {objects[0]} {locations[0]}") | F(f"at {objects[1]} {locations[0]}")
    # multi object search and deliver to different location
    goal6 = reduce(and_, [F(f"at {obj} {loc}") for obj in objects for loc in locations])
    goals = [goal1, goal2, goal3, goal4, goal5, goal6]
    return random.choice(goals)


def bench_procthor_learning_base(case: BenchmarkCase, do_plot: bool = False):
    # Lazy import to avoid loading heavy dependencies at startup
    try:
        from railroad.environment.procthor import ProcTHORScene, ProcTHOREnvironment
        from railroad.environment.procthor.learning.utils import get_default_fcnn_model_path
        from railroad.environment.procthor.resources import get_procthor_10k_dir
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall ProcTHOR dependencies with: pip install railroad[procthor]")
        return

    nn_model_path = get_default_fcnn_model_path()
    if case.use_learning and not nn_model_path.is_file():
        print(f"FCNN model file not found at {nn_model_path}. Skipping non-learned case.")
        return

    procthor_path = get_procthor_10k_dir()
    procthor_scene_cache = procthor_path / f"cache/scene_{case.seed}.pkl"
    if not procthor_scene_cache.is_file():
        raise FileNotFoundError(
            f"ProcTHOR cache not found: {procthor_scene_cache}. To generate, run the following command:\n"
            "uv run python -c 'from railroad.environment.procthor.resources import cache_procthor_scene; "
            "cache_procthor_scene(<seed_to_cache>)'\n"
            "Alternatively, to cache many scenes in a range, run:\n"
            "uv run python -c 'from railroad.environment.procthor.resources import cache_procthor_scene; "
            "cache_procthor_scene(<start_seed>, <end_seed>)'\n"
            "Since benchmarks are parallelized, non-cached scenes will not be loaded to avoid resource hogging."
        )
    scene = ProcTHORScene(seed=case.seed)

    random.seed(case.seed)
    num_objects = 4 if len(scene.objects) >= 4 else len(scene.objects)
    num_locations = 2 if len(scene.locations) >= 2 else 1
    target_objects = random.sample(list(scene.objects), k=num_objects)
    target_locations = random.sample(list(scene.locations.keys()), k=num_locations)

    goal = get_goal(target_objects, target_locations)
    case.goal = goal

    robot_names = [f"robot{i + 1}" for i in range(case.num_robots)]

    if case.use_learning:
        object_find_prob_fn = scene.get_object_find_prob_fn(nn_model_path=str(nn_model_path),
                                                            objects_to_find=target_objects)
    else:
        # If not using learning, use probability of 1.0 for all objects
        def object_find_prob_fn(robot: str, loc: str, obj: str) -> float:
            return 1.0

    # Build operators
    move_op = operators.construct_move_operator_blocking(scene.get_move_cost_fn())
    search_op = operators.construct_search_operator(object_find_prob_fn, 2.0)
    pick_op = operators.construct_pick_operator_blocking(2.0)
    place_op = operators.construct_place_operator_blocking(2.0)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Initial state
    initial_fluents = {F("revealed start_loc")}
    for robot in robot_names:
        initial_fluents.add(F(f"at {robot} start_loc"))
        initial_fluents.add(F(f"free {robot}"))
    initial_state = State(0.0, initial_fluents, [])

    # Create environment
    env = ProcTHOREnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={
            "robot": set(robot_names),
            "location": set(scene.locations.keys()),
            "object": set(target_objects),
        },
        operators=[no_op, pick_op, place_op, move_op, search_op],
    )

    # Planning loop
    max_iterations = 200  # Limit iterations to avoid infinite loops

    # Run planning loop
    start_time = time.perf_counter()

    # Dashboard with recording console
    recording_console = Console(record=True, force_terminal=True, width=120)

    def fluent_filter(f):
        return any(keyword in f.name for keyword in ["at", "holding", "found", "searched"])
    dashboard = PlannerDashboard(goal, env, fluent_filter=fluent_filter, print_on_exit=False, console=recording_console)

    for iteration in range(max_iterations):
        # Check if goal is reached
        if goal.evaluate(env.state.fluents):
            break

        # Get available actions
        all_actions = env.get_actions()

        # Plan next action
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(env.state, goal,
                           max_iterations=case.mcts.iterations,
                           c=case.mcts.c,
                           max_depth=20,
                           heuristic_multiplier=case.mcts.h_mult)

        if action_name == 'NONE':
            dashboard.console.print("No more actions available. Goal may not be achievable.")
            break

        # Execute action
        action = get_action_by_name(all_actions, action_name)
        env.act(action)
        dashboard.update(mcts, action_name)

    # Export the recorded console output as HTML
    actions_taken = [name for name, _ in dashboard.actions_taken]
    dashboard.print_history()
    html_output = recording_console.export_html(inline_styles=True)

    result = {
        "success": goal.evaluate(env.state.fluents),
        "wall_time": time.perf_counter() - start_time,
        "plan_cost": float(env.state.time),
        "actions_count": len(actions_taken),
        "actions": actions_taken,
        "log_html": html_output,  # Will be logged as HTML artifact
    }

    if do_plot:
        plot_image = dashboard.get_plot_image()
        if plot_image is not None:
            result["log_plot"] = plot_image

    return result


@benchmark(
    name="procthor_learning",
    description="Multi-robot planning in ProcTHOR using learned object properties.",
    tags=["procthor", "learning", "multi-agent"],
    timeout=180.0,
)
def bench_procthor_learning(case: BenchmarkCase):
    return bench_procthor_learning_base(case, do_plot=True)


bench_procthor_learning.add_cases([
    {
        "mcts.iterations": iterations,
        "mcts.c": c,
        "mcts.h_mult": h_mult,
        "num_robots": num_robots,
        "seed": seed,
        "use_learning": use_learning,
    }
    for c, num_robots, h_mult, iterations, use_learning, seed in itertools.product(
        [300],                      # mcts.c
        [1, 2],                     # num_robots
        [2],                        # mcts.h_mult
        [10000],                    # mcts.iterations
        [True, False],              # use_learning
        list(range(*SEED_RANGE))    # seed
    )
])

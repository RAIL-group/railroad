"""
Script: Multi-robot planning in ProcTHOR with learned or optimistic object properties.

Usage:
    uv run python scripts/procthor_planning.py --num_robots 2 --learned --seed 1000 --save_dir ./results
    uv run python scripts/procthor_planning.py --num_robots 1 --optimistic --seed 1005 --save_dir ./results
"""

import argparse
import time
import random
from functools import reduce
from operator import and_
from pathlib import Path
from typing import List

from railroad.core import Fluent as F, State, get_action_by_name
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.environment.procthor import ProcTHORScene, ProcTHOREnvironment
from railroad.environment.procthor.learning.utils import get_default_fcnn_model_path
from rich.console import Console


def get_goal(objects: List[str], locations: List[str]):
    """Generate diverse goals for ProcTHOR object search and delivery."""
    goal1 = F(f"found {objects[0]}")
    goal2 = reduce(and_, [F(f"found {obj}") for obj in objects])
    goal3 = reduce(and_, [F(f"at {obj} {locations[0]}") for obj in objects])
    goal4 = reduce(and_, [F(f"at {obj} {locations[i % len(locations)]}")
                          for i, obj in enumerate(objects)])
    goal5 = F(f"at {objects[0]} {locations[0]}") | F(f"at {objects[1]} {locations[0]}")
    goal6 = reduce(and_, [F(f"at {obj} {loc}") for obj in objects for loc in locations])
    goals = [goal1, goal2, goal3, goal4, goal5, goal6]
    return random.choice(goals)


def main(
    seed: int,
    num_robots: int,
    learned: bool,
    save_dir: str,
) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    mode = "learned" if learned else "optimistic"
    basename = f"procthor_{mode}_num_robots_{num_robots}_{seed}"

    nn_model_path = get_default_fcnn_model_path()
    if learned and not nn_model_path.is_file():
        raise FileNotFoundError(f"FCNN model file not found at {nn_model_path}")

    scene = ProcTHORScene(seed=seed)

    random.seed(seed)
    num_objects = 4 if len(scene.objects) >= 4 else len(scene.objects)
    num_locations = 2 if len(scene.locations) >= 2 else 1
    target_objects = random.sample(list(scene.objects), k=num_objects)
    target_locations = random.sample(list(scene.locations.keys()), k=num_locations)

    goal = get_goal(target_objects, target_locations)

    robot_names = [f"robot{i + 1}" for i in range(num_robots)]

    if learned:
        object_find_prob_fn = scene.get_object_find_prob_fn(
            nn_model_path=str(nn_model_path),
            objects_to_find=target_objects,
        )
    else:
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
    max_iterations = 200
    start_time = time.perf_counter()

    recording_console = Console(record=True, force_terminal=True, width=120)

    def fluent_filter(f):
        return any(keyword in f.name for keyword in ["at", "holding", "found", "searched"])

    dashboard = PlannerDashboard(
        goal, env, fluent_filter=fluent_filter,
        print_on_exit=False, console=recording_console,
    )

    for iteration in range(max_iterations):
        if goal.evaluate(env.state.fluents):
            break

        all_actions = env.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(
            env.state, goal,
            max_iterations=10000,
            c=300,
            max_depth=20,
            heuristic_multiplier=2.0,
        )

        if action_name == "NONE":
            dashboard.console.print("No more actions available. Goal may not be achievable.")
            break

        action = get_action_by_name(all_actions, action_name)
        env.act(action)
        dashboard.update(mcts, action_name)

    actions_taken = [name for name, _ in dashboard.actions_taken]
    dashboard.print_history()

    wall_time = time.perf_counter() - start_time
    success = goal.evaluate(env.state.fluents)
    plan_cost = float(env.state.time)

    # Save cost file
    cost_file = save_path / f"cost_{basename}.txt"
    with open(cost_file, "w") as f:
        f.write(f"success: {success}\n")
        f.write(f"wall_time: {wall_time:.2f}\n")
        f.write(f"plan_cost: {plan_cost:.2f}\n")
        f.write(f"actions_count: {len(actions_taken)}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"num_robots: {num_robots}\n")
        f.write(f"mode: {mode}\n")
        f.write(f"goal: {goal}\n")
        f.write(f"actions: {actions_taken}\n")
    print(f"Saved cost to {cost_file}")

    # Save plot image
    img_file = save_path / f"img_{basename}.png"
    dashboard.show_plots(save_plot=str(img_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-robot planning in ProcTHOR",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-robots", type=int, required=True)
    parser.add_argument("--use-learning", action="store_true", default=False,
                        help="Use learned object properties (default: optimistic with prob=1.0)")
    parser.add_argument("--save-dir", type=str, required=True)
    args = parser.parse_args()

    main(
        seed=args.seed,
        num_robots=args.num_robots,
        learned=args.use_learning,
        save_dir=args.save_dir,
    )

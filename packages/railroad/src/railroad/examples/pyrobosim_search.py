"""PyRoboSim multi-robot search example.

Demonstrates using PyRoboSim environment with MCTS planning
for multi-robot object search and retrieval, following the same
API structure as ProcTHOR examples.
"""

def sample_objects_and_location(scene, num_objects: int, seed: int | None = None):
    import random

    if seed:
        random.seed(seed)
    all_objects = sorted(list(scene.objects))
    all_locations = sorted(list(scene.object_locations.keys()))

    return (
        random.sample(all_objects, k=min(num_objects, len(all_objects))),
        random.choice(all_locations),
    )


def main(
    world_file: str | None = None,
    seed: int | None = None,
    num_objects: int = 2,
    num_robots: int = 2,
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    video_fps: int = 60,
    video_dpi: int = 150,
) -> None:
    """Run PyRoboSim multi-robot search example."""
    from railroad.environment.pyrobosim import PyRoboSimScene, PyRoboSimEnvironment, get_default_pyrobosim_world_file_path
    from railroad import operators
    from railroad.core import Fluent as F, get_action_by_name
    from railroad.dashboard import PlannerDashboard
    from railroad.planner import MCTSPlanner
    from railroad._bindings import State
    from pathlib import Path
    from functools import reduce
    from operator import and_

    # 1. Configuration & Scene Setup
    if world_file is None:
        world_file = str(get_default_pyrobosim_world_file_path())

    print(f"Loading PyRoboSim scene: {world_file}")
    scene = PyRoboSimScene(world_file)

    if seed is None:
        target_objects = ["apple0", "banana0"]
        target_location = "counter0"
    else:
        target_objects, target_location = sample_objects_and_location(scene, num_objects=num_objects, seed=seed)

    robot_names = [r.name for r in scene.world.robots][:num_robots]
    if len(robot_names) < num_robots:
        num_robots = len(robot_names)

    # 2. Build operators using scene info
    move_cost_fn = scene.get_move_cost_fn()

    def object_find_prob_fn(robot: str, location: str, obj: str) -> float:
        # Simple probability based on ground truth
        objs_at_loc = scene.object_locations.get(location, set())
        return 0.8 if obj in objs_at_loc else 0.1

    move_op = operators.construct_move_operator_blocking(move_cost_fn)
    search_op = operators.construct_search_operator(object_find_prob_fn, 5.0)
    pick_op = operators.construct_pick_operator_blocking(5.0)
    place_op = operators.construct_place_operator_blocking(5.0)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # 3. Define initial state
    initial_fluents = set()
    for robot in robot_names:
        robot_loc = f"{robot}_loc"
        initial_fluents.add(F(f"at {robot} {robot_loc}"))
        initial_fluents.add(F(f"free {robot}"))
        initial_fluents.add(F(f"revealed {robot_loc}"))
    initial_state = State(0.0, initial_fluents, [])

    # 4. Define goal: place target objects at target location
    goal = reduce(and_, [F(f"at {obj} {target_location}") & F(f"found {obj}")
                         for obj in target_objects])

    # 5. Create environment
    env = PyRoboSimEnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={
            "robot": set(robot_names),
            "location": set(scene.locations.keys()),
            "object": set(target_objects),
        },
        operators=[no_op, pick_op, place_op, move_op, search_op],
        show_plot=show_plot,
        record_plots=save_video is not None,
    )

    # 6. Planning loop
    max_iterations = 60

    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "holding", "found", "searched"])

    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Goal reached![/green]")
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(
                env.state,
                goal,
                max_iterations=10000,
                c=300,
                max_depth=20,
                heuristic_multiplier=2,
            )

            if action_name == "NONE":
                dashboard.console.print("No more actions available.")
                break

            action = get_action_by_name(all_actions, action_name)
            env.act(action)
            dashboard.update(mcts, action_name)

    # 7. Post-execution (Saving video/plots)
    if save_video and env.canvas:
        save_path = Path(save_video)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        env.canvas.save_animation(filepath=str(save_path))

    if env.canvas:
        env.canvas.wait_for_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-file", type=str, help="Path to the world YAML file.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--num-objects", type=int, default=2, help="Number of objects to search.")
    parser.add_argument("--num-robots", type=int, default=2, help="Number of robots.")
    parser.add_argument("--show-plot", action='store_true', help="Whether to show the plot window.")
    parser.add_argument("--save-video", type=str, help="Path to save video.")
    args = parser.parse_args()

    main(
        world_file=args.world_file,
        seed=args.seed,
        num_objects=args.num_objects,
        num_robots=args.num_robots,
        show_plot=args.show_plot,
        save_video=args.save_video,
    )

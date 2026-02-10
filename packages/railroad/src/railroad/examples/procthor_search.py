"""ProcTHOR multi-robot search example.

Demonstrates using ProcTHOR environment with MCTS planning
for multi-robot object search and retrieval.
"""


def sample_objects_and_location(scene, num_objects: int, seed: int | None = None):
    import random

    if seed:
        random.seed(seed)
    all_objects = sorted({
        obj
        for objs in scene.object_locations.values()
        for obj in objs
    })

    all_locations = sorted(scene.object_locations.keys())

    return (
        random.sample(list(all_objects), k=min(num_objects, len(all_objects))),
        random.choice(all_locations),
    )


def main(
    seed: int | None = None,
    num_objects: int = 2,
    num_robots: int = 2,
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    estimate_object_find_prob: bool = False,
    nn_model_path: str | None = None,
    video_fps: int = 60,
    video_dpi: int = 150,
) -> None:
    """Run ProcTHOR multi-robot search example."""
    # Lazy import to avoid loading heavy dependencies at startup
    try:
        from railroad.environment.procthor import ProcTHORScene, ProcTHOREnvironment
        from railroad.environment.procthor.learning.utils import get_default_fcnn_model_path
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall ProcTHOR dependencies with: pip install railroad[procthor]")
        return

    from railroad import operators
    from railroad.core import Fluent as F, get_action_by_name
    from railroad.dashboard import PlannerDashboard
    from railroad.planner import MCTSPlanner
    from railroad._bindings import State
    from pathlib import Path

    # Configuration
    robot_names = [f"robot{i + 1}" for i in range(num_robots)]

    if seed is None:
        # Use hardcoded defaults
        scene_seed = 4001
        print(f"Loading ProcTHOR scene (seed={scene_seed})...")
        scene = ProcTHORScene(seed=scene_seed)
        target_objects = ["teddybear_6", "pencil_17"]
        target_location = "garbagecan_5"
    else:
        scene_seed = seed
        print(f"Loading ProcTHOR scene (seed={scene_seed})...")
        scene = ProcTHORScene(seed=scene_seed)
        target_objects, target_location = sample_objects_and_location(scene, num_objects=num_objects, seed=seed)

    # Build operators
    move_cost_fn = scene.get_move_cost_fn()

    # If estimate_object_find_prob is True, use learned model to get object find probabilities

    if estimate_object_find_prob:
        model_path = Path(nn_model_path) if nn_model_path is not None else get_default_fcnn_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained neural network model not found at {model_path} to estimate object find probabilities. "
                "Please provide a valid path or omit the --estimate-object-find-prob flag."
            )
        object_find_prob_fn = scene.get_object_find_prob_fn(nn_model_path=str(model_path),
                                                            objects_to_find=target_objects)
    else:
        # Otherwise, create probability function based on ground truth
        def object_find_prob_fn(robot: str, location: str, obj: str) -> float:
            for loc, objs in scene.object_locations.items():
                if obj in objs:
                    return 0.8 if loc == location else 0.1
            return 0.1

    move_op = operators.construct_move_operator_blocking(move_cost_fn)
    search_op = operators.construct_search_operator(object_find_prob_fn, 10.0)
    pick_op = operators.construct_pick_operator_blocking(10.0)
    place_op = operators.construct_place_operator_blocking(10.0)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Initial state
    initial_fluents = {F("revealed start_loc")}
    for robot in robot_names:
        initial_fluents.add(F(f"at {robot} start_loc"))
        initial_fluents.add(F(f"free {robot}"))
    initial_state = State(0.0, initial_fluents, [])

    # Goal: place both objects at target location
    from functools import reduce
    from operator import and_
    goal = reduce(and_, [F(f"at {obj} {target_location}") & F(f"found {obj}")
                         for obj in target_objects])

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

    dashboard.show_plots(
        save_plot=save_plot, show_plot=show_plot, save_video=save_video,
        video_fps=video_fps, video_dpi=video_dpi,
    )


if __name__ == "__main__":
    main()

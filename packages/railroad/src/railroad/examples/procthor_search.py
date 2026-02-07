"""ProcTHOR multi-robot search example.

Demonstrates using ProcTHOR environment with MCTS planning
for multi-robot object search and retrieval.
"""

from pathlib import Path


def main() -> None:
    """Run ProcTHOR multi-robot search example."""
    # Lazy import to avoid loading heavy dependencies at startup
    try:
        from railroad.environment.procthor import ProcTHORScene, ProcTHOREnvironment
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall ProcTHOR dependencies with: pip install railroad[procthor]")
        return

    import matplotlib
    matplotlib.use('Agg')  # Use headless backend for file output
    import matplotlib.pyplot as plt
    from railroad import operators
    from railroad.core import Fluent as F, get_action_by_name
    from railroad.dashboard import PlannerDashboard
    from railroad.planner import MCTSPlanner
    from railroad._bindings import State
    from railroad.environment.procthor.plotting import (
        extract_robot_poses,
        plot_multi_robot_trajectories,
    )

    # Configuration
    seed = 4001
    robot_names = ["robot1", "robot2"]
    target_objects = ["teddybear_6", "pencil_17"]
    target_location = "garbagecan_5"
    save_dir = Path("./data/test_logs")

    print(f"Loading ProcTHOR scene (seed={seed})...")
    scene = ProcTHORScene(seed=seed)

    # Build operators
    move_cost_fn = scene.get_move_cost_fn()
    # All skill time functions take (robot, location, object) -> float
    def search_time_fn(r, loc, o):
        return 15.0 if r == "robot1" else 10.0

    def pick_time_fn(r, loc, o):
        return 15.0 if r == "robot1" else 10.0

    def place_time_fn(r, loc, o):
        return 15.0 if r == "robot1" else 10.0

    # Create probability function based on ground truth
    def object_find_prob(robot: str, location: str, obj: str) -> float:
        for loc, objs in scene.object_locations.items():
            if obj in objs:
                return 0.8 if loc == location else 0.1
        return 0.1

    move_op = operators.construct_move_operator_blocking(move_cost_fn)
    search_op = operators.construct_search_operator(object_find_prob, search_time_fn)
    pick_op = operators.construct_pick_operator_blocking(pick_time_fn)
    place_op = operators.construct_place_operator_blocking(place_time_fn)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Initial state
    initial_fluents = {
        F("revealed start_loc"),
        F("at robot1 start_loc"),
        F("free robot1"),
        F("at robot2 start_loc"),
        F("free robot2"),
    }
    initial_state = State(0.0, initial_fluents, [])

    # Goal: place both objects at target location
    goal = F(f"at {target_objects[0]} {target_location}") & F(
        f"at {target_objects[1]} {target_location}"
    )

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
    actions_taken: list[str] = []

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
            actions_taken.append(action_name)
            dashboard.update(mcts, action_name)

    # Plot results
    robot_locations = {name: "start_loc" for name in robot_names}
    robot_poses = extract_robot_poses(actions_taken, robot_locations, scene.locations)

    plt.figure(figsize=(16, 8))

    # Left panel: top-down view
    ax1 = plt.subplot(1, 2, 1)
    top_down_image = scene.get_top_down_image(orthographic=True)
    ax1.imshow(top_down_image)
    ax1.axis("off")
    ax1.set_title("Top-down View")

    # Right panel: trajectory plot
    ax2 = plt.subplot(1, 2, 2)
    plot_multi_robot_trajectories(ax2, scene.grid, robot_poses, scene.scene_graph)
    ax2.set_title(f"Multi-Robot Trajectories (Total time: {env.state.time:.1f}s)")

    # Save figure
    figpath = Path(save_dir) / f'procthor_run_{seed}.png'
    figpath.parent.mkdir(parents=True, exist_ok=True)
    figpath_str = figpath if figpath.as_posix().startswith(("/", "./", "../")) else f"./{figpath}"
    plt.savefig(figpath, dpi=300)
    dashboard.console.print(f"\nSaved plot to [yellow]{figpath_str}[/yellow]")


if __name__ == "__main__":
    main()

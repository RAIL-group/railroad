from railroad.environment.pyrobosim import (
    DecoupledPyRoboSimEnvironment,
    PyRoboSimScene,
    get_default_pyrobosim_world_file_path,
)
from railroad import operators
from railroad.planner import MCTSPlanner
from railroad.core import Fluent as F, get_action_by_name, State
from railroad.dashboard import PlannerDashboard
import argparse
import logging
import time


# Fixed operator times for non-move skills
SEARCH_TIME = 1.0
PICK_TIME = 1.0
PLACE_TIME = 1.0


def main(args):
    # Goal: move apple and banana to counter
    goal = F("at apple0 counter0") & F("at banana0 counter0")

    # 1. Initialize Scene
    scene = PyRoboSimScene(args.world_file)

    # 2. Create operators using scene info
    move_time_fn = scene.get_move_cost_fn()

    def object_find_prob(r, loc, o):
        return 0.8 if loc == "my_desk" and o == "apple0" else 0.2

    move_op = operators.construct_move_operator_blocking(move_time_fn)
    search_op = operators.construct_search_operator(object_find_prob, SEARCH_TIME)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)
    ops = [no_op, pick_op, place_op, move_op, search_op]

    # 3. Create the initial state fluents
    robot_names = [r.name for r in scene.world.robots]
    initial_fluents = set()
    for robot in robot_names:
        robot_loc = f"{robot}_loc"
        initial_fluents.add(F("at", robot, robot_loc))
        initial_fluents.add(F("free", robot))
        initial_fluents.add(F("revealed", robot_loc))
    state = State(0.0, initial_fluents)

    # 4. Create the DECOUPLED PyRoboSim environment
    # This will spawn the simulator in a separate process
    env = DecoupledPyRoboSimEnvironment(
        scene=scene,
        state=state,
        objects_by_type={
            "robot": set(robot_names),
            "location": set(scene.locations.keys()),
            "object": scene.objects,
        },
        operators=ops,
        show_plot=args.show_plot,
        record_plots=not args.no_video,
    )

    # Planning loop
    max_iterations = 60  # Limit iterations to avoid infinite loops

    def fluent_filter(f):
        return any(
            keyword in f.name for keyword in ["at", "holding", "found", "searched"]
        )

    try:
        with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
            for iteration in range(max_iterations):
                # Check if goal is reached
                if goal.evaluate(env.state.fluents):
                    dashboard.console.print("[bold green]Goal reached![/bold green]")
                    break

                # Get available actions
                all_actions = env.get_actions()
                # Plan next action
                mcts = MCTSPlanner(all_actions)
                # In decoupled mode, the planner runs concurrently with simulator
                action_name = mcts(
                    env.state,
                    goal,
                    max_iterations=10000,
                    c=300,
                    max_depth=20,
                    heuristic_multiplier=3,
                )

                if action_name == "NONE":
                    dashboard.console.print(
                        "No more actions available. Goal may not be achievable."
                    )
                    break

                # Execute action
                # Decoupled env.act() forwards to simulator process and waits for acknowledgment
                action = get_action_by_name(all_actions, action_name)
                env.act(action, do_interrupt=False)
                dashboard.update(mcts, action_name)

            if not goal.evaluate(env.state.fluents):
                dashboard.console.print("[bold red]Failed to reach goal within max iterations.[/bold red]")

            # Keep dashboard visible for a moment
            # time.sleep(2.0)

        if not args.no_video:
            from pathlib import Path

            Path("./data").mkdir(parents=True, exist_ok=True)
            output_path = "./data/pyrobosim_decoupled_demo.mp4"
            print(f"Saving animation to {output_path}...")
            env.save_animation(filepath=output_path)

    finally:
        # Cleanup is handled by DecoupledPyRoboSimEnvironment.__del__
        # but we can be explicit if we want.
        if hasattr(env, "_bridge"):
            env._bridge.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo of planning with Decoupled PyRoboSim Environment."
    )
    parser.add_argument(
        "--world-file",
        type=str,
        default=str(get_default_pyrobosim_world_file_path()),
        help="Path to the world YAML file.",
    )
    parser.add_argument(
        "--show-plot", action="store_true", help="Whether to show the plot window."
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Whether to disable generating video of the simulation.",
    )
    args = parser.parse_args()

    # Turn off all logging of level INFO and below (to suppress pyrobosim logs)
    logging.disable(logging.INFO)

    main(args)

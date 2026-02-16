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
PICK_TIME = 1.0
PLACE_TIME = 1.0


def main(args):
    # Goal: move apple and banana to counter
    goal = F("at apple0 counter0") & F("at banana0 counter0")

    # 1. Initialize Scene
    scene = PyRoboSimScene(args.world_file)

    # 2. Create operators using scene info
    def get_move_cost_fn():
        # Override scene's move cost to make it slower for better concurrency visibility
        scene_fn = scene.get_move_cost_fn()
        def slower_move_time(r, f, t):
            return scene_fn(r, f, t) * 2.0 # Half speed
        return slower_move_time

    move_time_fn = get_move_cost_fn()

    move_op = operators.construct_move_operator_blocking(move_time_fn)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Removed search_op
    ops = [no_op, pick_op, place_op, move_op]

    # 3. Create the initial state fluents
    robot_names = ["robot1", "robot2"]
    initial_fluents = set()

    # Reveal all locations and objects
    for loc_name in scene.locations.keys():
        initial_fluents.add(F("revealed", loc_name))

    for loc_name, objects in scene.object_locations.items():
        for obj_name in objects:
            initial_fluents.add(F("found", obj_name))
            initial_fluents.add(F("at", obj_name, loc_name))

    for robot in robot_names:
        robot_loc = f"{robot}_loc"
        initial_fluents.add(F("at", robot, robot_loc))
        initial_fluents.add(F("free", robot))
        initial_fluents.add(F("revealed", robot_loc))

    state = State(0.0, initial_fluents)

    # 4. Create the DECOUPLED PyRoboSim environment
    env = DecoupledPyRoboSimEnvironment(
        scene=scene,
        state=state,
        objects_by_type={
            "robot": set(robot_names),
            "location": set(scene.locations.keys()) | {f"{r}_loc" for r in robot_names},
            "object": scene.objects,
        },
        operators=ops,
        show_plot=args.show_plot,
        record_plots=not args.no_video,
    )

    # Planning loop
    max_iterations = 60

    def fluent_filter(f):
        return any(
            keyword in f.name for keyword in ["at", "holding", "found"]
        )

    try:
        with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
            for iteration in range(max_iterations):
                # Check if goal is reached
                if goal.evaluate(env.state.fluents):
                    dashboard.console.print(
                        "[bold green]Goal reached symbolically! Finalizing physical actions...[/bold green]"
                    )
                    break

                # Get available actions
                all_actions = env.get_actions()

                # Diagnostic: show which robots are busy
                busy_robots = [r for r in robot_names if F("free", r) not in env.fluents]
                if busy_robots:
                    dashboard.console.print(f"Busy robots: {', '.join(busy_robots)}")

                # Plan next action
                mcts = MCTSPlanner(all_actions)
                # In fully revealed mode, planning should be very fast
                t_plan_start = time.time()
                action_name = mcts(
                    env.state,
                    goal,
                    max_iterations=10000, # Lower iterations needed for revealed world
                    c=300,
                    max_depth=20,
                    heuristic_multiplier=2,
                )
                t_plan_end = time.time()
                dashboard.console.print(f"Planning took {t_plan_end - t_plan_start:.3f}s")

                if action_name == "NONE":
                    # If no robot is free, wait a bit for actions to finish
                    if busy_robots:
                        time.sleep(0.5)
                        continue
                    else:
                        dashboard.console.print(
                            "No more actions available and all robots idle. Goal may not be achievable."
                        )
                        break

                # Execute action
                action = get_action_by_name(all_actions, action_name)

                # Keep dashboard responsive during physical execution
                last_dash_update = [0.0]
                def dash_callback():
                    now = time.time()
                    if now - last_dash_update[0] > 0.1:
                        dashboard.refresh()
                        last_dash_update[0] = now

                env.act(action, loop_callback_fn=dash_callback, do_interrupt=False)
                dashboard.update(mcts, action_name)

            # Final Wait
            dashboard.console.print("[bold yellow]Waiting for all robots to become idle...[/bold yellow]")
            start_final_wait = time.time()
            while any(F("free", r) not in env.fluents for r in robot_names) and time.time() - start_final_wait < 30.0:
                _ = env.state
                dashboard.refresh()
                time.sleep(0.1)

            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[bold green]Mission Complete![/bold green]")
            else:
                dashboard.console.print(
                    "[bold red]Failed to reach goal within max iterations.[/bold red]"
                )

        if not args.no_video:
            from pathlib import Path
            Path("./data").mkdir(parents=True, exist_ok=True)
            output_path = "./data/pyrobosim_revealed_demo.mp4"
            print(f"Saving animation to {output_path}...")
            env.save_animation(filepath=output_path)

    finally:
        # Cleanup
        if hasattr(env, "_client"):
            env._client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo of planning with Decoupled PyRoboSim Environment (Fully Revealed)."
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

    # Turn off all logging of level INFO and below
    logging.disable(logging.INFO)

    main(args)

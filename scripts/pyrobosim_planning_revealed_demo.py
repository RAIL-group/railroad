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
    scene = PyRoboSimScene(args.world_file)

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

    move_time_fn = scene.get_move_cost_fn()
    move_op = operators.construct_move_operator_blocking(move_time_fn)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    ops = [no_op, pick_op, place_op, move_op]

    goal = F("at apple0 counter0") & F("at banana0 counter0")
    state = State(0.0, initial_fluents)

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

    max_iterations = 30

    def fluent_filter(f):
        return any(keyword in f.name for keyword in ["at", "holding", "found"])

    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        for iteration in range(max_iterations):
            # Check if goal is reached
            if goal.evaluate(env.state.fluents):
                dashboard.console.print(
                    "[bold green]Goal reached![/bold green]"
                )
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(
                env.state,
                goal,
                max_iterations=8000,
                c=300,
                max_depth=20,
                heuristic_multiplier=2,
            )

            if action_name == "NONE":
                dashboard.console.print("No actions available.")
                break

            action = get_action_by_name(all_actions, action_name)

            env.act(action, do_interrupt=False)
            dashboard.update(mcts, action_name)

    if not args.no_video:
        from pathlib import Path
        Path("./data").mkdir(parents=True, exist_ok=True)
        output_path = "./data/pyrobosim_revealed_demo.mp4"
        print(f"Saving animation to {output_path}...")
        env.save_animation(filepath=output_path)


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

from railroad.environment.pyrobosim import PyRoboSimEnvironment, PyRoboSimScene, get_default_pyrobosim_world_file_path
from railroad import operators
from railroad.planner import MCTSPlanner
from railroad.core import Fluent as F, get_action_by_name, State
from railroad.dashboard import PlannerDashboard
import argparse
import logging


# Fixed operator times for non-move skills
SEARCH_TIME = 1.0
PICK_TIME = 1.0
PLACE_TIME = 1.0


def main(args):
    goal = F("at apple0 counter0") & F("at banana0 counter0")

    scene = PyRoboSimScene(
        args.world_file,
        show_plot=args.show_plot,
        record_plots=not args.no_video,
    )

    move_time_fn = scene.get_move_cost_fn()

    def object_find_prob(r, loc, o):
        return 0.8 if loc == 'my_desk' and o == 'apple0' else 0.2

    move_op = operators.construct_move_operator_blocking(move_time_fn)
    search_op = operators.construct_search_operator(object_find_prob, SEARCH_TIME)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)
    ops = [no_op, pick_op, place_op, move_op, search_op]

    robot_names = scene.robot_names
    initial_fluents = set()
    for robot in robot_names:
        robot_loc = f"{robot}_loc"
        initial_fluents.add(F("at", robot, robot_loc))
        initial_fluents.add(F("free", robot))
        initial_fluents.add(F("revealed", robot_loc))

    env = PyRoboSimEnvironment(
        scene=scene,
        state=State(0.0, initial_fluents),
        objects_by_type={
            "robot": set(robot_names),
            "location": set(scene.locations.keys()),
            "object": scene.objects,
        },
        operators=ops,
    )

    def fluent_filter(f):
        return any(keyword in f.name for keyword in ["at", "holding", "found", "searched"])

    dashboard = PlannerDashboard(goal, env, fluent_filter=fluent_filter)

    for iteration in range(60):
        if goal.evaluate(env.state.fluents):
            break

        all_actions = env.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(env.state, goal, max_iterations=10000, c=300, max_depth=20, heuristic_multiplier=3)

        if action_name == 'NONE':
            dashboard.console.print("No more actions available. Goal may not be achievable.")
            break

        action = get_action_by_name(all_actions, action_name)
        env.act(action, do_interrupt=False)
        dashboard.update(mcts, action_name)

    if not args.no_video:
        from pathlib import Path
        Path("./data").mkdir(parents=True, exist_ok=True)
        env.save_animation(filepath='./data/pyrobosim_planning_demo.mp4')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--world-file", type=str, default=str(get_default_pyrobosim_world_file_path()),
                      help="Path to the world YAML file.")
    args.add_argument("--show-plot", action='store_true', help="Whether to show the plot window.")
    args.add_argument("--no-video", action='store_true', help="Whether to disable generating video of the simulation.")
    args = args.parse_args()

    logging.disable(logging.INFO)
    main(args)

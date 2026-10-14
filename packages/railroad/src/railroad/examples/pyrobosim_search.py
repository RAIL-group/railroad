"""PyRoboSim multi-robot fetch example.

Two robots search for an apple and a banana and carry them to the counter.
Planning is as in the other examples; the environment's actions run on the
PyRoboSim robots (``PyRoboSimEnvironment``). The planner never sees ground
truth: object locations are learned from what the robots detect when they
search.
"""

from __future__ import annotations

# Fixed skill times (simulated seconds), for planning and execution alike
SEARCH_TIME = 1.0
PICK_TIME = 1.0
PLACE_TIME = 1.0


def main(
    world_file: str | None = None,
    seed: int | None = 1,
    realtime_factor: float = 1.0,
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    video_fps: int = 60,
    video_dpi: int = 150,
    video_time: float | str | None = None,
) -> None:
    """Run the PyRoboSim fetch example.

    ``seed`` fixes the world's random object layout (None: random; some
    layouts leave no room to place objects on the counter).
    ``show_plot`` shows the simulator live and ``save_video`` records it
    (the robots as they actually moved), in place of the dashboard's
    trajectory plot and video; ``save_plot`` still saves the dashboard's
    plot. ``video_fps``/``video_dpi``/``video_time`` apply only to the
    dashboard's video, so they are unused here.
    """
    import logging

    from railroad.environment.pyrobosim import (
        DEFAULT_WORLD_FILE,
        PyRoboSimEnvironment,
        PyRoboSimScene,
    )

    from railroad import operators
    from railroad.core import Fluent as F, get_action_by_name
    from railroad.dashboard import PlannerDashboard
    from railroad.planner import MCTSPlanner

    # Turn off all logging of level INFO and below (to suppress pyrobosim logs)
    logging.disable(logging.INFO)

    scene = PyRoboSimScene(world_file or DEFAULT_WORLD_FILE, seed=seed)
    goal = F("at apple0 counter0") & F("at banana0 counter0")

    def object_find_prob(robot, loc, obj):
        return 0.8 if loc == "my_desk" and obj == "apple0" else 0.2

    move_op = operators.construct_move_operator_blocking(scene.move_time)
    search_op = operators.construct_search_operator(object_find_prob, SEARCH_TIME)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    class FetchEnvironment(PyRoboSimEnvironment):
        """Supply this problem's operators the way the API prefers: a subclass."""

        def define_operators(self):
            return [no_op, move_op, search_op, pick_op, place_op]

    env = FetchEnvironment(
        scene,
        realtime_factor=realtime_factor,
        skill_durations={"search": SEARCH_TIME, "pick": PICK_TIME, "place": PLACE_TIME},
        show_plot=show_plot,
        save_video=save_video,
    )

    # Planning loop
    max_iterations = 60

    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "holding", "found", "searched"])
    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        for iteration in range(max_iterations):
            # Check if goal is reached
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Goal achieved![/green]")
                break

            # Get available actions
            all_actions = env.get_actions()

            # Plan next action
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(env.state, goal, max_iterations=10000, c=300, max_depth=20, heuristic_multiplier=3)

            if action_name == "NONE":
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            # Execute action (on the PyRoboSim robots)
            action = get_action_by_name(all_actions, action_name)
            env.act(action)
            dashboard.update(mcts, action_name)

    location_coords = {name: (float(c[0]), float(c[1])) for name, c in scene.locations.items()}
    dashboard.show_plots(
        save_plot=save_plot, location_coords=location_coords,
    )
    env.close()  # let the robots finish; save the simulator video / keep its window open

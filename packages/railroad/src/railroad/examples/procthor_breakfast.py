"""ProcTHOR breakfast example.

Robots serve breakfast at a bed in any one of three ways (a boiled egg in a
bowl, peeled produce on a plate, toast on a plate) in a ProcTHOR house; see
railroad.environment.procthor.breakfast.
"""

from __future__ import annotations


def main(
    map_name: str = "8614",
    num_robots: int = 2,
    estimate_object_find_prob: bool = False,
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    video_fps: int = 60,
    video_dpi: int = 150,
    video_time: float | str | None = None,
) -> None:
    """Run the breakfast task on one map."""
    from railroad.core import get_action_by_name
    from railroad.dashboard import PlannerDashboard
    from railroad.environment.procthor.breakfast import BreakfastEnvironment, map_summary
    from railroad.planner import MCTSPlanner

    env = BreakfastEnvironment(
        map_name, num_robots, find_prob="learned" if estimate_object_find_prob else "oracle",
    )
    goal = env.goal
    print(f"Breakfast at {env.destination}; where things really are:")
    for role, places in map_summary(env).items():
        print(f"  {role}: {', '.join(places)}")

    def fluent_filter(f):
        return any(kw in f.name for kw in
                   ["at", "holding", "found", "searched", "boiled", "peeled", "toasted", "in"])

    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        for _ in range(80):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Breakfast is served![/green]")
                break
            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(env.state, goal, max_iterations=4000, c=400, max_depth=20)
            if action_name == "NONE":
                dashboard.console.print("No more actions available.")
                break
            env.act(get_action_by_name(all_actions, action_name))
            dashboard.update(mcts, action_name)

    dashboard.show_plots(
        save_plot=save_plot, show_plot=show_plot, save_video=save_video,
        video_fps=video_fps, video_dpi=video_dpi, video_time=video_time,
    )


if __name__ == "__main__":
    main()

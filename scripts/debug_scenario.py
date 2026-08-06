import time
from functools import partial

from interruption.dashboard_adapters import AstarDashboardPlanner
from interruption.environments import (
    construct_procthor_kitchen_environment,
    construct_simple_kitchen_environment,
    get_example_procthor_goal,
    get_example_procthor_task_distribution,
    get_simple_goal,
    get_simple_task_distribution,
)
from interruption.planner import astar_search
from interruption.utilities import (
    DistributionType,
    RandomVariableType,
    get_action_cost,
    get_augmented_task_dist,
    get_task_arrival_prob,
    negative_fluent_preprocessing,
    print_plan,
)
from railroad.core import ff_heuristic, get_action_by_name
from railroad.dashboard import PlannerDashboard


def main():
    """Debugging scenario for interruption ap planner, where the planner
    takes into account the possiblity of interrupting tasks arriving, but the 
    environment does not allow interrupting tasks to arrive.
    """
    # parameters
    prob_int = 0.3
    interruption_replaces = True # False signifies that the interruption should augment
    procthor_environment = True # False signifies the simple prototype environment
    seed = 201
    name = f"procthor-{"replace" if interruption_replaces else "augment"}-p={prob_int}-{seed}"
    # interruption_value_fn = partial(handcrafted_interruption_value, prob_int)
    interruption_value_fn = None

    if not procthor_environment:
        env = construct_simple_kitchen_environment()
        goal = get_simple_goal()
        interrupting_task_dist = get_simple_task_distribution()[0]
    else:
        env = construct_procthor_kitchen_environment(seed)
        goal = get_example_procthor_goal()
        interrupting_task_dist = get_example_procthor_task_distribution(0)

    # augment the task in the task distribution if necessary
    if not interruption_replaces:
        interrupting_task_dist = get_augmented_task_dist(goal, interrupting_task_dist)

    # fluent pre-processing for usage of FF heuristic
    all_goals = [goal] + list(interrupting_task_dist[0])
    actions, initial_state, converted_goals, _= negative_fluent_preprocessing(
        env.get_actions(), env.state, all_goals
    )

    goal = converted_goals[0]
    interrupting_task_dist = (list(converted_goals[1:]), interrupting_task_dist[1])

    # the prob of interruption depends on the length of time required to compute the longest action
    task_arrival_prob_fn = partial(
        get_task_arrival_prob, RandomVariableType.CONTINUOUS, prob_int,
        DistributionType.EXPONENTIAL, max(get_action_cost(act) for act in actions)
    )

    # keep track of planning time
    start = time.perf_counter()

    plan, cost, _ = astar_search(
        initial_state,
        goal,
        actions,
        interrupting_task_dist,
        ff_heuristic,
        task_arrival_prob_fn,
        interruption_value_fn,
        0,
        num_steps=100000,
        print_trace=False
    )

    duration = time.perf_counter() - start

    if not procthor_environment:
        # temporary, very basic plan outputs for debugging
        action_names = [action.name for action in plan]
        print(f"Probability of Interruption : {prob_int}")
        print_plan(action_names)
        print(f"Discounted Plan Cost: {cost}")
    else: # for procthor_environments
        dash_env = construct_procthor_kitchen_environment(seed)
        planner_factory = partial(AstarDashboardPlanner, heuristic_fn=ff_heuristic)
        with PlannerDashboard(goal, dash_env, planner_factory=planner_factory) as dashboard:
            adapter = AstarDashboardPlanner(dash_env.get_actions(), ff_heuristic)
            for converted_action in plan:
                action = get_action_by_name(dash_env.get_actions(), converted_action.name)
                dash_env.act(action)
                dashboard.update(adapter, action.name)

        # dashboard.show_plots(
        #     save_plot=f"{name}.jpg", show_plot=False, save_video=f"{name}.mp4",
        # )

    print(f"Planning took: {duration: .4f} seconds")


if __name__ == "__main__":
    main()

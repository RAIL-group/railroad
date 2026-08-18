"""
Quick script for manual evaluation of a trained GCN model on specified SceneGraphs.
"""
from collections.abc import Callable
from copy import deepcopy

from interruption.environments import (
    construct_procthor_kitchen_environment,
    get_alfred_task_distribution,
    get_example_procthor_goal,
    # get_example_procthor_task_distribution,
)
from interruption.experiments import (
    ExperimentConfig,
    ExperimentMode,
    ExperimentSeeds,
    TaskArrivalProb,
    initialize_experiment_data, ExperimentData
)
from interruption.planner import compute_interruption_value, astar_search
from interruption.utilities import DistributionType, RandomVariableType
from railroad.core import (
    ff_heuristic, convert_state_to_positive_preconditions, LiteralGoal, Fluent as F,
    Action, get_action_by_name
)
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE
from railroad.environment.procthor.scenegraph import SceneGraph


# constants
MODEL_PATH = DEFAULT_RESOURCES_BASE / "models"

def main():
    """
    Small script to manually compare the actual EV of state to the 
    predicted EV from the trained model.
    """
    # user specifications
    model_name = "best_model_experiment5.pt"
    desired_goal_states = [
        LiteralGoal(F("at knife_9 fridge_4")),
        F("at knife_9 fridge_4") & F("at pan_10 fridge_4")
    ]

    # setup
    seeds = ExperimentSeeds(procthor_seed=201, experiment_seed=20, object_placement_seed=13)
    task_arrival_model = TaskArrivalProb(
        0, RandomVariableType.CONTINUOUS,
        DistributionType.EXPONENTIAL
    )
    env = construct_procthor_kitchen_environment(seeds.procthor_seed)
    task_distribution = get_alfred_task_distribution(
        env.scene.objects,
        set(env.scene.locations),
        one_object_per_taskdist=True
    )

    data = initialize_experiment_data(
        ExperimentConfig(
            seeds,
            get_example_procthor_goal(),
            task_distribution,
            task_arrival_model,
            ff_heuristic,
            MODEL_PATH / model_name
        ),
        ExperimentMode.ANTICIPATORY_PLANNING # using AP to get access to the model
    )
    assert data.planner_parameters.interruption_value_fn is not None
    model = data.planner_parameters.interruption_value_fn
    data.planner_parameters.interruption_value_fn = None

    actual_ev, pred_ev = _get_actual_pred_expected_values(data, model)

    print(f"EV over the task dist of s_0: actual - {actual_ev:.4f}; predicted - {pred_ev:.4f}")

    for goal in desired_goal_states:
        data_copy = deepcopy(data)
        data_copy.search_problem.goal = goal
        initial_state = convert_state_to_positive_preconditions(
            data_copy.env.state, data_copy.neg_to_pos_mapping
        )

        plan, _, _, _ = astar_search(
            (initial_state, None),
            data_copy.search_problem,
            data_copy.planner_parameters
        )

        _update_env_and_scenegraph(plan, data_copy)

        actual_ev, pred_ev = _get_actual_pred_expected_values(data_copy, model)
        print(f"Goal: {goal}")
        print(f"EV over the task dist of s_g: actual - {actual_ev:.4f}; predicted - {pred_ev:.4f}")


def _update_env_and_scenegraph(plan: list[Action], data: ExperimentData) -> None:
    for act in plan:
        action = get_action_by_name(data.env.get_actions(), act.name)
        # perform the action
        data.env.act(action)
        # update the scene_graph
        data.env.update_scene_graph(action)


def _get_actual_pred_expected_values(
    data: ExperimentData,
    model: Callable[[SceneGraph], float]
) -> tuple[float, float]:
    # compute actual expected value
    assert data.search_problem.interrupting_task_dist is not None
    actual_ev = compute_interruption_value(
        convert_state_to_positive_preconditions(
            data.env.state, data.neg_to_pos_mapping
        ),
        data.search_problem.actions,
        data.search_problem.interrupting_task_dist,
        data.planner_parameters.heuristic_fn
    )

    # predicted ev
    predicted_ev = model(data.env.scene.scene_graph)
    return actual_ev, predicted_ev


if __name__ == "__main__":
    main()

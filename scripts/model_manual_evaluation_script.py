"""
Quick script for manual evaluation of a trained GCN model on specified SceneGraphs.
"""
from collections.abc import Callable
from copy import deepcopy
from functools import partial

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
    initialize_experiment_data, ExperimentData
)
from interruption.planner import compute_interruption_value
from interruption.utilities import RandomVariableType, get_task_arrival_prob
from railroad.core import (
    convert_state_to_positive_preconditions, Action, get_action_by_name
)
from railroad.environment.procthor.resources import DEFAULT_RESOURCES_BASE
from railroad.environment.procthor.scenegraph import SceneGraph


# constants
MODEL_PATH = DEFAULT_RESOURCES_BASE / "models"

def main():
    """
    Small script to manually compare the EV of states.
    """
    # user specifications
    model_name = "best_model_experiment10_val.pt"

    # setup
    seeds = ExperimentSeeds(procthor_seed=201, experiment_seed=20, object_placement_seed=21)
    task_arrival_fn = partial(
        get_task_arrival_prob, RandomVariableType.CONTINUOUS, -1, 30
    )

    env = construct_procthor_kitchen_environment(seeds.procthor_seed, remove_duplicates=True)
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
            task_arrival_fn,
            MODEL_PATH / model_name
        ),
        ExperimentMode.ANTICIPATORY_PLANNING, # using AP to get access to the model
        remove_duplicates=True
    )
    assert data.planner_parameters.interruption_value_fn is not None
    model = data.planner_parameters.interruption_value_fn
    data.planner_parameters.interruption_value_fn = None

    setup_plan = [
        get_action_by_name(data.search_problem.actions, "move robot1 start_loc fridge_4"),
        get_action_by_name(data.search_problem.actions, "pick robot1 r1-right fridge_4 spoon_15")
    ]

    _update_env_and_scenegraph(setup_plan, data)

    plans = [
        [get_action_by_name(data.search_problem.actions, "move robot1 fridge_4 countertop_3")],
        [get_action_by_name(data.search_problem.actions, "move robot1 fridge_4 garbagecan_5")]
    ]

    print(f"Initial State: {[fluent for fluent in data.env.state.fluents if "robot1" in fluent.args]}")
    for plan in plans:
        data_copy = deepcopy(data)
        _update_env_and_scenegraph(plan, data_copy)
        approx_ev = model(data_copy.env.scene.scene_graph)
        print(f"{[action.name for action in plan]} : EV over task distribution - {approx_ev}")


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
    actual_ev, _ = compute_interruption_value(
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

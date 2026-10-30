import random
from collections.abc import Callable
from typing import Sequence

from railroad.core import (
    Action,
    Goal,
    LiteralGoal,
    State,
    convert_action_effects,
    convert_action_to_positive_preconditions,
    convert_goal_to_positive_preconditions,
    convert_state_to_positive_preconditions,
    create_positive_fluent_mapping,
    extract_negative_goal_fluents,
    extract_negative_preconditions,
    transition,
)

from .operators import BLOCKING_PREDICATES


# utility functions for interruption anticipatory planning
def get_action_cost(action: Action) -> float:
    """
    Gets the total cost (reward) of performing an action: the time at which the
    robot is free again, plus any extra cost. Effects after that, such as a
    blocking operator's clean-up (see operators.BLOCKING_PREDICATES), don't count;
    an action that never frees a robot costs its last effect's time.
    """
    free_times = [
        effect.time for effect in action.effects
        if any(f.name == "free" and not f.negated for f in effect.resulting_fluents)
    ]
    end_time = max(free_times) if free_times else action.effects[-1].time
    return end_time + action.extra_cost


def get_next_state(
    state: State,
    action: Action,
    interrupting_prob_fn: Callable[[float], float] | float = 0
) -> tuple[State, float]:
    """
    Gets the state s' after performing an action a in s. This function
    assumes the state transition is deterministic. Additionally, returns
    the probability of an interrupting task arriving after the transition.
    """
    if isinstance(interrupting_prob_fn, (float, int)):
        interruption_prob = interrupting_prob_fn
    else:
        interruption_prob = interrupting_prob_fn(get_action_cost(action))
    outcomes = transition(state, action)
    assert len(outcomes) == 1
    next_state, prob = outcomes[0]
    assert prob == 1.0
    return next_state, interruption_prob


def negative_fluent_preprocessing(actions: list[Action], state: State, goals: list[Goal]):
    """
    Wrapper function to convert negative fluents to equivalent positive fluents. Important
    when using the FF heuristic.
    """

    # build negative fluent to equivalent positive fluent mapping
    negative_preconditions = extract_negative_preconditions(actions)
    for goal in goals:
        negative_preconditions = negative_preconditions | extract_negative_goal_fluents(goal)
    # Blocking facts stay negative: the search checks negative preconditions by
    # absence, and the FF heuristic's relaxation ignores them, so converting them
    # would only add a positive fact per object (not-just-picked/-placed) to every
    # state - most of a state's size - for no change in what is planned.
    negative_preconditions = {
        fluent for fluent in negative_preconditions if fluent.name not in BLOCKING_PREDICATES
    }
    mapping = create_positive_fluent_mapping(negative_preconditions)

    # convert actions using mapping
    converted_actions = []
    for action in actions:
        action_pos_precond = convert_action_to_positive_preconditions(action, mapping)
        converted_action = convert_action_effects(action_pos_precond, mapping)
        converted_actions.append(converted_action)

    # convert state using mapping
    converted_state = convert_state_to_positive_preconditions(state, mapping)

    # convert goals using mapping
    converted_goals = []
    for goal in goals:
        converted_goal = convert_goal_to_positive_preconditions(goal, mapping)
        converted_goals.append(converted_goal)
    return converted_actions, converted_state, converted_goals, mapping


def randomize_task_distribution_order(
    task_distribution: tuple[Sequence[Goal], list[float]],
    seed: int
) -> tuple[Goal, tuple[list[Goal], list[float]]]:
    """
    Helper function that randomly selects a task from the task
    distribution to be the current task and re-orders the tasks
    in the task distribution.
    """
    tasks, probs = task_distribution
    rng = random.Random(seed)
    idxes = rng.sample(range(len(tasks)), k=len(tasks))
    return tasks[idxes[-1]], ([tasks[i] for i in idxes], [probs[i] for i in idxes])


def extract_relevant_objects(task_distribution: Sequence[Goal]) -> list[str]:
    """
    Helper function for extracting the objects of tasks within the task distribution.
    Supports the filtering of objects within large ProcTHOR scenes to reduce the
    branching factor of search.
    """
    task_relevant_objects = []
    for task in task_distribution:
        # currently only supports LiteralGoals
        assert isinstance(task, LiteralGoal)
        task_relevant_objects.append(task.fluent().args[0])
    return task_relevant_objects



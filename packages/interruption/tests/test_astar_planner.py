from functools import partial

import numpy as np
import pytest
from interruption.operators import construct_assemble_operator
from interruption.planner import (
    InterruptionTrajectory,
    InterruptionSearchProblem,
    PlannerConfig,
    astar_search,
    check_value_cache,
    compute_interruption_value,
)
from interruption.planning_framework import get_no_int_prob
from interruption.utilities import (
    RandomVariableType,
    get_next_state,
    get_task_arrival_prob,
    negative_fluent_preprocessing,
    get_discounted_value,
    get_reward
)
from railroad.core import Fluent as F, LiteralGoal
from railroad.core import Operator, State, ff_heuristic, get_next_actions
from railroad.environment.symbolic import SymbolicEnvironment
from railroad.operators.core import (
    construct_move_operator,
    construct_pick_operator,
    construct_place_operator,
)

# global
INTERRUPTION_PROB = 0.1

def test_check_value_cache():
    initial_state = State(
        time=0,
        fluents={F("at robot1 kitchen"), F("free robot1")}
    )
    initial_fluents = frozenset(initial_state.fluents)

    value_cache = {}
    assert not check_value_cache(initial_fluents, value_cache)

    value_cache[initial_fluents] = 2.5
    assert check_value_cache(initial_fluents, value_cache)


def test_get_no_int_prob():
    traj = InterruptionTrajectory(
        state_history=[],
        plan=[],
        interruption_probs=[],
        scene_graph=None
    )
    assert get_no_int_prob(traj.interruption_probs) == 1
    traj.interruption_probs.append(0.1)
    assert get_no_int_prob(traj.interruption_probs) == 0.9
    traj.interruption_probs.append(0.1)
    assert get_no_int_prob(traj.interruption_probs) == 0.81


@pytest.mark.parametrize(
    'interruption_value, solution',
    [(0, [4, 7.6, 10.84]), (2, [4.2, 7.98, 11.382])]
)
def test_discounted_accumulated_cost(interruption_value, solution):
    # setup
    move_op = construct_move_operator(4)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
    }

    move_actions = move_op.instantiate(objects_by_type)
    assert len(move_actions) == 2

    initial_state = State(
        time=0,
        fluents={F("at robot1 kitchen"), F("free robot1")}
    )

    traj = InterruptionTrajectory(
        state_history=[initial_state],
        plan=[],
        interruption_probs=[],
        scene_graph=None
    )

    # solution stores the discounted accumulated cost after taking an action
    # from the current state of the trajectory
    for discounted_acc_cost in solution:
        applicable_actions = get_next_actions(
            traj.state_history[-1], move_actions
        )
        assert len(applicable_actions) == 1
        reward = get_reward(
            applicable_actions[0],
            get_no_int_prob(traj.interruption_probs),
            interruption_value * 0.1
        )
        assert reward + traj.cost == pytest.approx(discounted_acc_cost)

        # update the trajectory
        next_state, _ = get_next_state(traj.state_history[-1], applicable_actions[0])
        traj.level+=1
        traj.state_history.append(next_state)
        traj.plan.append(applicable_actions[0])
        traj.interruption_probs.append(0.1)
        traj.value += reward
        traj.cost += reward


def test_h():
    # setup
    move_op = construct_move_operator(4)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
    }

    move_actions = move_op.instantiate(objects_by_type)
    assert len(move_actions) == 2

    initial_state = State(
        time=0,
        fluents={F("at robot1 kitchen"), F("free robot1")}
    )

    traj = InterruptionTrajectory(
        state_history=[initial_state],
        plan=[],
        interruption_probs=[],
        scene_graph=None
    )

    # tests for when passed in hueristic_fn is an int
    assert get_discounted_value(
        5, get_no_int_prob(traj.interruption_probs) * (1 - 0.1), 0
    ) == pytest.approx(0.9 * 5)
    traj.interruption_probs.append(0.1)
    assert get_discounted_value(
        5, get_no_int_prob(traj.interruption_probs) * (1 - 0.1), 0
    ) == pytest.approx(0.81 * 5)
    traj.interruption_probs.append(0.1)
    assert get_discounted_value(
        5, get_no_int_prob(traj.interruption_probs) * (1 - 0.1), 0
    ) == pytest.approx(0.729 * 5)


def test_h_user_reward():
    # setup
    reward = 1
    move_op = construct_move_operator(4)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
    }

    move_actions = move_op.instantiate(objects_by_type)
    assert len(move_actions) == 2

    initial_state = State(
        time=0,
        fluents={F("at robot1 kitchen"), F("free robot1")}
    )

    traj = InterruptionTrajectory(
        state_history=[initial_state],
        plan=[],
        interruption_probs=[],
        scene_graph=None
    )

    # tests for when passed in hueristic_fn is an int
    assert get_discounted_value(
        5, get_no_int_prob(traj.interruption_probs) * (1 - 0.1), reward
    ) == pytest.approx(0.9 * 6)
    traj.interruption_probs.append(0.1)
    assert get_discounted_value(
        5, get_no_int_prob(traj.interruption_probs) * (1 - 0.1), reward
    ) == pytest.approx(0.81 * 6)
    traj.interruption_probs.append(0.1)
    assert get_discounted_value(
        5, get_no_int_prob(traj.interruption_probs) * (1 - 0.1), reward
    ) == pytest.approx(0.729 * 6)


@pytest.mark.parametrize("heuristic_fn", [0, 5])
def test_construct_trajectory(heuristic_fn):
    # setup
    move_op = construct_move_operator(4)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
    }

    move_actions = move_op.instantiate(objects_by_type)
    assert len(move_actions) == 2

    initial_state = State(
        time=0,
        fluents={F("at robot1 kitchen"), F("free robot1")}
    )

    goal = F("at robot1 living_room") & F("free robot1")

    applicable_actions = get_next_actions(initial_state, move_actions)
    assert len(applicable_actions) == 1
    action = applicable_actions[0]
    new_state, next_interruption_prob = get_next_state(initial_state, action, 0.1)

    traj = InterruptionTrajectory(
        state_history=[initial_state],
        plan=[],
        interruption_probs=[],
        scene_graph=None
    )
    search_problem = InterruptionSearchProblem(goal, applicable_actions)
    planner_params = PlannerConfig(get_no_int_prob, heuristic_fn)
    new_traj = traj.create_child(
        search_problem,
        planner_params,
        action,
        next_interruption_prob
    )

    assert new_traj.level == 1
    assert new_traj.cost == 4
    assert new_traj.value == new_traj.cost + heuristic_fn * 0.9
    assert len(new_traj.state_history) == 2
    assert new_traj.state_history == [initial_state, new_state]
    assert len(new_traj.plan) == 1
    assert new_traj.plan == [action]
    assert len(new_traj.interruption_probs) == 1
    assert new_traj.interruption_probs == [0.1]


@pytest.mark.parametrize("heuristic_fn", [0, 5, ff_heuristic])
def test_astart_search_nointdist(heuristic_fn):
    # setup
    move_op = construct_move_operator(4)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
    }

    move_actions = move_op.instantiate(objects_by_type)
    assert len(move_actions) == 2

    initial_state = State(
        time=0,
        fluents={F("at robot1 kitchen"), F("free robot1")}
    )

    goal = F("at robot1 living_room") & F("free robot1")

    # testing with no interrupting tasks
    search_problem = InterruptionSearchProblem(goal, move_actions)
    planner_params = PlannerConfig(get_no_int_prob, heuristic_fn, 0.1)

    plan, plan_cost, success, _ = astar_search(
        (initial_state, None),
        search_problem,
        planner_params
    )

    assert success is True
    assert len(plan) == 1
    assert plan[0].name == "move robot1 kitchen living_room"
    assert plan_cost == 4

    # setup test scenario 2
    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
        "object": {"water_bottle"}
    }
    pick_op = construct_pick_operator(3)
    place_op = construct_place_operator(2)

    # instantiate actions
    pick_actions = pick_op.instantiate(objects_by_type)
    move_actions = move_op.instantiate(objects_by_type)
    place_actions = place_op.instantiate(objects_by_type)
    all_actions = pick_actions + move_actions + place_actions

    initial_state = State(
        time=0,
        fluents={
            F("at robot1 kitchen"), F("free robot1"),
            F("at water_bottle kitchen"), ~F("hand-full robot1")
        }
    )

    goal = (
        F("at robot1 living_room") &
        F("free robot1") &
        F("at water_bottle living_room")
    )

    search_problem = InterruptionSearchProblem(goal, all_actions)
    planner_params = PlannerConfig(get_no_int_prob, heuristic_fn, 0.1)

    plan, plan_cost, success, _ = astar_search(
        (initial_state, None),
        search_problem,
        planner_params
    )
    assert success is True
    assert len(plan) == 3
    plan_with_names = [action.name for action in plan]
    # print(plan_with_names)
    assert plan_with_names == [
        "pick robot1 kitchen water_bottle",
        "move robot1 kitchen living_room",
        "place robot1 living_room water_bottle"
    ]
    assert plan_cost == pytest.approx(3 + 4 * 0.9 + 2 * 0.81)


@pytest.mark.parametrize("task_distribution", [
    ([F("at robot1 living_room") & F("free robot1")], [1]),
    (
        [
            F("at robot1 living_room") & F("free robot1"),
            F("at robot1 living_room") &
            F("free robot1") &
            F("at water_bottle living_room")
        ], [0.5, 0.5]
    )
])
def test_compute_interruption_value(task_distribution):
    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
        "object": {"water_bottle"}
    }
    move_op = construct_move_operator(4)
    pick_op = construct_pick_operator(3)
    place_op = construct_place_operator(2)

    # instantiate actions
    pick_actions = pick_op.instantiate(objects_by_type)
    move_actions = move_op.instantiate(objects_by_type)
    place_actions = place_op.instantiate(objects_by_type)
    all_actions = pick_actions + move_actions + place_actions

    initial_state = State(
        time=0,
        fluents={
            F("at robot1 kitchen"),
            F("free robot1"),
            F("at water_bottle kitchen"),
            ~F("hand-full robot1")
        }
    )
    expected_value = compute_interruption_value(initial_state, all_actions, task_distribution)

    if len(task_distribution[0]) == 1:
        assert expected_value == 4
    else:
        assert expected_value == pytest.approx(0.5*4 + 0.5*(3 + 4 + 2))


@pytest.mark.parametrize("heuristic_fn", [0, ff_heuristic])
@pytest.mark.parametrize(
    "interruption_prob_fn",
    [
        0.0,
        partial(get_task_arrival_prob, RandomVariableType.DISCRETE, 0.0),
        partial(get_task_arrival_prob, RandomVariableType.CONTINUOUS, 0.0)
    ]
)
def test_optimal_make_sandwhich_noint(heuristic_fn, interruption_prob_fn):
    # setup
    locations = {
            "refrigerator": np.array([0, 0]),
            "pantry": np.array([1, 0]),
            "countertop1": np.array([1,1]),
            "countertop2": np.array([2,2]),
            "table": np.array([0,2])
    }

    objects_by_type = {
        "robot": {"robot1"},
        "location": set(locations),
        "object": {"turkey", "bread", "sandwhich"}
    }

    initial_fluents = {
        F("free robot1"), F("at robot1 table"), F("is-turkey turkey"), F("is-bread bread"),
        ~F("hand-full robot1"), F("at turkey refrigerator"), F("at bread pantry"),
        ~F("prep-station table"), F("prep-station countertop2"), ~F("prep-station refrigerator"),
        ~F("sandwhich-made"), F("prep-station countertop1"), F("is-sandwhich sandwhich")
    }

    initial_state = State(0.0, initial_fluents)

    def move_time(robot, loc_from, loc_to):
        return float(np.linalg.norm(locations[loc_from] - locations[loc_to]))

    class TestEnvironment(SymbolicEnvironment):
        """
        Symbolic environment used for testing.
        """
        def define_operators(self) -> list[Operator]:
            move_op = construct_move_operator(move_time)
            pick_op = construct_pick_operator(1)
            place_op = construct_place_operator(1)
            assemble_op = construct_assemble_operator(3)
            return [pick_op, place_op, move_op, assemble_op]

    env = TestEnvironment(initial_state, objects_by_type)

    # Task: make sandwhich
    goal = LiteralGoal(F("sandwhich-made"))

    # fluent pre-processing for usage of FF heuristic
    actions, initial_state, converted_goals, _= negative_fluent_preprocessing(
        env.get_actions(), initial_state, [goal]
    )

    goal = converted_goals[0]

    search_problem = InterruptionSearchProblem(goal, actions)
    planner_params = PlannerConfig(get_no_int_prob, heuristic_fn, interruption_prob_fn)

    plan, cost, success, _ = astar_search(
        (initial_state, None),
        search_problem,
        planner_params
    )

    assert success is True
    assert cost == pytest.approx(12.414213562373096)
    solution = [
        'move robot1 table refrigerator',
        'pick robot1 refrigerator turkey',
        'move robot1 refrigerator countertop1',
        'place robot1 countertop1 turkey',
        'move robot1 countertop1 pantry',
        'pick robot1 pantry bread',
        'move robot1 pantry countertop1',
        'place robot1 countertop1 bread',
        'assemble robot1 turkey bread sandwhich countertop1'
    ]
    assert [a.name for a in plan] == solution

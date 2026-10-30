from functools import partial

import numpy as np
import pytest
from interruption.cost_formulations import NO_INTERRUPTION, DefaultFormulation
from interruption.operators import construct_assemble_operator
from interruption.planner import (
    InterruptionTrajectory,
    InterruptionSearchProblem,
    PlannerConfig,
    astar_search,
    check_value_cache,
    compute_interruption_value,
)
from interruption.utilities import (
    RandomVariableType,
    get_next_state,
    get_task_arrival_prob,
    negative_fluent_preprocessing,
    get_discounted_value,
    get_reward
)
from railroad.core import Action, Fluent as F, LiteralGoal
from railroad.core import Operator, State, ff_heuristic, get_next_actions
from railroad.environment.symbolic import SymbolicEnvironment
from railroad.operators.core import (
    construct_move_operator,
    construct_pick_operator,
    construct_place_operator,
)

# global
INTERRUPTION_PROB = 0.1
NO_INT = DefaultFormulation(NO_INTERRUPTION)

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
        state=initial_state,
        action=None,
        discount_factor=1,
        scene_graph=None
    )

    # solution stores the discounted accumulated cost after taking an action
    # from the current state of the trajectory
    for discounted_acc_cost in solution:
        applicable_actions = get_next_actions(
            traj.state, move_actions
        )
        assert len(applicable_actions) == 1
        reward = get_reward(
            applicable_actions[0],
            traj.discount_factor,
            interruption_value * 0.1
        )
        assert reward + traj.cost == pytest.approx(discounted_acc_cost)

        # update the trajectory
        next_state, _ = get_next_state(traj.state, applicable_actions[0])
        traj.state = next_state
        traj.action = applicable_actions[0]
        traj.discount_factor *= (1-0.1)
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
        state=initial_state,
        action=None,
        discount_factor=1,
        scene_graph=None
    )

    # tests for when passed in hueristic_fn is an int
    assert get_discounted_value(
        5, traj.discount_factor * (1 - 0.1), 0
    ) == pytest.approx(0.9 * 5)
    traj.discount_factor*=(1-0.1)
    assert get_discounted_value(
        5, traj.discount_factor * (1 - 0.1), 0
    ) == pytest.approx(0.81 * 5)
    traj.discount_factor*=(1-0.1)
    assert get_discounted_value(
        5, traj.discount_factor * (1 - 0.1), 0
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
        state=initial_state,
        action=None,
        discount_factor=1,
        scene_graph=None
    )

    # tests for when passed in hueristic_fn is an int
    assert get_discounted_value(
        5, traj.discount_factor * (1 - 0.1), reward
    ) == pytest.approx(0.9 * 6)
    traj.discount_factor*=(1-0.1)
    assert get_discounted_value(
        5, traj.discount_factor * (1 - 0.1), reward
    ) == pytest.approx(0.81 * 6)
    traj.discount_factor*=(1-0.1)
    assert get_discounted_value(
        5, traj.discount_factor * (1 - 0.1), reward
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
        state=initial_state,
        action=None,
        discount_factor=1,
        scene_graph=None
    )
    search_problem = InterruptionSearchProblem(goal, applicable_actions)
    planner_params = PlannerConfig(heuristic_fn=heuristic_fn, cost_formulation=NO_INT)
    new_traj = traj.create_child(
        search_problem,
        planner_params,
        action,
        next_interruption_prob
    )

    assert new_traj.cost == 4
    assert new_traj.value == new_traj.cost + heuristic_fn * 0.9
    assert new_traj.state == new_state
    assert new_traj.action == action
    assert new_traj.discount_factor == 0.9


@pytest.mark.parametrize("heuristic_fn", [0, 5, ff_heuristic])
def test_astar_search_nointdist(heuristic_fn):
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
    planner_params = PlannerConfig(
        heuristic_fn=heuristic_fn, cost_formulation=NO_INT, planner_interruption_prob_fn=0.1
    )

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
    planner_params = PlannerConfig(
        heuristic_fn=heuristic_fn, cost_formulation=NO_INT, planner_interruption_prob_fn=0.1
    )

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
    expected_value, _ = compute_interruption_value(initial_state, all_actions, task_distribution)

    if len(task_distribution[0]) == 1:
        assert expected_value == 4
    else:
        assert expected_value == pytest.approx(0.5*4 + 0.5*(3 + 4 + 2))


@pytest.mark.parametrize("heuristic_fn", [0, ff_heuristic])
@pytest.mark.parametrize(
    "interruption_prob_fn",
    [
        0.0,
        partial(get_task_arrival_prob, RandomVariableType.DISCRETE, 0.0, -1),
        partial(get_task_arrival_prob, RandomVariableType.CONTINUOUS, -1, float("inf"))
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
    planner_params = PlannerConfig(
        heuristic_fn=heuristic_fn, cost_formulation=NO_INT,
        planner_interruption_prob_fn=interruption_prob_fn,
    )

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


@pytest.mark.parametrize("show_progress", [False, True])
def test_astar_search_prints_progress_only_when_asked(show_progress, capsys):
    # the Python search draws a progress bar; the C++ one only prints the goal
    move_actions = construct_move_operator(4).instantiate(
        {"robot": {"robot1"}, "location": {"kitchen", "living_room"}}
    )
    initial_state = State(time=0, fluents={F("at robot1 kitchen"), F("free robot1")})
    goal = F("at robot1 living_room") & F("free robot1")
    search_problem = InterruptionSearchProblem(goal, move_actions)

    astar_search(
        (initial_state, None), search_problem,
        PlannerConfig(heuristic_fn=0, cost_formulation=NO_INT, use_cpp_search=False),
        show_progress=show_progress,
    )

    captured = capsys.readouterr()
    assert ("Current Goal" in captured.out) == show_progress
    assert ("it/s" in captured.err or "it]" in captured.err) == show_progress


def test_search_stats_report_expansions_against_the_budget():
    from interruption.planner import SearchStats

    move_actions = construct_move_operator(4).instantiate(
        {"robot": {"robot1"}, "location": {"kitchen", "living_room"}}
    )
    initial_state = State(time=0, fluents={F("at robot1 kitchen"), F("free robot1")})
    goal = F("at robot1 living_room") & F("free robot1")
    stats = SearchStats()

    astar_search(
        (initial_state, None), InterruptionSearchProblem(goal, move_actions),
        PlannerConfig(heuristic_fn=0, cost_formulation=NO_INT), num_steps=50, stats=stats,
    )

    assert stats.success and stats.budget == 50
    assert 0 < stats.expanded < 50
    assert stats.fraction == stats.expanded / 50


def test_search_stats_show_an_exhausted_budget():
    from interruption.planner import SearchStats

    move_actions = construct_move_operator(4).instantiate(
        {"robot": {"robot1"}, "location": {"kitchen", "living_room"}}
    )
    initial_state = State(time=0, fluents={F("at robot1 kitchen"), F("free robot1")})
    stats = SearchStats()

    # unreachable goal: the search runs until its budget is spent
    astar_search(
        (initial_state, None),
        InterruptionSearchProblem(LiteralGoal(F("at robot1 garage")), move_actions),
        PlannerConfig(heuristic_fn=0, cost_formulation=NO_INT), num_steps=1, stats=stats,
    )

    assert not stats.success
    assert (stats.expanded, stats.budget, stats.fraction) == (1, 1, 1.0)


def _named_actions(*names):
    return [Action(set(), [], name=name) for name in names]


def test_goal_objects_are_the_objects_its_literals_name():
    from interruption.planner import goal_objects

    goal = F("at-category cup_3 fridge") & F("at-category bowl_2 sofa")
    assert goal_objects(goal, {"cup_3", "bowl_2", "apple_41"}) == {"cup_3", "bowl_2"}


def test_filtering_keeps_moves_and_the_goal_objects_actions():
    from interruption.planner import filter_actions_to_objects

    actions = _named_actions(
        "move robot1 countertop_1 fridge_5",
        "pick-left robot1 countertop countertop_1 cup_3",
        "pick-left robot1 countertop countertop_1 apple_41",
    )
    kept = filter_actions_to_objects(actions, {"cup_3"}, {"cup_3", "apple_41"})
    assert [a.name for a in kept] == [
        "move robot1 countertop_1 fridge_5", "pick-left robot1 countertop countertop_1 cup_3"
    ]


def test_a_search_that_exhausts_every_reachable_state_returns_failure():
    from interruption.planner import SearchStats

    move_actions = construct_move_operator(4).instantiate(
        {"robot": {"robot1"}, "location": {"kitchen", "living_room"}}
    )
    initial_state = State(time=0, fluents={F("at robot1 kitchen"), F("free robot1")})
    stats = SearchStats()

    # unreachable, and the reachable space is far smaller than the budget
    plan, cost, success, _ = astar_search(
        (initial_state, None),
        InterruptionSearchProblem(LiteralGoal(F("at robot1 garage")), move_actions),
        PlannerConfig(heuristic_fn=0, cost_formulation=NO_INT), num_steps=1000, stats=stats,
    )

    assert (plan, cost, success) == ([], float("inf"), False)
    assert 0 < stats.expanded < 1000 and not stats.success


def test_held_objects_are_the_objects_in_a_robots_hands():
    from interruption.planner import held_objects

    state = State(time=0, fluents={
        F("holding-in-left robot1 bowl_32"), F("at apple_38 countertop_4"), F("free robot1"),
    })
    assert held_objects(state, {"bowl_32", "apple_38"}) == {"bowl_32"}

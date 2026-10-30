
import pytest
from interruption.cost_formulations import NO_INTERRUPTION, DefaultFormulation
from interruption.planner import (
    InterruptionTrajectory,
    InterruptionSearchProblem,
    PlannerConfig,
    astar_search,
    check_value_cache,
)
from interruption.utilities import (
    get_next_state,
)
from railroad.core import Action, Fluent as F, LiteralGoal
from railroad.core import State, ff_heuristic, get_next_actions
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

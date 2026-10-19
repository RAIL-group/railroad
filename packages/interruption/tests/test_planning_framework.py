from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from interruption import planning_framework as pf
from interruption.planning_framework import ap_heuristic_fn
from interruption.planner import InterruptionSearchProblem, PlannerConfig

from railroad.core import Fluent as F, Goal, LiteralGoal, State
from railroad.operators.core import construct_move_operator, construct_pick_operator
from railroad.environment.procthor.scene import ProcTHORScene
from railroad.environment.procthor.scenegraph import SceneGraph


def _build_move_and_pick_fixture() -> tuple[State, Goal, list]:
    """
    Minimal move-then-pick problem: robot r1 starts at "start", box is at
    "target". Reaching "holding r1 box" costs move (5.0) + pick (2.0) = 7.0,
    the same fixture shape used by railroad's own test_ff_heuristic_move_and_pick,
    rebuilt here with the real railroad.operators.core operators (matching this
    package's existing test_astar_planner.py convention).
    """
    move_op = construct_move_operator(5.0)
    pick_op = construct_pick_operator(2.0)
    objects_by_type = {
        "robot": ["r1"], "location": ["start", "target"], "object": ["box"],
    }
    actions = [*move_op.instantiate(objects_by_type), *pick_op.instantiate(objects_by_type)]
    state = State(time=0, fluents={F("at r1 start"), F("free r1"), F("at box target")})
    goal = LiteralGoal(F("holding r1 box"))
    return state, goal, actions


def test_ap_heuristic_fn_without_v_ap_matches_ff_heuristic():
    """include_v_ap=False must ignore v_ap entirely, not just default it to 0."""
    state, goal, actions = _build_move_and_pick_fixture()

    assert ap_heuristic_fn(state, goal, actions, v_ap=1000.0) == 7.0


def test_ap_heuristic_fn_with_v_ap_adds_expected_value():
    state, goal, actions = _build_move_and_pick_fixture()

    assert ap_heuristic_fn(state, goal, actions, v_ap=3.5, weights=(1, 1)) == pytest.approx(10.5)


@pytest.mark.parametrize("v_ap", [0.0, -2.0, 100.0])
def test_ap_heuristic_fn_with_v_ap_is_additive(v_ap):
    """
    Guards against a future change accidentally clamping or scaling v_ap
    instead of a plain addition -- nothing in ap_heuristic_fn currently
    prevents the result from going negative when v_ap is negative enough.
    """
    state, goal, actions = _build_move_and_pick_fixture()
    base = ap_heuristic_fn(state, goal, actions)

    assert ap_heuristic_fn(state, goal, actions, v_ap=v_ap, weights=(1, 1)) == pytest.approx(base + v_ap)


def test_ap_heuristic_fn_with_v_ap_true_and_zero_matches_without_v_ap():
    state, goal, actions = _build_move_and_pick_fixture()

    without = ap_heuristic_fn(state, goal, actions)
    with_zero = ap_heuristic_fn(state, goal, actions, v_ap=0.0, weights=(1, 1))

    assert with_zero == without


# ---------------------------------------------------------------------------
# anticipatory_planner: astar_search failure handling + no input mutation
# ---------------------------------------------------------------------------
#
# astar_search, focused_sampling and _get_sampled_augmented_tasks are stubbed,
# so these run without a real search or ProcTHOR scene. Two regressions are
# covered:
#   1. a failed astar_search must not be treated as a solution, must not crash,
#      and its partial-plan cost must not leak into the returned value;
#   2. interruption_problem.goal and search_params.interruption_value_fn must be
#      left exactly as passed in, so the same problem/config objects stay usable
#      for the next task (the data-generation loop reuses them).


class _FakeAstar:
    """
    Stand-in for astar_search. Returns queued
    (plan, value, success, scene_graph) tuples, one per call, and records the
    goal / interruption-value-fn visible on each call so tests can assert the
    mutate-then-restore behaviour. A queued exception instance is raised rather
    than returned.
    """

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def __call__(self, initial_state, problem, params, num_steps=20000):
        self.calls.append(
            SimpleNamespace(goal=problem.goal, ev_fn=params.interruption_value_fn)
        )
        assert self._results, "astar_search called more times than results provided"
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def ap_setup(monkeypatch):
    original_goal = LiteralGoal(F("holding r1 box"))
    aug_a = LiteralGoal(F("at mug counter"))
    aug_b = LiteralGoal(F("at pan sink"))

    ev_seen = []

    def ev_model(scene_graph):
        assert scene_graph is not None, "ev_model called with a None scene graph"
        ev_seen.append(scene_graph)
        return 10.0

    problem = InterruptionSearchProblem(goal=original_goal, actions=[])
    config = PlannerConfig(
        False,
        heuristic_fn=0.0,
        discount=1,
        interruption_value_fn=ev_model
    )
    scene = MagicMock(spec=ProcTHORScene)
    scene.grid = None
    scene.locations = {}
    scene.object_locations = {}

    initial_state = (State(time=0, fluents={F("free r1")}), SceneGraph())

    monkeypatch.setattr(pf, "focused_sampling", lambda *a, **k: (set(), set()))

    def run(astar_results, *, augmented_tasks=None, ap_debug=False):
        tasks = [aug_a, aug_b] if augmented_tasks is None else list(augmented_tasks)
        monkeypatch.setattr(pf, "_get_sampled_augmented_tasks", lambda *a, **k: list(tasks))
        if ap_debug:
            monkeypatch.setattr(pf, "AP_DEBUG", True)
        fake = _FakeAstar(astar_results)
        monkeypatch.setattr(pf, "astar_search", fake)
        # initial_state[1] / scene stand in for a SceneGraph / ProcTHORScene that
        # the stubbed astar_search and focused_sampling never actually touch.
        result = pf.anticipatory_planner(
            initial_state, problem, config, scene, {}
        )
        return result, fake

    return SimpleNamespace(
        run=run,
        problem=problem,
        config=config,
        original_goal=original_goal,
        aug_a=aug_a,
        aug_b=aug_b,
        ev_model=ev_model,
        ev_seen=ev_seen,
    )


def test_all_searches_failing_returns_unsuccessful_without_raising(ap_setup):
    (plan, value_sg, success), _ = ap_setup.run(
        [
            ([], 5.0, False, "sg_init"),
            (["partial_a"], 3.0, False, "sg_a"),
            (["partial_b"], 2.0, False, "sg_b"),
        ]
    )

    assert not success
    assert plan == []
    # the finite partial-plan cost (5.0/3.0/2.0) must not leak through
    assert value_sg == float("inf")


def test_initial_success_survives_all_augmented_failures(ap_setup):
    myopic_plan = ["move r1 start target", "pick r1 box target"]
    (plan, value_sg, success), _ = ap_setup.run(
        [
            (myopic_plan, 7.0, True, "sg_init"),
            (["junk_a"], 1.0, False, "sg_a"),
            (["junk_b"], 0.5, False, "sg_b"),
        ]
    )

    assert success
    assert plan == myopic_plan
    assert value_sg == 7.0


def test_cheaper_failed_augmented_plan_is_not_selected(ap_setup):
    myopic_plan = ["m1"]
    better_plan = ["m1", "m2"]
    (plan, value_sg, success), _ = ap_setup.run(
        [
            (myopic_plan, 20.0, True, "sg_init"),        # total 30.0
            (better_plan, 5.0, True, "sg_a"),            # total 15.0 -> selected
            (["cheap_but_failed"], 0.1, False, "sg_b"),  # total 10.1 but unsuccessful
        ]
    )

    assert success
    assert plan == better_plan
    assert value_sg == 5.0


def test_no_augmented_tasks_returns_myopic_result(ap_setup):
    myopic_plan = ["m1", "m2"]
    (plan, value_sg, success), fake = ap_setup.run(
        [(myopic_plan, 8.0, True, "sg_init")],
        augmented_tasks=[],
    )

    assert success
    assert plan == myopic_plan
    assert value_sg == 8.0
    assert len(fake.calls) == 1
    assert ap_setup.problem.goal is ap_setup.original_goal


def test_no_augmented_tasks_failed_initial_is_unsuccessful(ap_setup):
    (plan, value_sg, success), fake = ap_setup.run(
        [(["partial"], 3.0, False, "sg_init")],
        augmented_tasks=[],
    )

    assert not success
    assert plan == []
    assert value_sg == float("inf")
    assert len(fake.calls) == 1


def test_goal_restored_after_successful_planning(ap_setup):
    ap_setup.run(
        [
            (["p"], 7.0, True, "sg_init"),
            (["p"], 6.0, True, "sg_a"),
            (["p"], 6.0, True, "sg_b"),
        ]
    )

    assert ap_setup.problem.goal is ap_setup.original_goal


def test_goal_restored_when_search_fails(ap_setup):
    ap_setup.run(
        [
            ([], 5.0, False, "sg_init"),
            (["a"], 2.0, False, "sg_a"),
            (["b"], 2.0, False, "sg_b"),
        ]
    )

    assert ap_setup.problem.goal is ap_setup.original_goal


def test_interruption_value_fn_restored_after_planning(ap_setup):
    ap_setup.run(
        [
            (["p"], 7.0, True, "sg_init"),
            (["p"], 6.0, True, "sg_a"),
            (["p"], 6.0, True, "sg_b"),
        ]
    )

    assert ap_setup.config.interruption_value_fn is ap_setup.ev_model


def test_augmented_goals_are_tried_then_original_is_restored(ap_setup):
    (_, _, _), fake = ap_setup.run(
        [
            (["p"], 7.0, True, "sg_init"),
            (["p"], 6.0, True, "sg_a"),
            (["p"], 6.0, True, "sg_b"),
        ]
    )

    assert len(fake.calls) == 3
    seen_goals = [call.goal for call in fake.calls]
    assert seen_goals[0] is ap_setup.original_goal
    assert seen_goals[1] is ap_setup.aug_a
    assert seen_goals[2] is ap_setup.aug_b
    assert ap_setup.problem.goal is ap_setup.original_goal
    # the inner myopic searches must not carry the interruption value fn
    assert all(call.ev_fn is None for call in fake.calls)


def test_problem_object_is_reusable_across_calls(ap_setup):
    results = [
        (["p"], 7.0, True, "sg_init"),
        (["p"], 6.0, True, "sg_a"),
        (["p"], 6.0, True, "sg_b"),
    ]
    (_, _, _), fake1 = ap_setup.run(list(results))
    (_, _, _), fake2 = ap_setup.run(list(results))

    assert fake1.calls[0].goal is ap_setup.original_goal
    assert fake2.calls[0].goal is ap_setup.original_goal


def _run_search_with_retry(monkeypatch, outcomes, held=(), **kwargs):
    """
    search_with_retry on a 3-goal task whose searches succeed or fail in the given
    order. Returns its result, every search's action names, and the stand-ins.
    """
    from railroad.core import Action

    outcomes = iter(outcomes)
    searched_actions = []

    def search(mode, env, problem, *args):
        searched_actions.append([a.name for a in problem.actions])
        return ["plan"], next(outcomes), 0, 0


    monkeypatch.setattr(pf, "search_without_retry", search)
    monkeypatch.setattr(pf, "convert_state_to_positive_preconditions", lambda state, _: state)
    monkeypatch.setattr(pf, "_get_task_goal_count", lambda goal: 3)
    monkeypatch.setattr(pf, "_decompose_task", lambda state, task, *args: [task])

    goal = LiteralGoal(F("at-category apple_41 fridge"))
    actions = [
        Action(set(), [], name="move robot1 countertop_1 fridge_5"),
        Action(set(), [], name="pick-left robot1 countertop countertop_1 apple_41"),
        Action(set(), [], name="pick-left robot1 countertop countertop_1 pan_28"),
    ]
    search_problem = SimpleNamespace(goal=goal, actions=actions)
    planner_parameters = SimpleNamespace(max_task_complexity=-1, heuristic_fn=None)
    state = State(0.0, {F(f"holding-in-left robot1 {obj}") for obj in held})
    scene = SimpleNamespace(objects={"apple_41", "pan_28"})
    environment = SimpleNamespace(state=state, scene=scene)

    search_with_retry: Any = pf.search_with_retry  # called with stand-in arguments
    result = search_with_retry(
        pf.PlannerMode.MYOPIC, environment, search_problem, planner_parameters, {}, **kwargs
    )
    return result, searched_actions, search_problem, planner_parameters, goal, actions


def test_search_with_retry_needs_nothing_more_when_the_first_search_succeeds(monkeypatch):
    result, searched, problem, params, goal, actions = _run_search_with_retry(monkeypatch, [True])

    assert result == (["plan"], True, 0, 0)
    assert len(searched) == 1
    assert params.max_task_complexity == -1


def test_search_with_retry_retries_over_goal_objects_before_shrinking(monkeypatch):
    result, searched, problem, params, goal, actions = _run_search_with_retry(
        monkeypatch, [False, True]
    )

    assert result == (["plan"], True, 1, 1)
    # the retry saw only the move and the apple's actions, not the pan's
    assert searched[1] == [
        "move robot1 countertop_1 fridge_5", "pick-left robot1 countertop countertop_1 apple_41"
    ]
    # the task was solved whole, so the planner's complexity limit is unchanged
    assert params.max_task_complexity == -1
    assert problem.goal is goal and problem.actions is actions


def test_search_with_retry_shrinks_the_task_only_when_both_searches_fail(monkeypatch):
    result, searched, problem, params, goal, actions = _run_search_with_retry(
        monkeypatch, [False, False, True]
    )

    assert result == (["plan"], True, 2, 1)
    assert params.max_task_complexity == 2
    assert problem.goal is goal and problem.actions is actions


def test_search_with_retry_gives_up_when_no_single_goal_can_be_solved(monkeypatch):
    # every search fails: 3 task sizes, an unfiltered and a filtered search each
    result, searched, problem, params, goal, actions = _run_search_with_retry(
        monkeypatch, [False] * 6
    )

    assert result == (["plan"], False, 6, 3)
    # never below 1, which _decompose_task needs on the next call
    assert params.max_task_complexity == 1
    assert problem.goal is goal and problem.actions is actions


def test_search_with_retry_works_again_after_giving_up(monkeypatch):
    # giving up must not leave state behind that breaks the next call
    result, searched, problem, params, goal, actions = _run_search_with_retry(
        monkeypatch, [False] * 6
    )
    monkeypatch.setattr(pf, "search_without_retry", lambda *args, **kwargs: (["plan"], True, 0, 0))
    monkeypatch.setattr(pf, "_decompose_task", _decompose_requiring_a_size)

    search_with_retry: Any = pf.search_with_retry
    environment = SimpleNamespace(state=None, scene=SimpleNamespace(objects={"apple_41"}))
    assert search_with_retry(
        pf.PlannerMode.MYOPIC, environment, problem, params, {}
    )[:2] == (["plan"], True)


def _decompose_requiring_a_size(state, task, actions, heuristic_fn, num_goals_in_subtask):
    assert num_goals_in_subtask > 0  # as the real _decompose_task does
    return [task]


def test_search_with_retry_without_decomposition_stops_after_the_goal_object_retry(monkeypatch):
    result, searched, problem, params, goal, actions = _run_search_with_retry(
        monkeypatch, [False, False], allow_decomposition=False
    )

    assert result == (["plan"], False, 2, 1)
    assert len(searched) == 2
    assert params.max_task_complexity == -1


def test_search_with_retry_records_stats_for_both_searches(monkeypatch):
    from interruption.planner import SearchStats

    stats: list[SearchStats] = []
    _run_search_with_retry(monkeypatch, [False, True], search_stats=stats)

    assert [s.filtered for s in stats] == [False, True]


def test_the_goal_object_retry_keeps_objects_the_robot_is_holding(monkeypatch):
    # holding the pan, the robot must be able to put it down to pick up the apple
    result, searched, *_ = _run_search_with_retry(monkeypatch, [False, True], held=["pan_28"])

    assert result[:2] == (["plan"], True)
    assert searched[1] == [
        "move robot1 countertop_1 fridge_5",
        "pick-left robot1 countertop countertop_1 apple_41",
        "pick-left robot1 countertop countertop_1 pan_28",
    ]

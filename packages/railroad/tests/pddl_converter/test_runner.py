from pathlib import Path

import pytest

from railroad.pddl_converter import load_problem, solve

DATA = Path(__file__).parent / "data"


def test_solve_blocks_reaches_goal():
    problem = load_problem(DATA / "blocks-domain.pddl", DATA / "blocks-instance.pddl")
    result = solve(problem, seed=0, max_iterations=2000)
    assert result.success
    # Building a 4-block tower from the table takes at least 6 actions,
    # and unit durations make sim_time == plan length.
    assert len(result.plan) >= 6
    assert result.sim_time == pytest.approx(len(result.plan))


def test_solve_gripper_costs_minimizes_total_cost():
    problem = load_problem(
        DATA / "gripper-costs-domain.pddl", DATA / "gripper-costs-instance.pddl"
    )
    result = solve(problem, seed=0, max_iterations=2000)
    assert result.success
    # Optimal: pick (1) + move (5) + drop (1). sim_time is total cost.
    assert result.plan == ["pick ball1 rooma", "move rooma roomb", "drop ball1 roomb"]
    assert result.sim_time == pytest.approx(7.0)


def test_solve_probabilistic_slippery():
    problem = load_problem(
        DATA / "slippery-domain.pddl", DATA / "slippery-instance.pddl"
    )
    result = solve(problem, seed=0, max_iterations=1500)
    assert result.success
    # Two deliveries plus one extra pickup per slip.
    assert len(result.plan) >= 4


def test_solve_reports_no_grounded_actions():
    """Static pruning can legitimately empty the action set.

    `road` is static and `:init` has no road facts, so every binding of
    `drive` is pruned at grounding — no private state poking needed.
    """
    from railroad.pddl_converter import convert_texts

    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (road ?a ?b) (at ?a))
      (:action drive :parameters (?a ?b)
               :precondition (and (at ?a) (road ?a ?b))
               :effect (and (not (at ?a)) (at ?b))))
    """
    problem_text = """
    (define (problem p) (:domain d) (:objects l1 l2)
      (:init (at l1)) (:goal (at l2)))
    """
    problem = convert_texts(domain, problem_text)
    assert problem.ground_actions() == []
    result = solve(problem, seed=0)
    assert not result.success
    assert result.failure_reason == "no grounded actions"


def test_solve_reports_an_unsatisfiable_goal_immediately():
    """A goal folded to FalseGoal is provably unreachable — say so and stop.

    The planner cannot tell us: it clamps h = inf to the dead-end penalty (0
    by default) and keeps proposing actions, so without the short-circuit the
    loop burns max_steps and then blames the step budget.
    """
    from railroad._bindings import GoalType
    from railroad.pddl_converter import convert_texts

    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (fixed ?l) (visited ?l) (here ?l))
      (:action go :parameters (?a ?b)
               :precondition (and (here ?a) (fixed ?b))
               :effect (and (not (here ?a)) (here ?b) (visited ?b))))
    """
    # `fixed` is static, and the goal asks for one :init never established.
    problem_text = """
    (define (problem p) (:domain d) (:objects l1 l2)
      (:init (here l1) (fixed l2))
      (:goal (and (visited l2) (fixed l1))))
    """
    problem = convert_texts(domain, problem_text)
    assert problem.goal.get_type() == GoalType.FALSE_GOAL

    result = solve(problem, seed=0, max_steps=25)
    assert not result.success
    assert result.failure_reason == "goal is unsatisfiable"
    assert result.plan == []
    assert result.sim_time == 0.0



def test_solve_gives_up_when_the_goal_is_unachievable():
    """An unreachable goal ends the run instead of erroring out."""
    from railroad.pddl_converter import convert_texts

    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (idled) (unreachable))
      (:action idle :parameters () :precondition () :effect (idled)))
    """
    problem_text = """
    (define (problem p) (:domain d) (:objects x) (:init) (:goal (unreachable)))
    """
    problem = convert_texts(domain, problem_text)
    assert problem.ground_actions()  # `idle` exists; it just cannot help
    result = solve(problem, seed=0, max_steps=5, max_iterations=200)
    assert not result.success
    assert result.failure_reason


def test_negated_goal_literals_reach_the_heuristic():
    """Negated goal literals must be compiled to `not-*` before evaluation.

    Without that rewriting the FF heuristic cannot tell an irreversibly
    broken state from an intact one, and every state scores alike. This pins
    the preprocessing at the heuristic level, independent of which planner
    consumes it -- MCTS itself clamps h = inf to
    HEURISTIC_CANNOT_FIND_GOAL_PENALTY, so it cannot act on the distinction
    (see the converter README on solving).
    """
    from railroad.core import get_action_by_name, transition
    from railroad.pddl_converter import convert_texts
    from railroad.planner import MCTSPlanner

    domain = """
    (define (domain fragile) (:requirements :strips :conditional-effects)
      (:predicates (holding ?x) (fragile ?x) (padded ?x) (delivered ?x) (broken ?x))
      (:action pad :parameters (?x) :precondition (holding ?x) :effect (padded ?x))
      (:action drop-off :parameters (?x)
         :precondition (holding ?x)
         :effect (and (not (holding ?x)) (delivered ?x)
                      (when (and (fragile ?x) (not (padded ?x))) (broken ?x)))))
    """
    problem_text = """
    (define (problem p) (:domain fragile) (:objects vase)
      (:init (holding vase) (fragile vase))
      (:goal (and (delivered vase) (not (broken vase)))))
    """
    problem = convert_texts(domain, problem_text)
    actions = problem.ground_actions()
    planner = MCTSPlanner(actions)

    intact = problem.initial_state
    assert planner.heuristic(intact, problem.goal) < float("inf")

    # Dropping the unpadded vase breaks it, and nothing can un-break it.
    broken, _ = transition(intact, get_action_by_name(actions, "drop-off vase"))[0]
    assert any(str(f) == "(broken vase)" for f in broken.fluents)
    assert planner.heuristic(broken, problem.goal) == float("inf")

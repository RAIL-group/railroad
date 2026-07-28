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


def test_solve_reports_unreachable_goal():
    problem = load_problem(DATA / "blocks-domain.pddl", DATA / "blocks-instance.pddl")
    # Empty the grounded action set to force failure.
    problem._actions = []
    result = solve(problem, seed=0)
    assert not result.success
    assert result.failure_reason == "no grounded actions"


def test_greedy_planner_solves_blocks():
    problem = load_problem(DATA / "blocks-domain.pddl", DATA / "blocks-instance.pddl")
    result = solve(problem, seed=0, planner="greedy")
    assert result.success
    assert len(result.plan) >= 6


def test_unknown_planner_rejected():
    problem = load_problem(DATA / "blocks-domain.pddl", DATA / "blocks-instance.pddl")
    with pytest.raises(ValueError):
        solve(problem, planner="astar")


def test_greedy_planner_handles_negative_goal():
    """Greedy heuristic evaluation must preprocess negated goal literals.

    Without the not-* rewriting the FF heuristic returns inf everywhere and
    greedy picks arbitrarily — here walking into the irreversible mistake of
    dropping the fragile vase unpadded.
    """
    from railroad.pddl_converter import convert_texts

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
    result = solve(problem, seed=0, planner="greedy")
    assert result.success
    assert result.plan == ["pad vase", "drop-off vase"]

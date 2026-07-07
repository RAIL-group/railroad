from pathlib import Path

import pytest

from railroad.core import Fluent as F
from railroad.pddl_converter import convert_texts, load_problem
from railroad.pddl_converter.converter import EPSILON_DURATION
from railroad.pddl_converter.errors import PDDLParseError, UnsupportedPDDLError

DATA = Path(__file__).parent / "data"


def _minimal_problem(objects: str = "x1 x2", goal: str = "(p x1)") -> str:
    return f"(define (problem p) (:domain d) (:objects {objects}) (:init) (:goal {goal}))"


def _get_action(problem, name):
    return next(a for a in problem.ground_actions() if a.name == name)


# ============================================================================
# Types and objects
# ============================================================================


def test_type_hierarchy_flattening():
    domain = """
    (define (domain d) (:requirements :strips :typing)
      (:types truck airplane - vehicle vehicle city - object)
      (:predicates (in ?v - vehicle ?c - city))
      (:action noop :parameters (?v - vehicle) :precondition (in ?v ?v)
               :effect (in ?v ?v)))
    """
    problem = """
    (define (problem p) (:domain d)
      (:objects t1 - truck a1 - airplane c1 - city)
      (:init) (:goal (in t1 c1)))
    """
    converted = convert_texts(domain, problem)
    assert converted.objects_by_type["truck"] == {"t1"}
    assert converted.objects_by_type["vehicle"] == {"t1", "a1"}
    assert converted.objects_by_type["city"] == {"c1"}
    assert converted.objects_by_type["object"] == {"t1", "a1", "c1"}


def test_domain_constants_are_objects():
    domain = """
    (define (domain d) (:requirements :strips)
      (:constants home)
      (:predicates (at ?x) (visited ?x))
      (:action go :parameters (?x) :precondition (at home) :effect (visited ?x)))
    """
    converted = convert_texts(domain, _minimal_problem(goal="(visited x1)"))
    assert "home" in converted.objects_by_type["object"]


# ============================================================================
# Synthetic agent and durations
# ============================================================================


def test_synthetic_agent_serializes_actions():
    converted = load_problem(DATA / "blocks-domain.pddl", DATA / "blocks-instance.pddl")
    assert converted.agent == "agent"
    assert F("free agent") in converted.initial_state.fluents
    action = _get_action(converted, "pick-up a")
    assert F("free agent") in action.preconditions
    assert len(action.effects) == 2
    start, finish = action.effects
    assert start.time == 0.0
    assert finish.time == 1.0  # unit duration: no metric
    assert F("holding a") in finish.resulting_fluents
    assert F("free agent") in finish.resulting_fluents


def test_agent_name_avoids_collision():
    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (p ?x)) (:action a :parameters (?x) :precondition (p ?x) :effect (p ?x)))
    """
    converted = convert_texts(domain, _minimal_problem(objects="agent x1"))
    assert converted.agent == "_agent"


def test_cost_metric_maps_to_duration():
    converted = load_problem(
        DATA / "gripper-costs-domain.pddl", DATA / "gripper-costs-instance.pddl"
    )
    move = _get_action(converted, "move rooma roomb")
    assert move.effects[1].time == 5.0  # function-valued (move-cost rooma roomb)
    pick = _get_action(converted, "pick ball1 rooma")
    assert pick.effects[1].time == 1.0  # constant cost


def test_zero_cost_gets_epsilon_duration():
    domain = """
    (define (domain d) (:requirements :strips :action-costs)
      (:predicates (p ?x)) (:functions (total-cost))
      (:action a :parameters (?x) :precondition () :effect (p ?x)))
    """
    problem = """
    (define (problem p) (:domain d) (:objects x1) (:init (= (total-cost) 0))
      (:goal (p x1)) (:metric minimize (total-cost)))
    """
    converted = convert_texts(domain, problem)
    action = _get_action(converted, "a x1")
    assert action.effects[1].time == pytest.approx(EPSILON_DURATION)


def test_unsupported_metric_rejected():
    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (p ?x)) (:action a :parameters (?x) :precondition () :effect (p ?x)))
    """
    problem = _minimal_problem().replace(
        "(:goal (p x1))", "(:goal (p x1)) (:metric maximize (reward))"
    )
    with pytest.raises(UnsupportedPDDLError) as excinfo:
        convert_texts(domain, problem)
    assert excinfo.value.reason == "metric:maximize (reward)"


def test_reward_effects_rejected():
    domain = """
    (define (domain d) (:requirements :strips :rewards)
      (:predicates (p ?x))
      (:action a :parameters (?x) :precondition ()
               :effect (and (p ?x) (increase (reward) 1))))
    """
    with pytest.raises(UnsupportedPDDLError) as excinfo:
        convert_texts(domain, _minimal_problem())
    assert excinfo.value.reason == "rewards"


# ============================================================================
# Grounding: equality, duplicates, static pruning
# ============================================================================


def test_duplicate_bindings_allowed_without_inequality():
    converted = load_problem(DATA / "blocks-domain.pddl", DATA / "blocks-instance.pddl")
    names = {a.name for a in converted.ground_actions()}
    assert "stack a a" in names  # PDDL permits it (statically unreachable)


def test_inequality_constraint_filters_bindings():
    domain = """
    (define (domain d) (:requirements :strips :equality)
      (:predicates (p ?x ?y))
      (:action a :parameters (?x ?y) :precondition (not (= ?x ?y))
               :effect (p ?x ?y)))
    """
    converted = convert_texts(domain, _minimal_problem(goal="(p x1 x2)"))
    names = {a.name for a in converted.ground_actions()}
    assert names == {"a x1 x2", "a x2 x1"}


def test_static_preconditions_prune_grounding():
    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (road ?a ?b) (at ?a))
      (:action drive :parameters (?a ?b)
               :precondition (and (at ?a) (road ?a ?b))
               :effect (and (not (at ?a)) (at ?b))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects l1 l2 l3)
      (:init (at l1) (road l1 l2) (road l2 l3))
      (:goal (at l3)))
    """
    converted = convert_texts(domain, problem)
    names = {a.name for a in converted.ground_actions()}
    assert names == {"drive l1 l2", "drive l2 l3"}  # road is static


# ============================================================================
# Quantifiers
# ============================================================================


def test_forall_precondition_expands():
    domain = """
    (define (domain d) (:requirements :strips :universal-preconditions)
      (:predicates (done ?x) (all-done))
      (:action finish :parameters ()
               :precondition (forall (?x) (done ?x))
               :effect (all-done))
      (:action mark :parameters (?x) :precondition () :effect (done ?x)))
    """
    converted = convert_texts(domain, _minimal_problem(goal="(all-done)"))
    action = _get_action(converted, "finish")
    assert F("done x1") in action.preconditions
    assert F("done x2") in action.preconditions


def test_forall_effect_expands():
    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (wiped ?x) (go))
      (:action wipe-all :parameters ()
               :precondition (go)
               :effect (forall (?x) (wiped ?x))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects x1 x2) (:init (go))
      (:goal (wiped x1)))
    """
    converted = convert_texts(domain, problem)
    action = _get_action(converted, "wipe-all")
    assert F("wiped x1") in action.effects[1].resulting_fluents
    assert F("wiped x2") in action.effects[1].resulting_fluents


def test_exists_precondition_lifts_parameter():
    domain = """
    (define (domain d) (:requirements :strips :existential-preconditions)
      (:predicates (key ?k) (has ?k) (open ?door))
      (:action open-door :parameters (?door)
               :precondition (exists (?k) (and (key ?k) (has ?k)))
               :effect (open ?door)))
    """
    problem = """
    (define (problem p) (:domain d) (:objects door1 k1 k2)
      (:init (key k1) (key k2) (has k2))
      (:goal (open door1)))
    """
    converted = convert_texts(domain, problem)
    op = converted.compiled_operators[0].operator
    assert len(op.parameters) == 2  # ?door plus the lifted witness
    # `key` and `has` are both static, so only the k2 witness survives.
    # (?door is untyped, so all objects still bind it.)
    names = {a.name for a in converted.ground_actions()}
    assert "open-door door1 k2" in names
    assert "open-door door1 k1" not in names


def test_quantified_goal_compiles_to_goal_tree():
    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (done ?x))
      (:action do :parameters (?x) :precondition () :effect (done ?x)))
    """
    converted = convert_texts(domain, _minimal_problem(goal="(forall (?x) (done ?x))"))
    assert not converted.goal.evaluate({F("done x1")})
    assert converted.goal.evaluate({F("done x1"), F("done x2")})

    converted = convert_texts(domain, _minimal_problem(goal="(exists (?x) (done ?x))"))
    assert converted.goal.evaluate({F("done x2")})
    assert not converted.goal.evaluate(set())


def test_disjunctive_and_negated_goals():
    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (p ?x) (q ?x))
      (:action do :parameters (?x) :precondition () :effect (p ?x)))
    """
    converted = convert_texts(domain, _minimal_problem(goal="(or (p x1) (q x1))"))
    assert converted.goal.evaluate({F("q x1")})
    converted = convert_texts(domain, _minimal_problem(goal="(and (p x1) (not (q x1)))"))
    assert converted.goal.evaluate({F("p x1")})
    assert not converted.goal.evaluate({F("p x1"), F("q x1")})


def test_disjunctive_precondition_rejected():
    domain = """
    (define (domain d) (:requirements :strips :disjunctive-preconditions)
      (:predicates (p ?x) (q ?x))
      (:action a :parameters (?x) :precondition (or (p ?x) (q ?x))
               :effect (p ?x)))
    """
    with pytest.raises(UnsupportedPDDLError) as excinfo:
        convert_texts(domain, _minimal_problem())
    assert excinfo.value.reason == "disjunctive-preconditions"


# ============================================================================
# Probabilistic effects
# ============================================================================


def test_probabilistic_effect_structure():
    converted = load_problem(
        DATA / "slippery-domain.pddl", DATA / "slippery-instance.pddl"
    )
    pickup = _get_action(converted, "pickup b1")
    finish = pickup.effects[1]
    assert len(finish.prob_effects) == 2
    probs = [branch.prob for branch in finish.prob_effects]
    assert probs == pytest.approx([0.7, 0.3])
    success_fluents = set(finish.prob_effects[0].effects[0].resulting_fluents)
    assert F("holding b1") in success_fluents


def test_probabilistic_remainder_branch_added():
    domain = """
    (define (domain d) (:requirements :strips :probabilistic-effects)
      (:predicates (p ?x) (go))
      (:action a :parameters (?x) :precondition (go)
               :effect (probabilistic 0.4 (p ?x))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects x1) (:init (go)) (:goal (p x1)))
    """
    converted = convert_texts(domain, problem)
    action = _get_action(converted, "a x1")
    branches = action.effects[1].prob_effects
    assert [b.prob for b in branches] == pytest.approx([0.4, 0.6])
    assert list(branches[1].effects) == []  # implicit no-op remainder


def test_probability_sum_over_one_rejected():
    domain = """
    (define (domain d) (:requirements :strips :probabilistic-effects)
      (:predicates (p ?x))
      (:action a :parameters (?x) :precondition ()
               :effect (probabilistic 0.8 (p ?x) 0.7 (not (p ?x)))))
    """
    with pytest.raises(PDDLParseError):
        convert_texts(domain, _minimal_problem())


def test_cost_inside_probabilistic_branch_rejected():
    domain = """
    (define (domain d) (:requirements :strips :probabilistic-effects :action-costs)
      (:predicates (p ?x)) (:functions (total-cost))
      (:action a :parameters (?x) :precondition ()
               :effect (probabilistic 0.5 (increase (total-cost) 2))))
    """
    with pytest.raises(UnsupportedPDDLError) as excinfo:
        convert_texts(domain, _minimal_problem())
    assert excinfo.value.reason == "probabilistic-cost"


# ============================================================================
# Reserved names and validation
# ============================================================================


def test_reserved_predicates_renamed():
    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (free ?x) (waiting ?x) (not-here ?x) (used ?x))
      (:action use :parameters (?x) :precondition (free ?x)
               :effect (and (used ?x) (not (free ?x)))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects x1)
      (:init (free x1) (waiting x1) (not-here x1)) (:goal (used x1)))
    """
    converted = convert_texts(domain, problem)
    fluents = converted.initial_state.fluents
    assert F("pddl-free x1") in fluents
    assert F("pddl-waiting x1") in fluents
    assert F("pddl-not-here x1") in fluents
    # The only real `free` fluent is the synthetic agent's.
    free_fluents = [f for f in fluents if f.name == "free"]
    assert free_fluents == [F("free agent")]
    action = _get_action(converted, "use x1")
    assert F("pddl-free x1") in action.preconditions


def test_unbound_variable_rejected():
    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (p ?x ?y))
      (:action a :parameters (?x) :precondition () :effect (p ?x ?z)))
    """
    with pytest.raises(PDDLParseError):
        convert_texts(domain, _minimal_problem())


def test_domain_problem_mismatch_rejected():
    domain = "(define (domain d1) (:predicates (p)) (:action a :parameters () :precondition () :effect (p)))"
    with pytest.raises(PDDLParseError):
        convert_texts(domain, _minimal_problem().replace("(:domain d)", "(:domain other)"))


def test_bundled_domain_and_problem_in_one_text():
    """IPPC-2008 style: one file holding both defines converts cleanly."""
    bundled = """
    (define (domain d)
      (:requirements :strips)
      (:predicates (p ?x))
      (:action a :parameters (?x) :precondition () :effect (p ?x)))
    (define (problem prob-1)
      (:domain d) (:objects x1) (:init) (:goal (p x1)))
    """
    converted = convert_texts(bundled, bundled)
    assert converted.domain_name == "d"
    assert converted.problem_name == "prob-1"


def test_goal_reward_maximize_metric_reinterpreted():
    domain = """
    (define (domain d) (:requirements :strips :probabilistic-effects :rewards)
      (:predicates (p ?x))
      (:action a :parameters (?x) :precondition () :effect (p ?x)))
    """
    problem = """
    (define (problem p) (:domain d) (:objects x1) (:init)
      (:goal (p x1)) (:goal-reward 100) (:metric maximize (reward)))
    """
    converted = convert_texts(domain, problem)
    assert "reach-goal" in converted.metric
    assert _get_action(converted, "a x1").effects[1].time == 1.0


def test_self_loop_grounding_preserves_fluent():
    """A binding with ?from == ?to must not destroy the location fluent.

    The core applies deletes before adds (PDDL semantics), so an effect that
    both deletes and adds `at plane apt` leaves the fluent present.
    """
    from railroad.core import transition

    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (at ?x ?l))
      (:action fly :parameters (?x ?from ?to)
               :precondition (at ?x ?from)
               :effect (and (not (at ?x ?from)) (at ?x ?to))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects plane apt)
      (:init (at plane apt)) (:goal (at plane apt)))
    """
    converted = convert_texts(domain, problem)
    self_loop = _get_action(converted, "fly plane apt apt")
    (successor, prob), = transition(converted.initial_state, self_loop)
    assert prob == pytest.approx(1.0)
    assert F("at plane apt") in successor.fluents

from pathlib import Path

import pytest

from railroad._bindings import GoalType
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
    # Both predicates are dynamic, so goal compilation is exercised on its own.
    # A *static* goal predicate is folded against the initial state instead —
    # see test_static_goal_literals_are_folded_against_the_initial_state.
    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (p ?x) (q ?x))
      (:action do :parameters (?x) :precondition () :effect (p ?x))
      (:action doq :parameters (?x) :precondition () :effect (q ?x)))
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
    # `free` is dynamic (the action deletes it), so its renamed form stays in
    # the runtime state and preconditions. The only real `free` fluent is the
    # synthetic agent's.
    assert F("pddl-free x1") in fluents
    free_fluents = [f for f in fluents if f.name == "free"]
    assert free_fluents == [F("free agent")]
    action = _get_action(converted, "use x1")
    assert F("pddl-free x1") in action.preconditions
    # `waiting`/`not-here` are static and goal-irrelevant, so grounding
    # eliminates them entirely — and the reserved names never leak through.
    assert not any(f.name in ("waiting", "not-here") for f in fluents)


def test_reserved_set_tracks_the_core_list():
    """`at`/`found` are core-reserved too, and must be renamed with the rest.

    The FF heuristic's at-implies-found rule turns a required `at X L` into a
    required `found X` whenever `found X` is relaxed-reachable. A PDDL domain
    carrying both predicates would silently get distorted heuristic values;
    renaming `found` makes the rule unreachable, and `at` follows so this set
    stays a straight copy of the core's rather than a drifting carve-out.
    """
    from railroad.core import RESERVED_PLANNING_PREDICATES
    from railroad.pddl_converter.converter import _RESERVED_PREDICATES

    assert RESERVED_PLANNING_PREDICATES <= _RESERVED_PREDICATES

    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (at ?x ?l) (found ?x))
      (:action look :parameters (?x ?l) :precondition (at ?x ?l)
               :effect (found ?x))
      (:action move :parameters (?x ?from ?to) :precondition (at ?x ?from)
               :effect (and (not (at ?x ?from)) (at ?x ?to))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects k table shelf)
      (:init (at k table)) (:goal (found k)))
    """
    converted = convert_texts(domain, problem)
    fluents = converted.initial_state.fluents
    assert F("pddl-at k table") in fluents  # `at` is dynamic here, so it survives
    assert [f for f in fluents if f.name in ("at", "found")] == []
    assert {f.name for f in converted.goal.get_all_literals()} == {"pddl-found"}
    look = _get_action(converted, "look k table")
    assert F("pddl-at k table") in look.preconditions


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
    # `at` is one of the core's reserved names, so it is renamed (see
    # test_reserved_predicates_renamed).
    assert F("pddl-at plane apt") in successor.fluents


# ============================================================================
# Conditional effects
# ============================================================================


def _apply(converted, action_name):
    from railroad.core import transition

    action = _get_action(converted, action_name)
    ((successor, prob),) = transition(converted.initial_state, action)
    assert prob == pytest.approx(1.0)
    return successor


def test_conditional_effect_fires_when_condition_holds():
    domain = """
    (define (domain d) (:requirements :strips :conditional-effects)
      (:predicates (fragile ?x) (dropped ?x) (broken ?x))
      (:action drop :parameters (?x)
               :precondition ()
               :effect (and (dropped ?x) (when (fragile ?x) (broken ?x)))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects vase brick)
      (:init (fragile vase)) (:goal (dropped brick)))
    """
    converted = convert_texts(domain, problem)
    after_vase = _apply(converted, "drop vase")
    assert F("dropped vase") in after_vase.fluents
    assert F("broken vase") in after_vase.fluents
    after_brick = _apply(converted, "drop brick")
    assert F("dropped brick") in after_brick.fluents
    assert F("broken brick") not in after_brick.fluents


def test_conditional_effect_reads_pre_action_state():
    """Conditions are evaluated before the action's own effects apply."""
    domain = """
    (define (domain d) (:requirements :strips :conditional-effects)
      (:predicates (p) (q) (r))
      (:action a :parameters ()
               :precondition ()
               :effect (and (not (p)) (q) (when (p) (r)))))
    """
    problem = """
    (define (problem p1) (:domain d) (:objects x)
      (:init (p)) (:goal (q)))
    """
    converted = convert_texts(domain, problem)
    successor = _apply(converted, "a")
    # PDDL: the (when (p) ...) condition saw the pre-state where p held,
    # even though the same action deletes p.
    assert F("p") not in successor.fluents
    assert F("r") in successor.fluents


def test_conditional_effect_negative_condition():
    domain = """
    (define (domain d) (:requirements :strips :conditional-effects)
      (:predicates (locked ?x) (opened ?x) (tried ?x))
      (:action try-open :parameters (?x)
               :precondition ()
               :effect (and (tried ?x) (when (not (locked ?x)) (opened ?x)))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects door1 door2)
      (:init (locked door1)) (:goal (tried door1)))
    """
    converted = convert_texts(domain, problem)
    assert F("opened door1") not in _apply(converted, "try-open door1").fluents
    assert F("opened door2") in _apply(converted, "try-open door2").fluents


def test_forall_when_expands_per_object():
    domain = """
    (define (domain d) (:requirements :strips :conditional-effects)
      (:predicates (boarded ?p) (at-dest ?p) (moved))
      (:action move :parameters ()
               :precondition ()
               :effect (and (moved)
                            (forall (?p) (when (boarded ?p) (at-dest ?p))))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects alice bob carol)
      (:init (boarded alice) (boarded carol)) (:goal (moved)))
    """
    converted = convert_texts(domain, problem)
    successor = _apply(converted, "move")
    assert F("at-dest alice") in successor.fluents
    assert F("at-dest carol") in successor.fluents
    assert F("at-dest bob") not in successor.fluents


def test_when_inside_probabilistic_branch():
    domain = """
    (define (domain d)
      (:requirements :strips :probabilistic-effects :conditional-effects)
      (:predicates (armed) (fired) (hit))
      (:action fire :parameters ()
               :precondition ()
               :effect (and (fired)
                            (probabilistic 1.0 (when (armed) (hit))))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects x)
      (:init (armed)) (:goal (fired)))
    """
    converted = convert_texts(domain, problem)
    successor = _apply(converted, "fire")
    assert F("hit") in successor.fluents


def test_unsupported_when_condition_rejected():
    domain = """
    (define (domain d) (:requirements :strips :conditional-effects)
      (:predicates (p ?x) (q ?x) (r ?x))
      (:action a :parameters (?x)
               :precondition ()
               :effect (when (or (p ?x) (q ?x)) (r ?x))))
    """
    with pytest.raises(UnsupportedPDDLError) as excinfo:
        convert_texts(domain, _minimal_problem())
    assert excinfo.value.reason == "conditional-effect-condition"


def test_cost_inside_conditional_rejected():
    domain = """
    (define (domain d) (:requirements :strips :conditional-effects :action-costs)
      (:predicates (p ?x)) (:functions (total-cost))
      (:action a :parameters (?x) :precondition ()
               :effect (when (p ?x) (increase (total-cost) 2))))
    """
    with pytest.raises(UnsupportedPDDLError) as excinfo:
        convert_texts(domain, _minimal_problem())
    assert excinfo.value.reason == "conditional-cost"


def test_conditional_effects_visible_to_heuristic():
    """The relaxed heuristic optimistically assumes conditions hold, so a
    goal reachable only through a conditional effect gets a finite h."""
    from railroad.core import ff_heuristic

    domain = """
    (define (domain d) (:requirements :strips :conditional-effects)
      (:predicates (fragile ?x) (dropped ?x) (broken ?x))
      (:action drop :parameters (?x)
               :precondition ()
               :effect (and (dropped ?x) (when (fragile ?x) (broken ?x)))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects vase)
      (:init (fragile vase)) (:goal (broken vase)))
    """
    converted = convert_texts(domain, problem)
    h = ff_heuristic(
        converted.initial_state, converted.goal, converted.ground_actions()
    )
    assert h < float("inf")


def test_equality_in_when_condition():
    """(= ?x ?y) in a when-condition resolves via the seeded eq fluents."""
    domain = """
    (define (domain d) (:requirements :strips :conditional-effects :equality)
      (:predicates (painted ?x ?c) (repaint ?x ?c ?c2))
      (:action paint :parameters (?x ?old ?new)
               :precondition ()
               :effect (and (painted ?x ?new)
                            (when (not (= ?old ?new)) (not (painted ?x ?old))))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects obj red blue)
      (:init (painted obj red)) (:goal (painted obj blue)))
    """
    converted = convert_texts(domain, problem)
    # Equality is evaluated away at grounding; no pddl-eq fluents at runtime.
    assert not any(f.name == "pddl-eq" for f in converted.initial_state.fluents)

    changed = _apply(converted, "paint obj red blue")
    assert F("painted obj blue") in changed.fluents
    assert F("painted obj red") not in changed.fluents  # condition fired

    same = _apply(converted, "paint obj red red")
    assert F("painted obj red") in same.fluents  # condition did not fire


# ============================================================================
# Static elimination and the goal
# ============================================================================


def test_static_goal_literals_are_folded_against_the_initial_state():
    """A static goal literal is decided at compile time, not carried at runtime.

    `road` is static, so `(road l2 l3)` can never change truth value. Folding
    it leaves `(visited l2)` as the whole goal, and the fact no longer has to
    be kept in the state for the goal to be satisfiable.
    """
    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (road ?a ?b) (scenic ?a) (at ?a) (visited ?a))
      (:action drive :parameters (?a ?b)
               :precondition (and (at ?a) (road ?a ?b))
               :effect (and (not (at ?a)) (at ?b) (visited ?b))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects l1 l2 l3)
      (:init (at l1) (road l1 l2) (road l2 l3) (scenic l3))
      (:goal (and (visited l2) (road l2 l3))))
    """
    converted = convert_texts(domain, problem)

    # Static in the preconditions, so grounding strips it there...
    drive = _get_action(converted, "drive l1 l2")
    assert not any(f.name == "road" for f in drive.preconditions)
    # ...and static in the goal, so it is folded to True and the conjunction
    # collapses to the one dynamic literal.
    assert converted.goal.evaluate({F("visited l2")})
    # Nothing reads `road` or `scenic` at runtime any more, so neither is kept.
    assert not any(f.name in ("road", "scenic")
                   for f in converted.initial_state.fluents)

    assert not converted.goal.evaluate(converted.initial_state.fluents)
    from railroad.pddl_converter import solve
    assert solve(converted, seed=0).success


def test_unsatisfiable_static_goal_literal_makes_the_goal_false():
    """The other side of folding: a static literal that is false at t=0 can
    never become true, so the whole conjunction is unsatisfiable."""
    domain = """
    (define (domain d) (:requirements :strips)
      (:predicates (road ?a ?b) (at ?a) (visited ?a))
      (:action drive :parameters (?a ?b)
               :precondition (and (at ?a) (road ?a ?b))
               :effect (and (not (at ?a)) (at ?b) (visited ?b))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects l1 l2)
      (:init (at l1) (road l1 l2))
      (:goal (and (visited l2) (road l2 l1))))
    """
    converted = convert_texts(domain, problem)
    assert converted.goal.get_type() == GoalType.FALSE_GOAL
    assert not converted.goal.evaluate({F("visited l2"), F("road l2 l1")})


def test_reserved_predicates_renamed_without_a_predicates_section():
    """Renaming keys on predicates used, not only on those declared.

    `(:predicates ...)` is conventional but the parser does not require it;
    an undeclared `free` left unrenamed would be handed to the core's
    concurrency machinery as a schedulable agent.
    """
    domain = """
    (define (domain d) (:requirements :strips)
      (:action use :parameters (?x) :precondition (free ?x)
               :effect (and (used ?x) (not (free ?x)))))
    """
    problem = """
    (define (problem p) (:domain d) (:objects x1) (:init (free x1)) (:goal (used x1)))
    """
    converted = convert_texts(domain, problem)
    action = _get_action(converted, "use x1")
    assert F("pddl-free x1") in action.preconditions
    # The only real `free` is the synthetic agent's.
    free_fluents = [f for f in converted.initial_state.fluents if f.name == "free"]
    assert free_fluents == [F(f"free {converted.agent}")]


def test_eq_conditions_are_seeded_through_forall_effects():
    """`_uses_eq_conditions` must see ForallEffect branches too.

    The converter expands `forall` itself, but the helper walks the public
    Effect type; missing the branch would leave `(= ...)` unseeded and every
    such condition would silently evaluate false.
    """
    from railroad.core import Effect, ForallEffect
    from railroad.pddl_converter.converter import EQ_PREDICATE, _uses_eq_conditions

    effect = Effect(
        time=1.0,
        forall_effects=[ForallEffect(
            variables=[("?y", "item")],
            conditions={F(f"{EQ_PREDICATE} ?x ?y")},
            effects=[Effect(time=0, resulting_fluents={F("marked ?y")})],
        )],
    )
    assert _uses_eq_conditions([effect])

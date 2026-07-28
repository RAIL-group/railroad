"""TEMPORARY dual-run parity harness (design doc §7 item 1).

Grounds problems through both the old converter-private grounder and the new
core ``ground_operators`` and asserts structural equality. Deleted together
with the old grounder once green (including the network-gated IPC sweep).
"""

import os
from pathlib import Path

import pytest

from railroad.pddl_converter import convert_texts, load_problem
from railroad.pddl_converter.converter import _ground_operator
from railroad.pddl_converter.errors import PDDLParseError, UnsupportedPDDLError

DATA = Path(__file__).parent / "data"


def _old_ground(converted):
    actions = []
    for comp in converted.compiled_operators:
        actions.extend(
            _ground_operator(comp, converted.objects_by_type, converted._static_fluents)
        )
    return actions


def _assert_structurally_equal(converted):
    old = {a.name: a for a in _old_ground(converted)}
    new = {a.name: a for a in converted.ground_actions()}
    assert set(old) == set(new), (
        f"action name sets differ: only-old={sorted(set(old) - set(new))[:5]} "
        f"only-new={sorted(set(new) - set(old))[:5]}"
    )
    for name, old_action in old.items():
        new_action = new[name]
        assert set(old_action.preconditions) == set(new_action.preconditions), name
        assert old_action.extra_cost == new_action.extra_cost, name
        assert len(old_action.effects) == len(new_action.effects), name
        for old_eff, new_eff in zip(old_action.effects, new_action.effects):
            assert old_eff.time == pytest.approx(new_eff.time), name
            # GroundedEffect equality is hash-based and covers resulting
            # fluents plus probabilistic and conditional branches.
            assert old_eff == new_eff, name


# Inline feature-matrix problems (mirroring the converter test suite).
_FEATURE_PROBLEMS = {
    "inequality": (
        """
        (define (domain d) (:requirements :strips :equality)
          (:predicates (p ?x ?y))
          (:action a :parameters (?x ?y) :precondition (not (= ?x ?y))
                   :effect (p ?x ?y)))
        """,
        "(define (problem p) (:domain d) (:objects x1 x2 x3) (:init) (:goal (p x1 x2)))",
    ),
    "static-prune": (
        """
        (define (domain d) (:requirements :strips)
          (:predicates (road ?a ?b) (at ?a))
          (:action drive :parameters (?a ?b)
                   :precondition (and (at ?a) (road ?a ?b))
                   :effect (and (not (at ?a)) (at ?b))))
        """,
        """
        (define (problem p) (:domain d) (:objects l1 l2 l3)
          (:init (at l1) (road l1 l2) (road l2 l3))
          (:goal (at l3)))
        """,
    ),
    "equality-when": (
        """
        (define (domain d) (:requirements :strips :conditional-effects :equality)
          (:predicates (painted ?x ?c))
          (:action paint :parameters (?x ?old ?new)
                   :precondition ()
                   :effect (and (painted ?x ?new)
                                (when (not (= ?old ?new)) (not (painted ?x ?old))))))
        """,
        """
        (define (problem p) (:domain d) (:objects obj red blue)
          (:init (painted obj red)) (:goal (painted obj blue)))
        """,
    ),
    "forall-when": (
        """
        (define (domain d) (:requirements :strips :conditional-effects)
          (:predicates (boarded ?p) (at-dest ?p) (moved))
          (:action move :parameters ()
                   :precondition ()
                   :effect (and (moved)
                                (forall (?p) (when (boarded ?p) (at-dest ?p))))))
        """,
        """
        (define (problem p) (:domain d) (:objects alice bob)
          (:init (boarded alice)) (:goal (moved)))
        """,
    ),
    "exists-lift": (
        """
        (define (domain d) (:requirements :strips :existential-preconditions)
          (:predicates (key ?k) (has ?k) (open ?door))
          (:action open-door :parameters (?door)
                   :precondition (exists (?k) (and (key ?k) (has ?k)))
                   :effect (open ?door)))
        """,
        """
        (define (problem p) (:domain d) (:objects door1 k1 k2)
          (:init (key k1) (key k2) (has k2))
          (:goal (open door1)))
        """,
    ),
    "undefined-cost": (
        """
        (define (domain d) (:requirements :strips :action-costs)
          (:predicates (at ?a) (visited ?b))
          (:functions (total-cost) (dist ?a ?b))
          (:action go :parameters (?a ?b)
                   :precondition (at ?a)
                   :effect (and (visited ?b)
                                (increase (total-cost) (dist ?a ?b)))))
        """,
        """
        (define (problem p) (:domain d) (:objects l1 l2 l3)
          (:init (= (total-cost) 0) (at l1) (= (dist l1 l2) 4))
          (:goal (visited l2)) (:metric minimize (total-cost)))
        """,
    ),
}


@pytest.mark.parametrize("feature", sorted(_FEATURE_PROBLEMS))
def test_parity_feature_matrix(feature):
    domain, problem = _FEATURE_PROBLEMS[feature]
    _assert_structurally_equal(convert_texts(domain, problem))


@pytest.mark.parametrize(
    "domain_file, instance_file",
    [
        ("blocks-domain.pddl", "blocks-instance.pddl"),
        ("gripper-costs-domain.pddl", "gripper-costs-instance.pddl"),
        ("slippery-domain.pddl", "slippery-instance.pddl"),
    ],
)
def test_parity_vendored(domain_file, instance_file):
    _assert_structurally_equal(load_problem(DATA / domain_file, DATA / instance_file))


@pytest.mark.skipif(
    not os.environ.get("RAILROAD_PDDL_NETWORK_TESTS"),
    reason="set RAILROAD_PDDL_NETWORK_TESTS=1 to sweep IPC collections",
)
@pytest.mark.parametrize("collection", ["ipc-2000", "ippc-2006", "ippc-2008"])
def test_parity_ipc_collection(collection):
    from railroad.pddl_converter import fetch_domain, list_domains

    checked = 0
    for domain_name in list_domains(collection):
        try:
            fetched = fetch_domain(collection, domain_name, max_instances=1)
            if not fetched.instances:
                continue
            instance = fetched.instances[0]
            converted = load_problem(fetched.domain_for(instance), instance)
        except (UnsupportedPDDLError, PDDLParseError):
            continue
        except Exception:
            continue  # fetch errors etc. — the check table covers status parity
        _assert_structurally_equal(converted)
        checked += 1
    assert checked > 0

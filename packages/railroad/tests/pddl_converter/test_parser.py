from pathlib import Path

import pytest

from railroad.pddl_converter.errors import PDDLParseError, UnsupportedPDDLError
from railroad.pddl_converter.parser import (
    And,
    EffectAnd,
    Equals,
    Forall,
    Literal,
    Probabilistic,
    When,
    parse_domain,
    parse_problem,
    read_sexprs,
    tokenize,
)

DATA = Path(__file__).parent / "data"


def _domain_with(body: str) -> str:
    return f"(define (domain d) (:requirements :strips) {body})"


def _action_with(precondition: str = "(p)", effect: str = "(q)") -> str:
    return _domain_with(
        f"(:action a :parameters () :precondition {precondition} :effect {effect})"
    )


# ============================================================================
# Tokenizer / reader
# ============================================================================


def test_tokenize_strips_comments_and_lowercases():
    tokens = tokenize("(ON A B) ; a comment (not parsed)\n(clear C)")
    assert tokens == ["(", "on", "a", "b", ")", "(", "clear", "c", ")"]


def test_read_sexprs_nested():
    assert read_sexprs("(a (b c) d)") == [["a", ["b", "c"], "d"]]


@pytest.mark.parametrize("text", ["(a (b)", "(a)) ("])
def test_read_sexprs_unbalanced(text):
    with pytest.raises(PDDLParseError):
        read_sexprs(text)


# ============================================================================
# Domain parsing
# ============================================================================


def test_parse_blocks_domain():
    domain = parse_domain((DATA / "blocks-domain.pddl").read_text())
    assert domain.name == "blocks"
    assert ":typing" in domain.requirements
    assert domain.types == {"block": "object"}
    assert domain.predicates == {
        "on": 2, "ontable": 1, "clear": 1, "handempty": 0, "holding": 1,
    }
    assert [a.name for a in domain.actions] == [
        "pick-up", "put-down", "stack", "unstack",
    ]
    stack = domain.actions[2]
    assert stack.parameters == [("?x", "block"), ("?y", "block")]
    assert isinstance(stack.precondition, And)
    assert Literal("holding", ("?x",)) in stack.precondition.children


def test_parse_typed_list_defaults_to_object():
    domain = parse_domain(_domain_with("(:predicates (p ?a ?b - t ?c))"))
    assert domain.predicates == {"p": 3}


def test_parse_functions_section():
    domain = parse_domain(
        _domain_with("(:functions (total-cost) - number (road ?a ?b - loc))")
    )
    assert domain.functions == {"total-cost": 0, "road": 2}


def test_parse_probabilistic_effect_with_rational():
    domain = parse_domain((DATA / "slippery-domain.pddl").read_text())
    effect = domain.actions[0].effect
    assert isinstance(effect, EffectAnd)
    prob_nodes = [c for c in effect.children if isinstance(c, Probabilistic)]
    assert len(prob_nodes) == 1
    probs = [p for p, _ in prob_nodes[0].branches]
    assert probs == pytest.approx([0.7, 0.3])


def test_parse_quantifiers_and_equality():
    domain = parse_domain(
        _action_with(
            precondition="(and (forall (?y - t) (p ?y)) (not (= ?x ?y)))",
        )
    )
    precondition = domain.actions[0].precondition
    assert isinstance(precondition, And)
    assert isinstance(precondition.children[0], Forall)
    assert precondition.children[1] == Equals("?x", "?y", negated=True)


@pytest.mark.parametrize(
    "body, reason",
    [
        ("(:durative-action a :parameters ())", "durative-actions"),
        ("(:derived (p) (q))", "derived-predicates"),
        ("(:constraints (and))", "constraints"),
        ("(:types a - (either b c))", "either-types"),
    ],
)
def test_unsupported_domain_sections(body, reason):
    with pytest.raises(UnsupportedPDDLError) as excinfo:
        parse_domain(_domain_with(body))
    assert excinfo.value.reason == reason


@pytest.mark.parametrize(
    "precondition, reason",
    [
        ("(imply (p) (q))", "imply-conditions"),
        ("(preference pref1 (p))", "preferences"),
        ("(< (f) 3)", "numeric-conditions"),
        ("(= (f) 3)", "numeric-conditions"),
        ("(not (and (p) (q)))", "negated-compound-condition"),
    ],
)
def test_unsupported_preconditions(precondition, reason):
    with pytest.raises(UnsupportedPDDLError) as excinfo:
        parse_domain(_action_with(precondition=precondition))
    assert excinfo.value.reason == reason


def test_parse_conditional_effect():
    domain = parse_domain(_action_with(effect="(when (p) (and (q) (not (r))))"))
    when = domain.actions[0].effect
    assert isinstance(when, When)
    assert when.condition == Literal("p", ())
    assert isinstance(when.effect, EffectAnd)


@pytest.mark.parametrize(
    "effect, reason",
    [
        ("(oneof (p) (q))", "oneof-nondeterminism"),
        ("(assign (f) 3)", "numeric-effects"),
        ("(decrease (f) 3)", "numeric-effects"),
        ("(increase (f) (+ (f) 1))", "numeric-effects"),
    ],
)
def test_unsupported_effects(effect, reason):
    with pytest.raises(UnsupportedPDDLError) as excinfo:
        parse_domain(_action_with(effect=effect))
    assert excinfo.value.reason == reason


# ============================================================================
# Problem parsing
# ============================================================================


def test_parse_problem_basics():
    problem = parse_problem((DATA / "gripper-costs-instance.pddl").read_text())
    assert problem.name == "gripper-costs-1"
    assert problem.domain_name == "gripper-costs"
    assert problem.objects == {"rooma": "room", "roomb": "room", "ball1": "ball"}
    assert Literal("at-robby", ("rooma",)) in problem.init_literals
    assert problem.init_function_values[("move-cost", ("rooma", "roomb"))] == 5.0
    assert problem.init_function_values[("total-cost", ())] == 0.0
    assert problem.metric == ("minimize", "(total-cost)")


def test_parse_problem_requires_goal():
    with pytest.raises(UnsupportedPDDLError) as excinfo:
        parse_problem("(define (problem p) (:domain d) (:init (a)))")
    assert excinfo.value.reason == "no-goal"


def test_parse_problem_negative_init_ignored():
    problem = parse_problem(
        "(define (problem p) (:domain d) (:init (a) (not (b))) (:goal (a)))"
    )
    assert problem.init_literals == [Literal("a", ())]

"""Tests for ground_operators: static-precondition grounding in core.

Per design doc §7.1, tests here assert behavior (which actions exist, what
plans do) rather than grounded-action structure, except where structure is
the subject — those pin the relevant flags explicitly.
"""

import pytest

from railroad._bindings import Fluent as F, State
from railroad.core import Effect, Eq, ForallEffect, Neq, Operator, ground_operators

ROBOTS = {"robot": {"r1"}}


def _move_op(preconditions_extra=(), time=5.0):
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r"), *preconditions_extra],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
            Effect(time=time, resulting_fluents={F("free ?r"), F("at ?r ?to")}),
        ],
    )


def test_static_precondition_prunes_grounding():
    """Only connected moves exist; `connected` is inferred static."""
    op = _move_op(preconditions_extra=[F("connected ?from ?to")])
    result = ground_operators(
        [op],
        {**ROBOTS, "location": {"a", "b", "c"}},
        {F("at r1 a"), F("free r1"), F("connected a b"), F("connected b c")},
    )
    assert result.static_predicates == {"connected"}
    assert {a.name for a in result.actions} == {"move r1 a b", "move r1 b c"}


def test_forbid_duplicate_bindings_matches_instantiate():
    """allow_duplicate_bindings=False reproduces instantiate() exactly."""
    op = _move_op()
    universe = {**ROBOTS, "location": {"a", "b", "c"}}
    result = ground_operators([op], universe, set())
    assert {a.name for a in result.actions} == {
        a.name for a in op.instantiate(universe)
    }

    permissive = ground_operators(
        [op], universe, set(), allow_duplicate_bindings=True
    )
    assert {a.name for a in permissive.actions} > {a.name for a in result.actions}
    assert "move r1 a a" in {a.name for a in permissive.actions}


def test_equality_constraints():
    """Neq/Eq filter bindings at grounding; instantiate refuses them."""
    op = Operator(
        name="pair",
        parameters=[("?a", "item"), ("?b", "item")],
        preconditions=[F("ready"), Neq("?a", "?b")],
        effects=[Effect(time=1.0, resulting_fluents={F("paired ?a ?b")})],
    )
    universe = {"item": {"x", "y"}}
    # `ready` is static (no effect touches it) so it needs a fact to hold.
    # eliminate_static is pinned off: this test is about grounded structure
    # (Neq must never appear as a runtime precondition, `ready` must).
    result = ground_operators(
        [op], universe, {F("ready")}, allow_duplicate_bindings=True,
        eliminate_static=False,
    )
    assert {a.name for a in result.actions} == {"pair x y", "pair y x"}
    # Neq is a grounding constraint, not a runtime precondition.
    assert all(a.preconditions == frozenset({F("ready")}) for a in result.actions)

    self_op = Operator(
        name="self",
        parameters=[("?a", "item"), ("?b", "item")],
        preconditions=[Eq("?a", "?b")],
        effects=[Effect(time=1.0, resulting_fluents={F("selfed ?a")})],
    )
    self_result = ground_operators(
        [self_op], universe, set(), allow_duplicate_bindings=True
    )
    assert {a.name for a in self_result.actions} == {"self x x", "self y y"}

    with pytest.raises(TypeError, match="grounding constraints"):
        op.instantiate(universe)


def test_skip_on_exception_skips_single_binding():
    class Undefined(Exception):
        pass

    def duration(r, frm, to):
        if to == "b":
            raise Undefined
        return 3.0

    op = Operator(
        name="go",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from")],
        effects=[
            Effect(
                time=(duration, ["?r", "?from", "?to"]),
                resulting_fluents={F("at ?r ?to")},
            )
        ],
    )
    result = ground_operators(
        [op], {**ROBOTS, "location": {"a", "b", "c"}}, set(), skip_on=(Undefined,)
    )
    names = {a.name for a in result.actions}
    assert names == {"go r1 a c", "go r1 c a", "go r1 b a", "go r1 b c"}


def test_static_inference_scans_branch_effects():
    """A predicate touched only inside prob/cond/forall sub-effects is
    dynamic; treat_dynamic forces predicates dynamic; assert_static is loud.

    Misclassifying these as static would silently prune real actions.
    """
    op = Operator(
        name="act",
        parameters=[("?x", "item")],
        preconditions=[F("in-prob ?x"), F("in-cond ?x"), F("in-forall ?x"),
                       F("env-owned ?x"), F("truly-static ?x")],
        effects=[
            Effect(
                time=1.0,
                # `go` is made dynamic below so the branch condition stays
                # runtime; the point here is that branch *effect* predicates
                # are dynamic while conditions only read.
                resulting_fluents={~F("go")},
                prob_effects=[(1.0, [Effect(time=0, resulting_fluents={F("in-prob ?x")})])],
                cond_effects=[({F("go")}, [Effect(time=0, resulting_fluents={F("in-cond ?x")})])],
                forall_effects=[ForallEffect(
                    [("?y", "item")], set(),
                    [Effect(time=0, resulting_fluents={F("in-forall ?y")})],
                )],
            ),
        ],
    )
    facts = {F("in-prob a"), F("in-cond a"), F("in-forall a"),
             F("env-owned a"), F("truly-static a")}
    result = ground_operators(
        [op], {"item": {"a", "b"}}, facts, treat_dynamic={"env-owned"}
    )
    # Only truly-static may prune: item b lacks the fact, so only `act a`
    # exists; the branch-touched and env-owned predicates stay runtime.
    assert result.static_predicates == {"truly-static"}
    assert {a.name for a in result.actions} == {"act a"}

    with pytest.raises(ValueError, match="touched by effects"):
        ground_operators(
            [op], {"item": {"a"}}, facts, assert_static={"in-prob"}
        )


def test_stats_show_early_pruning():
    """Backtracking visits a tiny fraction of the nominal cross-product.

    Freecell-shaped: four parameters chained by static relations, so early
    checks prune each dead pair before deeper parameters ever expand.
    """
    n = 20
    items = {f"i{j}" for j in range(n)}
    chain = {
        F(f"rel i{j} i{j + 1}") for j in range(n - 1)
    }
    op = Operator(
        name="chain4",
        parameters=[("?a", "item"), ("?b", "item"), ("?c", "item"), ("?d", "item")],
        preconditions=[F("rel ?a ?b"), F("rel ?b ?c"), F("rel ?c ?d")],
        effects=[Effect(time=1.0, resulting_fluents={F("done ?a ?d")})],
    )
    result = ground_operators(
        [op], {"item": items}, chain, allow_duplicate_bindings=True
    )
    assert result.stats.actions_kept == n - 3  # the n-3 length-3 chain walks
    assert result.stats.nominal_bindings == n**4
    assert result.stats.visited_bindings < result.stats.nominal_bindings / 10


def test_condition_simplification():
    """Static conjuncts of `when` conditions are evaluated at grounding."""
    op = Operator(
        name="drop",
        parameters=[("?x", "item")],
        preconditions=[F("holding ?x")],
        effects=[Effect(
            time=1.0,
            resulting_fluents={~F("holding ?x")},
            cond_effects=[(
                {F("fragile ?x"), F("armed")},
                [Effect(time=0, resulting_fluents={F("broken ?x")})],
            )],
        )],
    )
    # `fragile` is static; `armed` is dynamic (touched below via a second op).
    arm_op = Operator(
        name="arm", parameters=[], preconditions=[],
        effects=[Effect(time=1.0, resulting_fluents={F("armed")})],
    )
    result = ground_operators(
        [op, arm_op],
        {"item": {"vase", "brick"}},
        {F("holding vase"), F("holding brick"), F("fragile vase")},
    )
    by_name = {a.name: a for a in result.actions}
    # vase: fragile conjunct verified true and removed; dynamic conjunct stays.
    (vase_branch,) = by_name["drop vase"].effects[0].cond_effects
    assert set(vase_branch.conditions) == {F("armed")}
    # brick: fragile is false, so the branch is gone entirely.
    assert not by_name["drop brick"].effects[0].cond_effects


def test_negated_static_check_prunes_bindings():
    """Bindings violating a negated static precondition are never grounded."""
    op = _move_op(preconditions_extra=[~F("blocked ?to")])
    result = ground_operators(
        [op],
        {**ROBOTS, "location": {"a", "b"}},
        {F("at r1 a"), F("free r1"), F("blocked b")},
    )
    assert {a.name for a in result.actions} == {"move r1 b a"}


def test_eliminate_static_strips_and_reports():
    """Static preconditions are stripped; unreferenced facts are eliminable."""
    op = _move_op(preconditions_extra=[F("connected ?from ?to")])
    facts = {
        F("at r1 a"), F("free r1"),
        F("connected a b"), F("landmark b"),
    }
    result = ground_operators(
        [op], {**ROBOTS, "location": {"a", "b"}}, facts,
        runtime_referenced={"landmark"},  # e.g. read by the goal
    )
    (action,) = result.actions
    assert action.name == "move r1 a b"
    # The verified static precondition is gone; dynamic ones remain.
    assert set(action.preconditions) == {F("at r1 a"), F("free r1")}
    # `connected` facts are dead weight at runtime; `landmark` is referenced.
    assert result.eliminable_fluents == {F("connected a b")}
    assert result.eliminated_predicates == {"connected"}


def test_unbound_precondition_variable_is_an_error():
    """A misspelled parameter must not silently disable a static constraint.

    Without the check the binding-time check is never scheduled (nothing ever
    binds `?form`) and eliminate_static then strips the precondition anyway,
    so the typo would *widen* the action set instead of narrowing it.
    """
    op = Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("connected ?form ?to")],  # typo
        effects=[Effect(time=1.0, resulting_fluents={F("at ?r ?to")})],
    )
    with pytest.raises(ValueError, match=r"unbound variable\(s\) \['\?form'\]"):
        ground_operators(
            [op], {**ROBOTS, "location": {"a", "b"}}, {F("connected a b")}
        )


def test_unbound_constraint_variable_is_an_error():
    op = Operator(
        name="pair",
        parameters=[("?a", "item")],
        preconditions=[Neq("?a", "?b")],
        effects=[Effect(time=1.0, resulting_fluents={F("paired ?a")})],
    )
    with pytest.raises(ValueError, match=r"unbound variable\(s\) \['\?b'\]"):
        ground_operators([op], {"item": {"x", "y"}}, set())


def test_unknown_parameter_type_is_an_error_but_empty_type_is_fine():
    """A missing type is a typo; a declared-but-empty one is legitimate."""
    op = _move_op()
    with pytest.raises(ValueError, match="absent from objects_by_type"):
        ground_operators([op], ROBOTS, set())  # no "location" key at all

    empty = ground_operators([op], {**ROBOTS, "location": set()}, set())
    assert empty.actions == []


def test_infinite_time_inside_a_branch_drops_the_action():
    """`inf` sub-effect times count, not just top-level ones."""
    op = Operator(
        name="risky",
        parameters=[("?r", "robot")],
        preconditions=[F("free ?r")],
        effects=[Effect(
            time=1.0,
            resulting_fluents={F("free ?r")},
            cond_effects=[({F("armed")}, [
                Effect(time=float("inf"), resulting_fluents={F("boom ?r")}),
            ])],
        )],
    )
    # `armed` is dynamic, so the branch survives grounding and its inf stands.
    arm = Operator(
        name="arm", parameters=[], preconditions=[],
        effects=[Effect(time=1.0, resulting_fluents={F("armed")})],
    )
    result = ground_operators([op, arm], ROBOTS, {F("free r1")})
    assert {a.name for a in result.actions} == {"arm"}


def test_infinite_time_in_a_statically_dropped_branch_keeps_the_action():
    """The filter runs after simplification, so a dead branch is harmless."""
    op = Operator(
        name="risky",
        parameters=[("?r", "robot")],
        preconditions=[F("free ?r")],
        effects=[Effect(
            time=1.0,
            resulting_fluents={F("free ?r")},
            # `armed` is static and absent, so this branch can never fire.
            cond_effects=[({F("armed")}, [
                Effect(time=float("inf"), resulting_fluents={F("boom ?r")}),
            ])],
        )],
    )
    result = ground_operators([op], ROBOTS, {F("free r1")})
    assert {a.name for a in result.actions} == {"risky r1"}

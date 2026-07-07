"""Convert parsed PDDL/PPDDL into railroad planning problems.

Mapping semantics (see README.md in this package for the full story):

- A synthetic serializing agent is introduced: every action requires
  ``free <agent>``, consumes it at t=0, and releases it when the action
  completes. This satisfies the railroad core's hardcoded ``free`` fluent
  semantics and serializes the (single-agent) PDDL problem.
- With ``(:metric minimize (total-cost))``, each action's duration is its
  ``(increase (total-cost) ...)`` amount, so minimizing completion time is
  exactly minimizing total cost. Without a metric, PDDL's implicit objective
  is plan length, so every action gets duration 1.
- PPDDL ``(probabilistic ...)`` effects become railroad probabilistic effect
  branches; an implicit remainder branch is added when probabilities sum
  to < 1.
- Quantifiers are compiled out over the problem's finite object universe:
  ``forall`` expands to conjunctions, ``exists`` in preconditions lifts the
  quantified variable into an extra operator parameter, and quantifiers in
  goals become And/Or goal trees.
"""

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

from railroad._bindings import FalseGoal, TrueGoal
from railroad.core import OptExpr
from railroad.core import (
    Action,
    AndGoal,
    Effect,
    Fluent,
    Goal,
    LiteralGoal,
    Operator,
    OrGoal,
    State,
)

from .errors import PDDLParseError, UnsupportedPDDLError
from .parser import (
    OBJECT_TYPE,
    And,
    ConditionNode,
    EffectAnd,
    EffectForall,
    EffectNode,
    Equals,
    Exists,
    Forall,
    Increase,
    Literal,
    Or,
    PDDLDomain,
    PDDLProblem,
    Probabilistic,
    TypedVars,
    When,
)

# Zero-cost actions still need to advance time (the event queue is what
# produces decision points), so they get a tiny positive duration.
EPSILON_DURATION = 1e-3

PROBABILITY_TOLERANCE = 1e-6

# Fluent names with hardcoded semantics in the railroad C++ core, plus the
# "not-" prefix reserved by negative-precondition preprocessing. PDDL
# predicates that collide are transparently renamed with this prefix.
_RENAME_PREFIX = "pddl-"

# Equality in `when` conditions compiles to this static predicate; the
# initial state is seeded with (pddl-eq o o) for every object, so condition
# evaluation by fluent lookup answers (= a b) exactly.
EQ_PREDICATE = "pddl-eq"

_RESERVED_PREDICATES = {"free", "waiting", "not", EQ_PREDICATE}

CostAmount = Union[float, Tuple[str, Tuple[str, ...]]]


class _UndefinedFunctionValue(Exception):
    """A grounding referenced a cost function value absent from ``:init``.

    PDDL leaves such groundings undefined, so the grounder skips them.
    """


def _is_var(term: str) -> bool:
    return term.startswith("?")


# ============================================================================
#  Output containers
# ============================================================================


@dataclass
class CompiledOperator:
    """A railroad Operator plus grounding-time-only constraints."""

    operator: Operator
    # (=)/(not (=)) preconditions, evaluated per binding while grounding.
    equality_constraints: List[Equals]
    # Preconditions over static predicates (never in any effect); evaluated
    # against the initial state per binding while grounding.
    static_preconditions: List[Literal]


@dataclass
class ConvertedProblem:
    domain_name: str
    problem_name: str
    agent: str
    metric: str  # human-readable description of the applied objective mapping
    objects_by_type: Dict[str, Set[str]]
    initial_state: State
    goal: Goal
    compiled_operators: List[CompiledOperator]
    _static_fluents: FrozenSet[Fluent] = field(default_factory=frozenset)
    _actions: Optional[List[Action]] = None

    @property
    def operators(self) -> List[Operator]:
        return [c.operator for c in self.compiled_operators]

    def ground_actions(self) -> List[Action]:
        """Ground all operators (cached). May be large for big instances."""
        if self._actions is None:
            self._actions = []
            for comp in self.compiled_operators:
                self._actions.extend(
                    _ground_operator(comp, self.objects_by_type, self._static_fluents)
                )
        return self._actions


# ============================================================================
#  Entry point
# ============================================================================


def convert(domain: PDDLDomain, problem: PDDLProblem) -> ConvertedProblem:
    """Convert a parsed domain/problem pair into a railroad problem."""
    if problem.domain_name and problem.domain_name != domain.name:
        raise PDDLParseError(
            f"Problem {problem.name} is for domain {problem.domain_name!r}, "
            f"not {domain.name!r}"
        )

    object_types = dict(domain.constants)
    object_types.update(problem.objects)
    type_objects = _build_type_objects(domain, object_types)

    rename = _build_predicate_renaming(domain)
    agent = _pick_agent_name(object_types)
    duration_mode, metric_desc = _resolve_metric(problem)

    compiled_operators = [
        _compile_action(
            action, domain, problem, type_objects, rename, agent, duration_mode
        )
        for action in domain.actions
    ]

    init_fluents = {
        _make_fluent(lit, rename) for lit in problem.init_literals
    }
    static_predicates = _find_static_predicates(domain, rename)
    for comp in compiled_operators:
        comp.static_preconditions = [
            lit
            for lit in comp.static_preconditions
            if lit.predicate in static_predicates
        ]

    initial_fluents = set(init_fluents)
    initial_fluents.add(Fluent("free", agent))
    if any(
        _uses_eq_conditions(comp.operator.effects) for comp in compiled_operators
    ):
        initial_fluents.update(
            Fluent(EQ_PREDICATE, obj, obj) for obj in object_types
        )
    goal = _compile_goal(problem.goal, type_objects, rename)

    return ConvertedProblem(
        domain_name=domain.name,
        problem_name=problem.name,
        agent=agent,
        metric=metric_desc,
        objects_by_type={t: set(objs) for t, objs in type_objects.items()},
        initial_state=State(0.0, initial_fluents, []),
        goal=goal,
        compiled_operators=compiled_operators,
        _static_fluents=frozenset(init_fluents),
    )


# ============================================================================
#  Types, names, metric
# ============================================================================


def _build_type_objects(
    domain: PDDLDomain, object_types: Dict[str, str]
) -> Dict[str, Set[str]]:
    """Flatten the type hierarchy: each type maps to objects of it or any subtype."""
    parents = dict(domain.types)
    parents.setdefault(OBJECT_TYPE, OBJECT_TYPE)

    def ancestors(typ: str) -> Set[str]:
        seen = {typ}
        while typ != OBJECT_TYPE:
            typ = parents.get(typ, OBJECT_TYPE)
            if typ in seen:  # cycle guard
                break
            seen.add(typ)
        seen.add(OBJECT_TYPE)
        return seen

    type_objects: Dict[str, Set[str]] = {t: set() for t in parents}
    type_objects.setdefault(OBJECT_TYPE, set())
    for obj, typ in object_types.items():
        for t in ancestors(typ):
            type_objects.setdefault(t, set()).add(obj)
    return type_objects


def _build_predicate_renaming(domain: PDDLDomain) -> Dict[str, str]:
    """Rename predicates that collide with railroad-reserved fluent names."""
    rename = {}
    for pred in domain.predicates:
        if pred in _RESERVED_PREDICATES or pred.startswith("not-"):
            rename[pred] = _RENAME_PREFIX + pred
    return rename


def _pick_agent_name(object_types: Dict[str, str]) -> str:
    agent = "agent"
    while agent in object_types:
        agent = "_" + agent
    return agent


def _resolve_metric(problem: PDDLProblem) -> Tuple[str, str]:
    """Return (duration_mode, description); raise on unsupported objectives."""
    if problem.metric is None:
        return "unit", "no metric: minimize plan length (all durations 1)"
    direction, expr = problem.metric
    if direction == "minimize" and expr in ("(total-cost)", "total-cost"):
        return "cost", "minimize (total-cost): action cost mapped to duration"
    if direction == "minimize" and expr in ("(total-time)", "total-time"):
        return "unit", "minimize (total-time): non-durative, all durations 1"
    if (
        direction == "maximize"
        and expr in ("(reward)", "reward")
        and problem.goal_reward is not None
    ):
        # IPPC goal-directed convention: the only reward is the goal reward
        # (reward-bearing effects are rejected separately), so maximizing
        # reward is reaching the goal; we minimize expected plan length.
        return "unit", (
            "maximize (reward) with goal-reward only: reinterpreted as "
            "reach-goal, minimize expected plan length"
        )
    raise UnsupportedPDDLError(
        f"metric:{direction} {expr}", f"in problem {problem.name}"
    )


def _make_fluent(lit: Literal, rename: Dict[str, str]) -> Fluent:
    name = rename.get(lit.predicate, lit.predicate)
    return Fluent(name, *lit.args, negated=lit.negated)


def _find_static_predicates(
    domain: PDDLDomain, rename: Dict[str, str]
) -> Set[str]:
    """Predicates that never appear in any effect (so their truth is fixed)."""
    dynamic: Set[str] = set()

    def visit(node: EffectNode) -> None:
        if isinstance(node, Literal):
            dynamic.add(rename.get(node.predicate, node.predicate))
        elif isinstance(node, EffectAnd):
            for c in node.children:
                visit(c)
        elif isinstance(node, EffectForall):
            visit(node.body)
        elif isinstance(node, Probabilistic):
            for _, branch in node.branches:
                visit(branch)
        elif isinstance(node, When):
            # Only the effect side writes; the condition merely reads.
            visit(node.effect)
        # Increase nodes carry no predicates.

    for action in domain.actions:
        visit(action.effect)
    all_preds = {rename.get(p, p) for p in domain.predicates}
    return all_preds - dynamic


# ============================================================================
#  Substitution over AST nodes
# ============================================================================


def _sub_terms(args: Sequence[str], mapping: Dict[str, str]) -> Tuple[str, ...]:
    return tuple(mapping.get(a, a) for a in args)


def _substitute_condition(node: ConditionNode, mapping: Dict[str, str]) -> ConditionNode:
    if isinstance(node, Literal):
        return Literal(node.predicate, _sub_terms(node.args, mapping), node.negated)
    if isinstance(node, Equals):
        return Equals(
            mapping.get(node.left, node.left),
            mapping.get(node.right, node.right),
            node.negated,
        )
    if isinstance(node, And):
        return And([_substitute_condition(c, mapping) for c in node.children])
    if isinstance(node, Or):
        return Or([_substitute_condition(c, mapping) for c in node.children])
    if isinstance(node, (Forall, Exists)):
        # Quantified variables shadow the outer mapping.
        inner = {k: v for k, v in mapping.items() if k not in dict(node.variables)}
        return type(node)(node.variables, _substitute_condition(node.body, inner))
    raise PDDLParseError(f"Unexpected condition node: {node!r}")


def _substitute_effect(node: EffectNode, mapping: Dict[str, str]) -> EffectNode:
    if isinstance(node, Literal):
        return Literal(node.predicate, _sub_terms(node.args, mapping), node.negated)
    if isinstance(node, EffectAnd):
        return EffectAnd([_substitute_effect(c, mapping) for c in node.children])
    if isinstance(node, EffectForall):
        inner = {k: v for k, v in mapping.items() if k not in dict(node.variables)}
        return EffectForall(node.variables, _substitute_effect(node.body, inner))
    if isinstance(node, Probabilistic):
        return Probabilistic(
            [(p, _substitute_effect(e, mapping)) for p, e in node.branches]
        )
    if isinstance(node, When):
        return When(
            _substitute_condition(node.condition, mapping),
            _substitute_effect(node.effect, mapping),
        )
    if isinstance(node, Increase):
        amount = node.amount
        if isinstance(amount, tuple):
            amount = (amount[0], _sub_terms(amount[1], mapping))
        return Increase(node.function, _sub_terms(node.args, mapping), amount)
    raise PDDLParseError(f"Unexpected effect node: {node!r}")


def _quantifier_expansions(
    variables: TypedVars, type_objects: Dict[str, Set[str]], context: str
) -> List[Dict[str, str]]:
    domains = []
    for var, typ in variables:
        objs = sorted(type_objects.get(typ, set()))
        domains.append([(var, o) for o in objs])
    return [dict(combo) for combo in itertools.product(*domains)]


# ============================================================================
#  Precondition compilation
# ============================================================================


@dataclass
class _CompiledPrecondition:
    literals: List[Literal] = field(default_factory=list)
    equalities: List[Equals] = field(default_factory=list)
    extra_parameters: TypedVars = field(default_factory=list)


def _compile_precondition(
    node: Optional[ConditionNode],
    type_objects: Dict[str, Set[str]],
    context: str,
    counter: itertools.count,
) -> _CompiledPrecondition:
    out = _CompiledPrecondition()
    if node is not None:
        _compile_precondition_into(node, type_objects, context, counter, out)
    return out


def _compile_precondition_into(
    node: ConditionNode,
    type_objects: Dict[str, Set[str]],
    context: str,
    counter: itertools.count,
    out: _CompiledPrecondition,
) -> None:
    if isinstance(node, Literal):
        out.literals.append(node)
    elif isinstance(node, Equals):
        out.equalities.append(node)
    elif isinstance(node, And):
        for c in node.children:
            _compile_precondition_into(c, type_objects, context, counter, out)
    elif isinstance(node, Or):
        raise UnsupportedPDDLError(
            "disjunctive-preconditions", f"(or ...) in {context}"
        )
    elif isinstance(node, Forall):
        for mapping in _quantifier_expansions(node.variables, type_objects, context):
            _compile_precondition_into(
                _substitute_condition(node.body, mapping),
                type_objects,
                context,
                counter,
                out,
            )
    elif isinstance(node, Exists):
        # Lift the quantified variables into extra operator parameters: the
        # grounder enumerates witnesses, so the action is applicable iff some
        # binding satisfies the body.
        mapping = {}
        for var, typ in node.variables:
            fresh = f"{var}-e{next(counter)}"
            mapping[var] = fresh
            out.extra_parameters.append((fresh, typ))
        _compile_precondition_into(
            _substitute_condition(node.body, mapping),
            type_objects,
            context,
            counter,
            out,
        )
    else:
        raise PDDLParseError(f"Unexpected condition node in {context}: {node!r}")


# ============================================================================
#  Effect compilation
# ============================================================================


@dataclass
class _CompiledEffect:
    literals: List[Literal] = field(default_factory=list)
    # Each group is one (probabilistic ...) construct: a list of branches.
    # Groups are independent of each other; branches within a group are
    # mutually exclusive.
    prob_groups: List[List[Tuple[float, "_CompiledEffect"]]] = field(
        default_factory=list
    )
    # Each group is one (when ...) construct: condition literals plus the
    # sub-effect applied when they hold.
    cond_groups: List[Tuple[List[Literal], "_CompiledEffect"]] = field(
        default_factory=list
    )
    cost_terms: List[CostAmount] = field(default_factory=list)


def _compile_effect(
    node: EffectNode,
    type_objects: Dict[str, Set[str]],
    context: str,
    nested_in: Optional[str] = None,
) -> _CompiledEffect:
    out = _CompiledEffect()
    _compile_effect_into(node, type_objects, context, nested_in, out)
    return out


def _compile_effect_into(
    node: EffectNode,
    type_objects: Dict[str, Set[str]],
    context: str,
    nested_in: Optional[str],
    out: _CompiledEffect,
) -> None:
    if isinstance(node, Literal):
        out.literals.append(node)
    elif isinstance(node, EffectAnd):
        for c in node.children:
            _compile_effect_into(c, type_objects, context, nested_in, out)
    elif isinstance(node, EffectForall):
        for mapping in _quantifier_expansions(node.variables, type_objects, context):
            _compile_effect_into(
                _substitute_effect(node.body, mapping),
                type_objects,
                context,
                nested_in,
                out,
            )
    elif isinstance(node, Probabilistic):
        total = 0.0
        branches: List[Tuple[float, _CompiledEffect]] = []
        for prob, branch_node in node.branches:
            if prob < -PROBABILITY_TOLERANCE or prob > 1 + PROBABILITY_TOLERANCE:
                raise PDDLParseError(f"Probability {prob} out of range in {context}")
            total += prob
            branches.append(
                (
                    prob,
                    _compile_effect(
                        branch_node, type_objects, context, "probabilistic"
                    ),
                )
            )
        if total > 1 + PROBABILITY_TOLERANCE:
            raise PDDLParseError(
                f"Probabilities sum to {total} > 1 in {context}"
            )
        out.prob_groups.append(branches)
    elif isinstance(node, When):
        conditions = _compile_when_condition(node.condition, type_objects, context)
        sub_effect = _compile_effect(node.effect, type_objects, context, "conditional")
        out.cond_groups.append((conditions, sub_effect))
    elif isinstance(node, Increase):
        if node.function == "reward":
            raise UnsupportedPDDLError("rewards", f"(increase (reward) ...) in {context}")
        if node.function != "total-cost":
            raise UnsupportedPDDLError(
                "numeric-effects", f"(increase ({node.function} ...) ...) in {context}"
            )
        if node.args:
            raise UnsupportedPDDLError(
                "numeric-effects", f"parameterized total-cost in {context}"
            )
        if nested_in is not None:
            raise UnsupportedPDDLError(
                f"{nested_in}-cost",
                f"(increase (total-cost) ...) inside a {nested_in} branch in {context}",
            )
        out.cost_terms.append(node.amount)
    else:
        raise PDDLParseError(f"Unexpected effect node in {context}: {node!r}")


def _compile_when_condition(
    node: ConditionNode, type_objects: Dict[str, Set[str]], context: str
) -> List[Literal]:
    """Compile a ``when`` condition to a conjunction of literals.

    Negated literals are fine (the core evaluates them by absence); anything
    that cannot be flattened to a conjunction — disjunction, equality,
    existential — is unsupported.
    """
    if isinstance(node, Literal):
        return [node]
    if isinstance(node, Equals):
        # Equality is static, so it becomes a lookup against the seeded
        # (pddl-eq o o) fluents (see convert()).
        return [Literal(EQ_PREDICATE, (node.left, node.right), node.negated)]
    if isinstance(node, And):
        return [
            lit
            for child in node.children
            for lit in _compile_when_condition(child, type_objects, context)
        ]
    if isinstance(node, Forall):
        return [
            lit
            for mapping in _quantifier_expansions(node.variables, type_objects, context)
            for lit in _compile_when_condition(
                _substitute_condition(node.body, mapping), type_objects, context
            )
        ]
    raise UnsupportedPDDLError(
        "conditional-effect-condition",
        f"unsupported (when ...) condition {type(node).__name__} in {context}",
    )


def _make_duration(
    cost_terms: List[CostAmount],
    fn_values: Dict[Tuple[str, Tuple[str, ...]], float],
    duration_mode: str,
    context: str,
):
    """Build the railroad OptExpr for an action's duration."""
    if duration_mode == "unit":
        return 1.0
    constant = sum(t for t in cost_terms if isinstance(t, float))
    refs = [t for t in cost_terms if isinstance(t, tuple)]
    if not refs:
        return max(constant, EPSILON_DURATION)

    # Duration depends on grounded function values, e.g.
    # (increase (total-cost) (road-length ?a ?b)). Flatten all referenced
    # args into one OptExpr arg list; the callable re-partitions them.
    shape = [(fn, len(args)) for fn, args in refs]
    flat_args = [a for _, args in refs for a in args]

    def duration_fn(*values: str) -> float:
        total = constant
        i = 0
        for fn, n in shape:
            key = (fn, tuple(values[i : i + n]))
            i += n
            if key not in fn_values:
                raise _UndefinedFunctionValue(
                    f"No :init value for function {key} needed by {context}"
                )
            total += fn_values[key]
        return max(total, EPSILON_DURATION)

    return (duration_fn, flat_args)


# ============================================================================
#  Action compilation
# ============================================================================


def _compile_action(
    action,
    domain: PDDLDomain,
    problem: PDDLProblem,
    type_objects: Dict[str, Set[str]],
    rename: Dict[str, str],
    agent: str,
    duration_mode: str,
) -> CompiledOperator:
    context = f"action {action.name}"
    counter = itertools.count()
    pre = _compile_precondition(action.precondition, type_objects, context, counter)
    eff = _compile_effect(action.effect, type_objects, context)

    parameters = list(action.parameters) + pre.extra_parameters
    param_names = {var for var, _ in parameters}
    _check_variables_bound(pre.literals, eff, pre.equalities, param_names, context)

    duration = _make_duration(
        eff.cost_terms, problem.init_function_values, duration_mode, context
    )

    preconditions = [Fluent("free", agent)] + [
        _make_fluent(lit, rename) for lit in pre.literals
    ]

    completion_fluents = {_make_fluent(lit, rename) for lit in eff.literals}
    completion_fluents.add(Fluent("free", agent))
    first_group = eff.prob_groups[0] if eff.prob_groups else None
    # All conditional branches ride on the single completion effect: their
    # conditions are evaluated when it fires, before its fluents apply, so
    # they see the pre-action state (PDDL `when` semantics).
    effects = [
        Effect(time=0, resulting_fluents={Fluent("free", agent, negated=True)}),
        Effect(
            time=duration,
            resulting_fluents=completion_fluents,
            prob_effects=_group_to_prob_effects(first_group, rename)
            if first_group
            else [],
            cond_effects=_cond_groups_to_effects(eff.cond_groups, rename),
        ),
    ]
    for group in eff.prob_groups[1:]:
        effects.append(
            Effect(time=duration, prob_effects=_group_to_prob_effects(group, rename))
        )

    operator = Operator(
        name=action.name,
        parameters=parameters,
        preconditions=preconditions,
        effects=effects,
    )
    # Every precondition literal is a static-check candidate; convert() prunes
    # this list down to genuinely static predicates afterwards.
    static_candidates = [
        Literal(rename.get(l.predicate, l.predicate), l.args, l.negated)
        for l in pre.literals
    ]
    return CompiledOperator(operator, pre.equalities, static_candidates)


def _group_to_prob_effects(
    group: List[Tuple[float, _CompiledEffect]], rename: Dict[str, str]
) -> List[Tuple[OptExpr, List[Effect]]]:
    branches: List[Tuple[OptExpr, List[Effect]]] = []
    total = 0.0
    for prob, compiled in group:
        total += prob
        branches.append((prob, _compiled_to_sub_effects(compiled, rename)))
    if total < 1 - PROBABILITY_TOLERANCE:
        branches.append((1 - total, []))  # implicit "nothing happens"
    return branches


def _uses_eq_conditions(effects: Sequence[Effect]) -> bool:
    """Whether any conditional branch (recursively) tests the eq predicate."""
    for eff in effects:
        for conditions, sub_effects in eff.cond_effects:
            if any(f.name == EQ_PREDICATE for f in conditions):
                return True
            if _uses_eq_conditions(sub_effects):
                return True
        for _, sub_effects in eff.prob_effects:
            if _uses_eq_conditions(sub_effects):
                return True
    return False


def _cond_groups_to_effects(
    cond_groups: List[Tuple[List[Literal], _CompiledEffect]],
    rename: Dict[str, str],
) -> List[Tuple[Set[Fluent], List[Effect]]]:
    return [
        (
            {_make_fluent(lit, rename) for lit in conditions},
            _compiled_to_sub_effects(sub_effect, rename),
        )
        for conditions, sub_effect in cond_groups
    ]


def _compiled_to_sub_effects(
    compiled: _CompiledEffect, rename: Dict[str, str]
) -> List[Effect]:
    fluents = {_make_fluent(lit, rename) for lit in compiled.literals}
    first_group = compiled.prob_groups[0] if compiled.prob_groups else None
    sub_effects = []
    if fluents or first_group or compiled.cond_groups:
        sub_effects.append(
            Effect(
                time=0,
                resulting_fluents=fluents,
                prob_effects=_group_to_prob_effects(first_group, rename)
                if first_group
                else [],
                cond_effects=_cond_groups_to_effects(compiled.cond_groups, rename),
            )
        )
    for group in compiled.prob_groups[1:]:
        sub_effects.append(
            Effect(time=0, prob_effects=_group_to_prob_effects(group, rename))
        )
    return sub_effects


def _check_variables_bound(
    literals: List[Literal],
    eff: _CompiledEffect,
    equalities: List[Equals],
    param_names: Set[str],
    context: str,
) -> None:
    def check_terms(terms) -> None:
        for t in terms:
            if _is_var(t) and t not in param_names:
                raise PDDLParseError(f"Unbound variable {t} in {context}")

    for lit in literals:
        check_terms(lit.args)
    for eq in equalities:
        check_terms([eq.left, eq.right])

    def check_effect(e: _CompiledEffect) -> None:
        for lit in e.literals:
            check_terms(lit.args)
        for term in e.cost_terms:
            if isinstance(term, tuple):
                check_terms(term[1])
        for group in e.prob_groups:
            for _, branch in group:
                check_effect(branch)
        for conditions, sub_effect in e.cond_groups:
            for lit in conditions:
                check_terms(lit.args)
            check_effect(sub_effect)

    check_effect(eff)


# ============================================================================
#  Goal compilation
# ============================================================================


def _compile_goal(
    node: Optional[ConditionNode],
    type_objects: Dict[str, Set[str]],
    rename: Dict[str, str],
) -> Goal:
    if node is None:
        return TrueGoal()
    if isinstance(node, Literal):
        for arg in node.args:
            if _is_var(arg):
                raise PDDLParseError(f"Unbound variable {arg} in goal")
        return LiteralGoal(_make_fluent(node, rename))
    if isinstance(node, Equals):
        left, right = node.left, node.right
        if _is_var(left) or _is_var(right):
            raise PDDLParseError("Unbound variable in goal equality")
        return TrueGoal() if (left == right) != node.negated else FalseGoal()
    if isinstance(node, And):
        return AndGoal([_compile_goal(c, type_objects, rename) for c in node.children])
    if isinstance(node, Or):
        return OrGoal([_compile_goal(c, type_objects, rename) for c in node.children])
    if isinstance(node, (Forall, Exists)):
        children = [
            _compile_goal(_substitute_condition(node.body, m), type_objects, rename)
            for m in _quantifier_expansions(node.variables, type_objects, "goal")
        ]
        if not children:
            return TrueGoal() if isinstance(node, Forall) else FalseGoal()
        return AndGoal(children) if isinstance(node, Forall) else OrGoal(children)
    raise PDDLParseError(f"Unexpected goal node: {node!r}")


# ============================================================================
#  Grounding
# ============================================================================


def _ground_operator(
    comp: CompiledOperator,
    objects_by_type: Dict[str, Set[str]],
    static_fluents: FrozenSet[Fluent],
) -> List[Action]:
    """Enumerate bindings via backtracking with early constraint pruning.

    Unlike ``Operator.instantiate`` this allows the same object to bind
    multiple parameters (PDDL permits it; domains that need distinctness use
    ``(not (= ?x ?y))``), and it evaluates equality constraints plus static
    preconditions as soon as their variables are bound, which keeps the
    enumeration tractable for large IPC instances.
    """
    op = comp.operator
    params = op.parameters
    domains = [sorted(objects_by_type.get(typ, set())) for _, typ in params]

    # Constraints as (needed_vars, check_fn(binding) -> bool).
    constraints: List[Tuple[Set[str], object]] = []
    for eq in comp.equality_constraints:
        needed = {t for t in (eq.left, eq.right) if _is_var(t)}
        constraints.append((needed, eq))
    for lit in comp.static_preconditions:
        if lit.negated:
            continue  # a negative static precondition rarely prunes; skip
        needed = {a for a in lit.args if _is_var(a)}
        constraints.append((needed, lit))

    order = _order_parameters(params, [needed for needed, _ in constraints])

    # Assign each constraint to the earliest depth at which all its variables
    # are bound; variable-free constraints are checked once up front.
    bound_after: List[Set[str]] = []
    acc: Set[str] = set()
    for pos in order:
        acc.add(params[pos][0])
        bound_after.append(set(acc))
    checks_at: List[List[object]] = [[] for _ in order]
    always: List[object] = []
    for needed, check in constraints:
        if not needed:
            always.append(check)
            continue
        for depth, bound in enumerate(bound_after):
            if needed <= bound:
                checks_at[depth].append(check)
                break

    def satisfied(check, binding: Dict[str, str]) -> bool:
        if isinstance(check, Equals):
            left = binding.get(check.left, check.left)
            right = binding.get(check.right, check.right)
            return (left == right) != check.negated
        assert isinstance(check, Literal)
        fluent = Fluent(
            check.predicate, *[binding.get(a, a) for a in check.args]
        )
        return fluent in static_fluents

    if not all(satisfied(c, {}) for c in always):
        return []

    actions: List[Action] = []
    binding: Dict[str, str] = {}

    def dfs(depth: int) -> None:
        if depth == len(order):
            try:
                actions.append(op._ground(binding, objects_by_type))
            except _UndefinedFunctionValue:
                pass  # duration undefined for this binding -> not a real action
            return
        pos = order[depth]
        var = params[pos][0]
        for obj in domains[pos]:
            binding[var] = obj
            if all(satisfied(c, binding) for c in checks_at[depth]):
                dfs(depth + 1)
        del binding[var]

    if all(len(d) > 0 for d in domains):
        dfs(0)
    return actions


def _order_parameters(
    params: TypedVars, constraint_vars: List[Set[str]]
) -> List[int]:
    """Greedy ordering: bind next the parameter completing the most constraints."""
    remaining = list(range(len(params)))
    bound: Set[str] = set()
    order: List[int] = []
    while remaining:
        def score(i: int) -> int:
            var = params[i][0]
            return sum(
                1
                for needed in constraint_vars
                if var in needed and needed <= (bound | {var})
            )

        best = max(remaining, key=score)
        order.append(best)
        remaining.remove(best)
        bound.add(params[best][0])
    return order

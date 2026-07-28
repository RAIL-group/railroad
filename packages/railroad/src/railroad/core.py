# Re-import required dependencies due to kernel reset
from typing import Callable, List, Tuple, Dict, Set, Union, Sequence, Collection, Mapping, Optional
import itertools
import math

from railroad._bindings import GroundedEffect, Fluent, Action, State
from railroad._bindings import transition
from railroad._bindings import LiteralGoal, AndGoal, OrGoal, Goal

__all__ = ["transition"]  # re-exported from _bindings
from railroad._bindings import ff_heuristic as _ff_heuristic_cpp


def ff_heuristic(
    state: State,
    goal: Union[Goal, Fluent],
    all_actions: List[Action],
    lambda_add: float = 0.5,
    lambda_max: float = 0.0,
    lambda_ff: float = 0.5,
    at_implies_found: bool = True,
) -> float:
    """Compute FF heuristic value for a state (probabilistic version).

    Args:
        state: The current state
        goal: Goal to achieve. Can be:
            - A Goal object: F("a") & F("b"), AndGoal([...]), etc.
            - A single Fluent: F("visited a") (auto-wrapped to LiteralGoal)
        all_actions: List of all available actions
        lambda_add: weight on the additive component (sum optimistic_cost over goal fluents)
        lambda_max: weight on the max component (max optimistic_cost over goal fluents)
        lambda_ff: weight on the relaxed-plan-cost component (sum action_duration over unique actions)
        at_implies_found: when True (default), any required ``at <entity> <loc>``
            also requires a reachable ``found <entity>``; lets search goals be
            left implicit. Unreachable ``found`` (e.g. for robots) is skipped.

    Returns:
        Heuristic value (estimated cost to reach goal). Defaults are an even
        split between h_add and h_ff (0.5, 0.0, 0.5).
    """
    # Normalize goal (wrap Fluent in LiteralGoal if needed)
    if isinstance(goal, Fluent):
        goal = LiteralGoal(goal)
    return _ff_heuristic_cpp(
        state, goal, all_actions,
        lambda_add=lambda_add, lambda_max=lambda_max, lambda_ff=lambda_ff,
        at_implies_found=at_implies_found,
    )


Num = Union[float, int]


Binding = Dict[str, str]
Bindable = Callable[[Binding], Num]
OptExpr = Union[float, Tuple[Callable[..., float], List[str]]]


def _make_bindable(opt_expr: OptExpr) -> Bindable:
    if isinstance(opt_expr, Num):
        return lambda *args: opt_expr
    else:
        fn = opt_expr[0]
        args = opt_expr[1]
        return lambda b: fn(*[b.get(arg, arg) for arg in args])


class ForallEffect:
    """Universally quantified conditional sub-effects on an :class:`Effect`.

    The native analogue of PDDL's ``(forall (?x - t) (when <cond> <eff>))``:
    at grounding time (``Operator.instantiate``), one conditional branch is
    produced per assignment of the quantified variables to objects of their
    types. With empty ``conditions`` the sub-effects apply unconditionally
    for every object (a plain universal effect).

    Quantified variables may appear in ``conditions`` and inside ``effects``;
    a quantified variable shadows an operator parameter of the same name.

    Example (briefcase: moving relocates exactly the items inside)::

        Effect(
            time=2.0,
            resulting_fluents={F("at briefcase ?to"), F("not at briefcase ?from")},
            forall_effects=[ForallEffect(
                variables=[("?obj", "item")],
                conditions={F("in ?obj")},
                effects=[Effect(time=0, resulting_fluents={
                    F("at ?obj ?to"), F("not at ?obj ?from")})],
            )],
        )
    """

    def __init__(
        self,
        variables: List[Tuple[str, str]],
        conditions: Set[Fluent],
        effects: List["Effect"],
    ):
        self.variables = variables
        self.conditions = conditions
        self.effects = effects


class Effect:
    def __init__(
        self,
        time: OptExpr,
        prob_effects: Optional[List[Tuple[OptExpr, List["Effect"]]]] = None,
        resulting_fluents: Optional[Set[Fluent]] = None,
        cond_effects: Optional[List[Tuple[Set[Fluent], List["Effect"]]]] = None,
        forall_effects: Optional[List[ForallEffect]] = None,
    ):
        """A (possibly branching) effect scheduled ``time`` after its action.

        ``cond_effects`` are conditional branches (PDDL ``when``): each
        ``(conditions, sub_effects)`` pair applies its sub-effects only if the
        conditions hold in the state at the moment this effect fires,
        evaluated before this effect's own ``resulting_fluents`` apply.
        Negated condition fluents use negation-as-absence.

        Branching sub-effects (conditional and probabilistic) are applied
        *after* this effect's own ``resulting_fluents``, even when scheduled
        at the same time. Deletes-before-adds therefore holds within a single
        effect's ``resulting_fluents`` but not across an effect and its
        branches: a triggered sub-effect that deletes a fluent this effect
        adds leaves it absent.

        ``forall_effects`` are universally quantified conditional branches
        (PDDL ``forall``+``when``); see :class:`ForallEffect`. They expand
        into ``cond_effects``-style branches at grounding time, so operators
        using them must be grounded through ``Operator.instantiate`` (which
        supplies the object universe).
        """
        self.time = _make_bindable(time)
        self.prob_effects = [
            (_make_bindable(prob), effects) for prob, effects in prob_effects or []
        ]
        self.resulting_fluents = resulting_fluents if resulting_fluents is not None else set()
        self.cond_effects = cond_effects if cond_effects is not None else []
        self.forall_effects = forall_effects if forall_effects is not None else []
        self.is_probabilistic = bool(self.prob_effects)
        self.is_conditional = bool(self.cond_effects) or bool(self.forall_effects)

    def _ground(
        self,
        binding: Binding,
        objects_by_type: Optional[Mapping[str, Collection[str]]] = None,
    ) -> "GroundedEffect":
        def ground_fluent(f: Fluent, b: Binding = binding) -> Fluent:
            return Fluent(
                f.name, *[b.get(arg, arg) for arg in f.args], negated=f.negated
            )

        if self.is_probabilistic:
            grounded_prob_effects = tuple(
                (
                    prob(binding),
                    tuple(e._ground(binding, objects_by_type) for e in effect_list),
                )
                for prob, effect_list in self.prob_effects
            )
        else:
            grounded_prob_effects = tuple()

        grounded_cond_effects = [
            (
                {ground_fluent(f) for f in conditions},
                tuple(e._ground(binding, objects_by_type) for e in effect_list),
            )
            for conditions, effect_list in self.cond_effects
        ]

        for forall in self.forall_effects:
            if objects_by_type is None:
                raise ValueError(
                    "Effect has forall_effects, which require the object "
                    "universe to expand; ground via Operator.instantiate() "
                    "or pass objects_by_type to _ground()."
                )
            for _, typ in forall.variables:
                if typ not in objects_by_type:
                    raise ValueError(
                        f"forall_effects quantify over type {typ!r}, which is "
                        f"missing from objects_by_type "
                        f"(known types: {sorted(objects_by_type)})."
                    )
            domains = [sorted(objects_by_type[typ]) for _, typ in forall.variables]
            for combo in itertools.product(*domains):
                # Quantified variables shadow same-named outer parameters.
                quantified = dict(binding)
                quantified.update(
                    {var: obj for (var, _), obj in zip(forall.variables, combo)}
                )
                grounded_cond_effects.append(
                    (
                        {ground_fluent(f, quantified) for f in forall.conditions},
                        tuple(
                            e._ground(quantified, objects_by_type)
                            for e in forall.effects
                        ),
                    )
                )

        grounded_time: float = self.time(binding)
        grounded_resulting_fluents = {
            ground_fluent(f) for f in self.resulting_fluents
        }

        return GroundedEffect(
            grounded_time,
            prob_effects=grounded_prob_effects,
            resulting_fluents=grounded_resulting_fluents,
            cond_effects=tuple(grounded_cond_effects),
        )


class Eq:
    """Grounding-time equality constraint (PDDL ``(= ?x ?y)``).

    Placed in an Operator's ``preconditions`` list alongside Fluents; each
    term is a variable (``?x``) or an object name. Evaluated while grounding
    (see :func:`ground_operators`) and never appears on the grounded Action.
    ``Operator.instantiate`` has no machinery for constraints and raises if
    an operator carries one.
    """

    __slots__ = ("left", "right", "negated")

    def __init__(self, left: str, right: str, negated: bool = False):
        self.left = left
        self.right = right
        self.negated = negated

    def __repr__(self) -> str:
        op = "!=" if self.negated else "=="
        return f"<{type(self).__name__} {self.left} {op} {self.right}>"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Eq)
            and (self.left, self.right, self.negated)
            == (other.left, other.right, other.negated)
        )

    def __hash__(self) -> int:
        return hash((self.left, self.right, self.negated))


def Neq(left: str, right: str) -> Eq:
    """PDDL ``(not (= ?x ?y))``: the terms must bind different objects."""
    return Eq(left, right, negated=True)


class Operator:
    def __init__(
        self,
        name: str,
        parameters: List[Tuple[str, str]],
        preconditions: Sequence[Union[Fluent, Eq]],
        effects: Sequence[Effect],
        extra_cost: float = 0.0,
    ):
        self.name = name
        self.parameters = parameters
        # Grounding-time constraints (Eq/Neq) live beside ordinary fluent
        # preconditions in the input list, PDDL-style, but are split out:
        # they constrain which bindings exist rather than when actions apply.
        self.preconditions = [p for p in preconditions if isinstance(p, Fluent)]
        self.grounding_constraints = [p for p in preconditions if isinstance(p, Eq)]
        self.effects = effects
        self.extra_cost = extra_cost

    def instantiate(self, objects_by_type: Mapping[str, Collection[str]]) -> List[Action]:
        if self.grounding_constraints:
            raise TypeError(
                f"Operator {self.name!r} has grounding constraints (Eq/Neq), "
                "which Operator.instantiate does not evaluate; ground it via "
                "ground_operators() instead."
            )
        grounded_actions = []
        domains = [objects_by_type[typ] for _, typ in self.parameters]
        for assignment in itertools.product(*domains):
            binding = {var: obj for (var, _), obj in zip(self.parameters, assignment)}
            if len(set(binding.values())) != len(binding):
                continue
            grounded_actions.append(self._ground(binding, objects_by_type))
        return grounded_actions

    def _ground(
        self,
        binding: Dict[str, str],
        objects_by_type: Optional[Mapping[str, Collection[str]]] = None,
    ) -> Action:
        def evaluate(value):
            return value(binding) if callable(value) else value

        grounded_preconditions = frozenset(
            self._substitute_fluent(f, binding) for f in self.preconditions
        )
        grounded_effects = [eff._ground(binding, objects_by_type) for eff in self.effects]

        name_str = " ".join(
            [self.name] + [binding[var] for var, _ in self.parameters]
        )
        return Action(grounded_preconditions, grounded_effects, name=name_str, extra_cost=self.extra_cost)

    def _substitute_fluent(self, fluent: Fluent, binding: Dict[str, str]) -> Fluent:
        grounded_args = tuple(binding.get(arg, arg) for arg in fluent.args)
        return Fluent(fluent.name, *grounded_args, negated=fluent.negated)


def get_action_by_name(actions: List[Action], name: str) -> Action:
    for action in actions:
        if action.name == name:
            return action
    raise ValueError(f"No action found with name: {name}")


# ============================================================================
#  Grounding with static-precondition pruning
#  (design: docs/design/static-grounding.md)
# ============================================================================


def _is_var(term: str) -> bool:
    return term.startswith("?")


def _collect_dynamic_predicates(effect: Effect, out: Set[str]) -> None:
    for fluent in effect.resulting_fluents:
        out.add(fluent.name)
    for _, sub_effects in effect.prob_effects:
        for sub in sub_effects:
            _collect_dynamic_predicates(sub, out)
    # Conditional branches: only the effect side writes; conditions read.
    for _, sub_effects in effect.cond_effects:
        for sub in sub_effects:
            _collect_dynamic_predicates(sub, out)
    for forall in effect.forall_effects:
        for sub in forall.effects:
            _collect_dynamic_predicates(sub, out)


def dynamic_predicates(operators: Sequence[Operator]) -> Set[str]:
    """Predicates touched by some effect of some operator (all others are
    static: their truth is fixed for the lifetime of the problem)."""
    dynamic: Set[str] = set()
    for op in operators:
        for effect in op.effects:
            _collect_dynamic_predicates(effect, dynamic)
    return dynamic


class GroundingStats:
    """Enumeration counters from one ground_operators() call."""

    __slots__ = ("nominal_bindings", "visited_bindings", "actions_kept")

    def __init__(self) -> None:
        self.nominal_bindings = 0
        self.visited_bindings = 0
        self.actions_kept = 0

    def __repr__(self) -> str:
        return (
            f"<GroundingStats nominal={self.nominal_bindings} "
            f"visited={self.visited_bindings} kept={self.actions_kept}>"
        )


class GroundingResult:
    __slots__ = ("actions", "static_predicates", "stats")

    def __init__(
        self,
        actions: List[Action],
        static_predicates: Collection[str],
        stats: GroundingStats,
    ) -> None:
        self.actions = actions
        self.static_predicates = frozenset(static_predicates)
        self.stats = stats


def ground_operators(
    operators: Sequence[Operator],
    objects_by_type: Mapping[str, Collection[str]],
    initial_fluents: Collection[Fluent],
    *,
    allow_duplicate_bindings: bool = False,
    skip_on: Tuple[type[BaseException], ...] = (),
    assert_static: Optional[Collection[str]] = None,
    treat_dynamic: Optional[Collection[str]] = None,
) -> GroundingResult:
    """Ground operators via backtracking with static-precondition pruning.

    A predicate is *static* if no effect of any operator in ``operators``
    touches it (see :func:`dynamic_predicates`); positive preconditions on
    static predicates, and Eq/Neq constraints, are evaluated against
    ``initial_fluents`` as soon as their variables are bound, pruning the
    enumeration early. Negated static preconditions are left as runtime
    preconditions (they rarely prune).

    Args:
        allow_duplicate_bindings: if True, the same object may bind several
            parameters (PDDL semantics; express distinctness with Neq). If
            False, replicates ``Operator.instantiate``'s all-distinct rule.
        skip_on: exception types that, when raised while grounding one
            binding (e.g. by a duration callable), skip that binding only.
        assert_static: predicates the caller believes are static; raises
            ValueError if some effect touches one (catching mis-assumptions
            loudly instead of silently mis-pruning).
        treat_dynamic: predicates to force dynamic even though no operator
            effect touches them — required when the execution environment
            mutates them out-of-band (e.g. ObjectSearchEnvironment's
            revelation adds ``revealed``/``found``/``at`` fluents).

    Actions with non-finite effect times are dropped (an ``inf`` duration
    marks a statically impossible action, e.g. a move between unknown
    locations).
    """
    dynamic = dynamic_predicates(operators)
    if treat_dynamic:
        dynamic |= set(treat_dynamic)
    if assert_static:
        violations = sorted(set(assert_static) & dynamic)
        if violations:
            raise ValueError(
                f"assert_static predicates are touched by effects: {violations}"
            )

    precondition_predicates = {
        f.name for op in operators for f in op.preconditions
    }
    static_predicates = precondition_predicates - dynamic

    # Fact index for O(1) static checks: predicate -> set of arg tuples.
    facts: Dict[str, Set[Tuple[str, ...]]] = {}
    for fluent in initial_fluents:
        if fluent.name in static_predicates and not fluent.negated:
            facts.setdefault(fluent.name, set()).add(tuple(fluent.args))

    stats = GroundingStats()
    actions: List[Action] = []
    for op in operators:
        actions.extend(
            _ground_one_operator(
                op,
                objects_by_type,
                facts,
                static_predicates,
                allow_duplicate_bindings,
                skip_on,
                stats,
            )
        )

    actions = [
        a
        for a in actions
        if all(math.isfinite(eff.time) for eff in a.effects)
    ]
    stats.actions_kept = len(actions)
    return GroundingResult(actions, static_predicates, stats)


def _ground_one_operator(
    op: Operator,
    objects_by_type: Mapping[str, Collection[str]],
    facts: Dict[str, Set[Tuple[str, ...]]],
    static_predicates: Set[str],
    allow_duplicate_bindings: bool,
    skip_on: Tuple[type[BaseException], ...],
    stats: GroundingStats,
) -> List[Action]:
    """Backtracking enumeration for one operator (see ground_operators)."""
    params = op.parameters
    domains = [sorted(objects_by_type.get(typ, set())) for _, typ in params]
    stats.nominal_bindings += math.prod(len(d) for d in domains)

    # Checks as (needed_vars, check) where check is an Eq or a static Fluent.
    checks: List[Tuple[Set[str], Union[Eq, Fluent]]] = []
    for eq in op.grounding_constraints:
        checks.append(({t for t in (eq.left, eq.right) if _is_var(t)}, eq))
    for fluent in op.preconditions:
        if fluent.name in static_predicates and not fluent.negated:
            checks.append(({a for a in fluent.args if _is_var(a)}, fluent))

    order = _order_parameters(params, [needed for needed, _ in checks])

    # Assign each check to the earliest depth at which all its variables are
    # bound; variable-free checks are evaluated once up front.
    bound_after: List[Set[str]] = []
    acc: Set[str] = set()
    for pos in order:
        acc.add(params[pos][0])
        bound_after.append(set(acc))
    checks_at: List[List[Union[Eq, Fluent]]] = [[] for _ in order]
    always: List[Union[Eq, Fluent]] = []
    for needed, check in checks:
        if not needed:
            always.append(check)
            continue
        for depth, bound in enumerate(bound_after):
            if needed <= bound:
                checks_at[depth].append(check)
                break

    def satisfied(check: Union[Eq, Fluent], binding: Dict[str, str]) -> bool:
        if isinstance(check, Eq):
            left = binding.get(check.left, check.left)
            right = binding.get(check.right, check.right)
            return (left == right) != check.negated
        grounded = tuple(binding.get(a, a) for a in check.args)
        return grounded in facts.get(check.name, ())

    if not all(satisfied(c, {}) for c in always):
        return []

    actions: List[Action] = []
    binding: Dict[str, str] = {}
    bound_objects: Set[str] = set()

    def dfs(depth: int) -> None:
        if depth == len(order):
            try:
                actions.append(op._ground(binding, objects_by_type))
            except skip_on:
                pass  # this binding is not a real action (e.g. undefined cost)
            return
        pos = order[depth]
        var = params[pos][0]
        for obj in domains[pos]:
            if not allow_duplicate_bindings:
                if obj in bound_objects:
                    continue
                bound_objects.add(obj)
            stats.visited_bindings += 1
            binding[var] = obj
            if all(satisfied(c, binding) for c in checks_at[depth]):
                dfs(depth + 1)
            if not allow_duplicate_bindings:
                bound_objects.discard(obj)
        binding.pop(var, None)

    if all(len(d) > 0 for d in domains):
        dfs(0)
    return actions


def _order_parameters(
    params: List[Tuple[str, str]], constraint_vars: List[Set[str]]
) -> List[int]:
    """Greedy ordering: bind next the parameter completing the most checks."""
    remaining = list(range(len(params)))
    order: List[int] = []
    bound: Set[str] = set()
    while remaining:
        def gain(pos: int) -> int:
            would_bind = bound | {params[pos][0]}
            return sum(
                1
                for needed in constraint_vars
                if needed and needed <= would_bind and not needed <= bound
            )

        best = max(remaining, key=lambda pos: (gain(pos), -pos))
        order.append(best)
        bound.add(params[best][0])
        remaining.remove(best)
    return order


def get_next_actions(state: State, all_actions: List[Action]) -> List[Action]:
    # Step 1: Extract all `free(...)` fluents
    free_robot_fluents: List[Fluent] = sorted(
        [f for f in state.fluents if f.name == "free"], key=lambda f: str(f)
    )
    # neg_fluents = {~f for f in free_robot_fluents}
    neg_state = state.copy()
    neg_fluents: Set[Fluent] = {~f for f in free_robot_fluents}
    neg_state.update_fluents(neg_fluents)

    # Step 2: Check each robot individually
    for free_pred in free_robot_fluents:
        # Create a restricted version of the state
        combined_fluents: Set[Fluent] = neg_state.fluents | {free_pred}
        temp_state = State(
            time=state.time,
            fluents=combined_fluents,
        )

        # Step 3: Check for applicable actions
        applicable = [a for a in all_actions if temp_state.satisfies_precondition(a)]
        if applicable:
            return applicable

    # Step 4: Otherwise, return any possible actions
    return [a for a in all_actions if state.satisfies_precondition(a)]


# ============================================================================
# Negative Precondition Preprocessing Functions
# ============================================================================


def extract_negative_preconditions(actions: List[Action]) -> Set[Fluent]:
    """Extract all negative preconditions from a list of actions.

    Args:
        actions: List of Action objects

    Returns:
        Set of Fluent objects that appear as negative preconditions.
        These fluents are the "flipped" versions (i.e., the positive form).
        For example, if action has precondition ~F("hand_full r1"),
        this returns {F("hand_full r1")}.
    """
    negative_fluents = set()
    for action in actions:
        # _neg_precond_flipped contains the positive version of negative preconditions
        negative_fluents.update(action._neg_precond_flipped)
    return negative_fluents


def extract_negative_goal_fluents(goal: Goal) -> Set[Fluent]:
    """Extract all negative fluents from a Goal object.

    This is needed to extend the negative-to-positive mapping to include
    goal fluents, not just action precondition fluents.

    Args:
        goal: A Goal object (LiteralGoal, AndGoal, OrGoal, etc.)

    Returns:
        Set of positive Fluent objects that appear negated in the goal.
        For example, if goal has ~F("at Book table"), returns {F("at Book table")}.
    """
    from railroad._bindings import GoalType

    negative_fluents = set()
    goal_type = goal.get_type()

    if isinstance(goal, LiteralGoal):
        fluent = goal.fluent()
        if fluent.negated:
            # Return the positive form
            negative_fluents.add(~fluent)
    elif goal_type in (GoalType.AND, GoalType.OR):
        for child in goal.children():
            negative_fluents.update(extract_negative_goal_fluents(child))

    return negative_fluents


def extract_all_negative_fluents(
    actions: Optional[List[Action]] = None,
    goal: Optional[Goal] = None
) -> Set[Fluent]:
    """Extract negative fluents from actions and/or goals.

    This unified function collects all fluents that appear in negated form,
    either as negative preconditions in actions or as negative literals in goals.
    The returned fluents are in positive form (ready for mapping creation).

    Args:
        actions: Optional list of Action objects
        goal: Optional Goal object

    Returns:
        Set of positive Fluent objects that appear negated in actions/goals.
    """
    negative_fluents = set()

    if actions:
        negative_fluents.update(extract_negative_preconditions(actions))

    if goal:
        negative_fluents.update(extract_negative_goal_fluents(goal))

    return negative_fluents


def create_positive_fluent_mapping(negative_fluents: Set[Fluent]) -> Dict[Fluent, Fluent]:
    """Create mapping from negative fluents to their positive "not-" versions.

    Args:
        negative_fluents: Set of fluents that appear in negative preconditions

    Returns:
        Dictionary mapping each fluent to its "not-" version.
        For example: F("hand_full r1") -> F("not-hand_full r1")
    """
    mapping = {}
    for fluent in negative_fluents:
        # Create positive version with "not-" prefix
        not_name = f"not-{fluent.name}"
        not_fluent = Fluent(not_name, *fluent.args)
        mapping[fluent] = not_fluent
    return mapping


def _augment_fluents_for_mapping(
    fluents: Set[Fluent], neg_to_pos_mapping: Dict[Fluent, Fluent]
) -> Set[Fluent]:
    """Augment a fluent set with "not-" bookkeeping fluents.

    Adding F("P") also removes F("not-P"); removing F("P") adds F("not-P").
    """
    augmented = set(fluents)
    for fluent in fluents:
        if fluent.negated:
            positive_fluent = ~fluent
            if positive_fluent in neg_to_pos_mapping:
                augmented.add(neg_to_pos_mapping[positive_fluent])
        else:
            if fluent in neg_to_pos_mapping:
                augmented.add(~neg_to_pos_mapping[fluent])
    return augmented


def _augment_grounded_effect_for_mapping(
    effect: GroundedEffect, neg_to_pos_mapping: Dict[Fluent, Fluent]
) -> GroundedEffect:
    """Recursively augment a GroundedEffect (and its probabilistic and
    conditional branches) with "not-" bookkeeping fluents.

    Conditional branch *conditions* are left untouched: they read the state
    directly with negation-as-absence, so they need no bookkeeping.
    """
    augmented_fluents = _augment_fluents_for_mapping(
        effect.resulting_fluents, neg_to_pos_mapping
    )
    converted_prob_effects = [
        (
            prob_branch.prob,
            [
                _augment_grounded_effect_for_mapping(e, neg_to_pos_mapping)
                for e in prob_branch.effects
            ],
        )
        for prob_branch in effect.prob_effects
    ]
    converted_cond_effects = [
        (
            cond_branch.conditions,
            [
                _augment_grounded_effect_for_mapping(e, neg_to_pos_mapping)
                for e in cond_branch.effects
            ],
        )
        for cond_branch in effect.cond_effects
    ]
    return GroundedEffect(
        time=effect.time,
        resulting_fluents=augmented_fluents,
        prob_effects=converted_prob_effects,
        cond_effects=converted_cond_effects,
    )


def convert_state_to_positive_preconditions(
    state: State,
    neg_to_pos_mapping: Dict[Fluent, Fluent]
) -> State:
    """Convert state to use positive versions of negative preconditions.

    For each negative precondition that could exist, adds the corresponding
    positive "not-" fluent if the original fluent is absent. Also converts
    upcoming effects to maintain consistency with the mapping.

    Args:
        state: Original state
        neg_to_pos_mapping: Mapping from fluents to their "not-" versions

    Returns:
        New State with additional positive fluents representing absence
        and converted upcoming effects.
        For example, if F("hand_full r1") is not in state.fluents,
        adds F("not-hand_full r1") to indicate hand is not full.
    """
    new_fluents = set(state.fluents)

    for original_fluent, not_fluent in neg_to_pos_mapping.items():
        # If the original fluent is NOT in the state, add the "not-" version
        if original_fluent not in state.fluents:
            new_fluents.add(not_fluent)

    # Convert all upcoming effects (which are tuples of (time, effect))
    converted_effects = [
        (time, _augment_grounded_effect_for_mapping(eff, neg_to_pos_mapping))
        for time, eff in state.upcoming_effects
    ]

    return State(time=state.time, fluents=new_fluents, upcoming_effects=converted_effects)


def convert_action_to_positive_preconditions(
    action: Action,
    neg_to_pos_mapping: Dict[Fluent, Fluent]
) -> Action:
    """Convert action's negative preconditions to positive "not-" versions.

    Args:
        action: Original action with negative preconditions
        neg_to_pos_mapping: Mapping from fluents to their "not-" versions

    Returns:
        New Action with negative preconditions replaced by positive ones.
        For example, precondition ~F("hand_full r1") becomes F("not-hand_full r1").
    """
    new_preconditions = set()

    # Add all positive preconditions as-is
    new_preconditions.update(action._pos_precond)

    # Replace negative preconditions with positive "not-" versions
    for neg_fluent in action._neg_precond_flipped:
        if neg_fluent in neg_to_pos_mapping:
            new_preconditions.add(neg_to_pos_mapping[neg_fluent])
        else:
            # If not in mapping, keep as negative (shouldn't happen after preprocessing)
            new_preconditions.add(~neg_fluent)

    # Create new action with converted preconditions
    return Action(new_preconditions, action.effects, name=action.name, extra_cost=action.extra_cost)


def convert_action_effects(
    action: Action,
    neg_to_pos_mapping: Dict[Fluent, Fluent]
) -> Action:
    """Convert action's effects to maintain consistency with positive preconditions.

    When an effect adds or removes a fluent that has a negative precondition mapping,
    this function adds the corresponding "not-" fluent to maintain consistency.

    For example:
    - If effect adds F("hand_full"), also add ~F("not-hand_full")
    - If effect removes F("hand_full") (i.e., ~F("hand_full")), also add F("not-hand_full")

    Args:
        action: Original action
        neg_to_pos_mapping: Mapping from fluents to their "not-" versions

    Returns:
        New Action with augmented effects
    """
    converted_effects = [
        _augment_grounded_effect_for_mapping(eff, neg_to_pos_mapping)
        for eff in action.effects
    ]

    # Create new action with converted effects
    return Action(action.preconditions, converted_effects, name=action.name, extra_cost=action.extra_cost)


def preprocess_actions_for_relaxed_planning(
    actions: List[Action],
    initial_state: State
) -> Tuple[List[Action], State, Dict[Fluent, Fluent]]:
    """Preprocess actions and state to convert negative preconditions to positive.

    This is a one-time preprocessing step that should be done after actions
    are instantiated. The resulting actions and state can then be used for
    planning with algorithms like FF heuristic that work better with positive
    preconditions.

    Args:
        actions: List of instantiated actions
        initial_state: Initial state of the planning problem

    Returns:
        Tuple of (converted_actions, converted_state, mapping_dict):
        - converted_actions: Actions with negative preconditions replaced
        - converted_state: State with additional positive fluents
        - mapping_dict: Mapping used for conversion (for debugging/inspection)
    """
    # Step 1: Extract all negative preconditions
    negative_fluents = extract_negative_preconditions(actions)

    # Step 2: Create mapping to positive "not-" versions
    neg_to_pos_mapping = create_positive_fluent_mapping(negative_fluents)

    # Step 3: Convert all actions (preconditions and effects)
    converted_actions = []
    for action in actions:
        # First convert preconditions
        action_with_preconds = convert_action_to_positive_preconditions(action, neg_to_pos_mapping)
        # Then convert effects
        action_with_effects = convert_action_effects(action_with_preconds, neg_to_pos_mapping)
        converted_actions.append(action_with_effects)

    # Step 4: Convert initial state
    converted_state = convert_state_to_positive_preconditions(initial_state, neg_to_pos_mapping)

    return converted_actions, converted_state, neg_to_pos_mapping


def convert_goal_to_positive_preconditions(
    goal,  # Goal type from bindings
    neg_to_pos_mapping: Dict[Fluent, Fluent]
):
    """Convert a Goal's negative fluents to positive "not-" equivalents.

    This is necessary when using Goals with the MCTSPlanner, which converts
    negative preconditions to positive forms internally. Without this conversion,
    the heuristic function won't correctly evaluate goal literals.

    Args:
        goal: A Goal object (LiteralGoal, AndGoal, OrGoal, etc.)
        neg_to_pos_mapping: Mapping from fluents to their "not-" versions

    Returns:
        A new Goal with converted fluents
    """
    from railroad._bindings import (
        GoalType,
        LiteralGoal,
        TrueGoal,
        FalseGoal,
    )

    goal_type = goal.get_type()

    if goal_type == GoalType.TRUE_GOAL:
        return TrueGoal()
    elif goal_type == GoalType.FALSE_GOAL:
        return FalseGoal()
    elif goal_type == GoalType.LITERAL:
        fluent = goal.fluent()
        converted_fluent = _convert_fluent(fluent, neg_to_pos_mapping)
        return LiteralGoal(converted_fluent)
    elif goal_type == GoalType.AND:
        converted_children = [
            convert_goal_to_positive_preconditions(child, neg_to_pos_mapping)
            for child in goal.children()
        ]
        return AndGoal(converted_children)
    elif goal_type == GoalType.OR:
        converted_children = [
            convert_goal_to_positive_preconditions(child, neg_to_pos_mapping)
            for child in goal.children()
        ]
        return OrGoal(converted_children)
    else:
        # Unknown goal type, return as-is
        return goal


def _convert_fluent(
    fluent: Fluent,
    neg_to_pos_mapping: Dict[Fluent, Fluent]
) -> Fluent:
    """Convert a single fluent using the negative-to-positive mapping.

    Handles the conversion of negative fluents like ~F("P") to F("not-P").
    """
    if fluent.negated:
        # Fluent is ~F("P") - we want F("not-P")
        positive_fluent = ~fluent  # Get F("P")
        if positive_fluent in neg_to_pos_mapping:
            # Return F("not-P") instead of ~F("P")
            return neg_to_pos_mapping[positive_fluent]
    # No conversion needed
    return fluent

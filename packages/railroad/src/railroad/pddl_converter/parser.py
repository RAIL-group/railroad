"""Parser for classical PDDL and PPDDL (probabilistic) domains/problems.

The parser is deliberately permissive about *structure* — conditions and
effects are kept as small AST nodes — so that the converter, not the parser,
decides what railroad can represent. Features that cannot even be represented
in the AST (durative actions, derived predicates, conditional effects, ...)
raise :class:`UnsupportedPDDLError` here with a machine-readable reason.

PDDL is case-insensitive; all symbols are normalized to lowercase.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .errors import PDDLParseError, UnsupportedPDDLError

# An s-expression: either a symbol or a nested list of s-expressions.
Sexpr = Union[str, List["Sexpr"]]

OBJECT_TYPE = "object"


# ============================================================================
#  S-expression tokenizer / reader
# ============================================================================


def tokenize(text: str) -> List[str]:
    """Split PDDL text into tokens, stripping ``;`` comments."""
    tokens: List[str] = []
    current: List[str] = []
    in_comment = False
    for ch in text:
        if in_comment:
            if ch == "\n":
                in_comment = False
            continue
        if ch == ";":
            in_comment = True
            ch = " "
        if ch in "()":
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(ch)
        elif ch.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch.lower())
    if current:
        tokens.append("".join(current))
    return tokens


def read_sexprs(text: str) -> List[Sexpr]:
    """Parse text into a list of top-level s-expressions."""
    tokens = tokenize(text)
    stack: List[List[Sexpr]] = [[]]
    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) == 1:
                raise PDDLParseError("Unbalanced ')'")
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok)
    if len(stack) != 1:
        raise PDDLParseError("Unbalanced '(' — truncated input?")
    return stack[0]


def _head(sexpr: Sexpr) -> Optional[str]:
    if isinstance(sexpr, list) and sexpr and isinstance(sexpr[0], str):
        return sexpr[0]
    return None


# ============================================================================
#  AST nodes
# ============================================================================

# (var, type) pairs; types default to "object".
TypedVars = List[Tuple[str, str]]


@dataclass(frozen=True)
class Literal:
    predicate: str
    args: Tuple[str, ...]
    negated: bool = False


@dataclass(frozen=True)
class Equals:
    left: str
    right: str
    negated: bool = False


@dataclass
class And:
    children: List["ConditionNode"]


@dataclass
class Or:
    children: List["ConditionNode"]


@dataclass
class Forall:
    variables: TypedVars
    body: "ConditionNode"


@dataclass
class Exists:
    variables: TypedVars
    body: "ConditionNode"


ConditionNode = Union[Literal, Equals, And, Or, Forall, Exists]


@dataclass
class EffectAnd:
    children: List["EffectNode"]


@dataclass
class EffectForall:
    variables: TypedVars
    body: "EffectNode"


@dataclass
class Probabilistic:
    # (probability, effect) branches. Probabilities are constants; if they sum
    # to < 1, the remainder is an implicit "nothing happens" branch.
    branches: List[Tuple[float, "EffectNode"]]


@dataclass
class Increase:
    """``(increase (<function> <args>) <amount>)``.

    ``amount`` is either a constant or a ``(function, args)`` reference whose
    value is looked up in the problem's ``:init``.
    """

    function: str
    args: Tuple[str, ...]
    amount: Union[float, Tuple[str, Tuple[str, ...]]]


EffectNode = Union[Literal, EffectAnd, EffectForall, Probabilistic, Increase]


# ============================================================================
#  Domain / problem containers
# ============================================================================


@dataclass
class PDDLActionDef:
    name: str
    parameters: TypedVars
    precondition: Optional[ConditionNode]
    effect: EffectNode


@dataclass
class PDDLDomain:
    name: str
    requirements: List[str] = field(default_factory=list)
    # type -> parent type ("object" is the root and its own parent).
    types: Dict[str, str] = field(default_factory=dict)
    # constant name -> type
    constants: Dict[str, str] = field(default_factory=dict)
    # predicate name -> arity
    predicates: Dict[str, int] = field(default_factory=dict)
    # function name -> arity (only ever consumed for cost lookup)
    functions: Dict[str, int] = field(default_factory=dict)
    actions: List[PDDLActionDef] = field(default_factory=list)


@dataclass
class PDDLProblem:
    name: str
    domain_name: str
    # object name -> type
    objects: Dict[str, str] = field(default_factory=dict)
    init_literals: List[Literal] = field(default_factory=list)
    # (function, args) -> value, from  (= (fn a b) 3)  entries in :init
    init_function_values: Dict[Tuple[str, Tuple[str, ...]], float] = field(
        default_factory=dict
    )
    goal: Optional[ConditionNode] = None
    # ("minimize"|"maximize", expression sexpr rendered as a string)
    metric: Optional[Tuple[str, str]] = None
    goal_reward: Optional[float] = None


# ============================================================================
#  Shared helpers
# ============================================================================


def _parse_typed_list(items: List[Sexpr], what: str) -> List[Tuple[str, str]]:
    """Parse a PDDL typed list ``a b - t1 c - t2 d`` into (name, type) pairs.

    Trailing names with no ``- type`` default to :data:`OBJECT_TYPE`.
    """
    result: List[Tuple[str, str]] = []
    pending: List[str] = []
    i = 0
    while i < len(items):
        item = items[i]
        if item == "-":
            if i + 1 >= len(items):
                raise PDDLParseError(f"Dangling '-' in {what} typed list")
            type_expr = items[i + 1]
            if isinstance(type_expr, list):
                if _head(type_expr) == "either":
                    raise UnsupportedPDDLError(
                        "either-types", f"(either ...) type in {what}"
                    )
                raise PDDLParseError(f"Unexpected compound type in {what}")
            result.extend((name, type_expr) for name in pending)
            pending = []
            i += 2
        elif isinstance(item, str):
            pending.append(item)
            i += 1
        else:
            raise PDDLParseError(f"Unexpected nested list in {what} typed list")
    result.extend((name, OBJECT_TYPE) for name in pending)
    return result


def _parse_atom(sexpr: Sexpr, context: str) -> Tuple[str, Tuple[str, ...]]:
    if not isinstance(sexpr, list) or not sexpr:
        raise PDDLParseError(f"Expected atom in {context}, got {sexpr!r}")
    name = sexpr[0]
    if not isinstance(name, str):
        raise PDDLParseError(f"Expected predicate name in {context}")
    args = []
    for a in sexpr[1:]:
        if not isinstance(a, str):
            raise UnsupportedPDDLError(
                "object-fluents", f"non-symbol argument in {context}: {a!r}"
            )
        args.append(a)
    return name, tuple(args)


_NUMERIC_COMPARISONS = {"<", ">", "<=", ">="}
_NUMERIC_OPS = {"+", "-", "*", "/"}


def parse_condition(sexpr: Sexpr, context: str) -> ConditionNode:
    """Parse a goal-description (precondition/goal) tree."""
    head = _head(sexpr)
    if head == "and":
        assert isinstance(sexpr, list)
        return And([parse_condition(c, context) for c in sexpr[1:]])
    if head == "or":
        assert isinstance(sexpr, list)
        return Or([parse_condition(c, context) for c in sexpr[1:]])
    if head == "not":
        assert isinstance(sexpr, list)
        if len(sexpr) != 2:
            raise PDDLParseError(f"(not ...) takes one argument in {context}")
        inner = parse_condition(sexpr[1], context)
        if isinstance(inner, Literal):
            return Literal(inner.predicate, inner.args, negated=not inner.negated)
        if isinstance(inner, Equals):
            return Equals(inner.left, inner.right, negated=not inner.negated)
        raise UnsupportedPDDLError(
            "negated-compound-condition",
            f"(not ...) over a non-literal in {context}",
        )
    if head in ("forall", "exists"):
        assert isinstance(sexpr, list)
        if len(sexpr) != 3 or not isinstance(sexpr[1], list):
            raise PDDLParseError(f"Malformed ({head} (vars) body) in {context}")
        variables = _parse_typed_list(sexpr[1], f"{head} in {context}")
        body = parse_condition(sexpr[2], context)
        return (Forall if head == "forall" else Exists)(variables, body)
    if head == "imply":
        raise UnsupportedPDDLError("imply-conditions", f"(imply ...) in {context}")
    if head == "preference":
        raise UnsupportedPDDLError("preferences", f"(preference ...) in {context}")
    if head in _NUMERIC_COMPARISONS:
        raise UnsupportedPDDLError(
            "numeric-conditions", f"({head} ...) in {context}"
        )
    if head == "=":
        assert isinstance(sexpr, list)
        if len(sexpr) != 3:
            raise PDDLParseError(f"(= ...) takes two arguments in {context}")
        left, right = sexpr[1], sexpr[2]
        if isinstance(left, list) or isinstance(right, list):
            raise UnsupportedPDDLError(
                "numeric-conditions", f"numeric (= ...) in {context}"
            )
        return Equals(left, right)
    name, args = _parse_atom(sexpr, context)
    return Literal(name, args)


def _parse_number(tok: Sexpr, context: str) -> float:
    if not isinstance(tok, str):
        raise PDDLParseError(f"Expected a number in {context}, got {tok!r}")
    try:
        if "/" in tok:  # PPDDL allows rational probabilities like 1/2
            num, den = tok.split("/", 1)
            return float(num) / float(den)
        return float(tok)
    except (ValueError, ZeroDivisionError) as exc:
        raise PDDLParseError(f"Bad number {tok!r} in {context}") from exc


def parse_effect(sexpr: Sexpr, context: str) -> EffectNode:
    """Parse an effect tree."""
    head = _head(sexpr)
    if head == "and":
        assert isinstance(sexpr, list)
        return EffectAnd([parse_effect(c, context) for c in sexpr[1:]])
    if head == "not":
        assert isinstance(sexpr, list)
        if len(sexpr) != 2:
            raise PDDLParseError(f"(not ...) takes one argument in {context}")
        name, args = _parse_atom(sexpr[1], context)
        return Literal(name, args, negated=True)
    if head == "forall":
        assert isinstance(sexpr, list)
        if len(sexpr) != 3 or not isinstance(sexpr[1], list):
            raise PDDLParseError(f"Malformed (forall (vars) effect) in {context}")
        variables = _parse_typed_list(sexpr[1], f"forall in {context}")
        return EffectForall(variables, parse_effect(sexpr[2], context))
    if head == "when":
        raise UnsupportedPDDLError(
            "conditional-effects", f"(when ...) effect in {context}"
        )
    if head == "probabilistic":
        assert isinstance(sexpr, list)
        rest = sexpr[1:]
        if len(rest) % 2 != 0:
            raise PDDLParseError(f"(probabilistic ...) needs prob/effect pairs in {context}")
        branches = []
        for i in range(0, len(rest), 2):
            prob = _parse_number(rest[i], f"probabilistic in {context}")
            branches.append((prob, parse_effect(rest[i + 1], context)))
        return Probabilistic(branches)
    if head == "oneof":
        raise UnsupportedPDDLError(
            "oneof-nondeterminism", f"(oneof ...) effect in {context}"
        )
    if head in ("increase", "decrease"):
        assert isinstance(sexpr, list)
        if head == "decrease":
            raise UnsupportedPDDLError(
                "numeric-effects", f"(decrease ...) in {context}"
            )
        if len(sexpr) != 3:
            raise PDDLParseError(f"(increase ...) takes two arguments in {context}")
        target = sexpr[1]
        if isinstance(target, str):  # PPDDL allows bare `reward`
            fn_name, fn_args = target, ()
        else:
            fn_name, fn_args = _parse_atom(target, context)
        amount = sexpr[2]
        if isinstance(amount, str):
            parsed_amount: Union[float, Tuple[str, Tuple[str, ...]]] = _parse_number(
                amount, context
            )
        else:
            amt_head = _head(amount)
            if amt_head in _NUMERIC_OPS:
                raise UnsupportedPDDLError(
                    "numeric-effects", f"arithmetic cost expression in {context}"
                )
            parsed_amount = _parse_atom(amount, context)
        return Increase(fn_name, tuple(fn_args), parsed_amount)
    if head in ("assign", "scale-up", "scale-down"):
        raise UnsupportedPDDLError("numeric-effects", f"({head} ...) in {context}")
    name, args = _parse_atom(sexpr, context)
    return Literal(name, args)


# ============================================================================
#  Domain parsing
# ============================================================================


def _find_define(top: List[Sexpr], kind: str) -> Optional[List[Sexpr]]:
    """Find ``(define (<kind> ...) ...)`` among top-level s-expressions.

    Some benchmark files (notably IPPC-2008) bundle a domain define and a
    problem define in one file, so we match on the header kind rather than
    taking the first define.
    """
    for sexpr in top:
        if _head(sexpr) != "define" or not isinstance(sexpr, list):
            continue
        header = sexpr[1] if len(sexpr) > 1 else None
        if isinstance(header, list) and _head(header) == kind and len(header) >= 2:
            return sexpr
    return None


def parse_domain(text: str) -> PDDLDomain:
    top = read_sexprs(text)
    define = _find_define(top, "domain")
    if define is None:
        raise PDDLParseError("No (define (domain <name>) ...) found")
    header = define[1]
    assert isinstance(header, list)
    domain = PDDLDomain(name=str(header[1]))

    for section in define[2:]:
        head = _head(section)
        if head is None or not isinstance(section, list):
            raise PDDLParseError(f"Unexpected domain section: {section!r}")
        if head == ":requirements":
            domain.requirements = [str(r) for r in section[1:]]
        elif head == ":types":
            for name, parent in _parse_typed_list(section[1:], ":types"):
                domain.types[name] = parent
        elif head == ":constants":
            for name, typ in _parse_typed_list(section[1:], ":constants"):
                domain.constants[name] = typ
        elif head == ":predicates":
            for pred in section[1:]:
                if not isinstance(pred, list) or not pred or not isinstance(pred[0], str):
                    raise PDDLParseError(f"Malformed predicate declaration: {pred!r}")
                params = _parse_typed_list(pred[1:], f"predicate {pred[0]}")
                domain.predicates[pred[0]] = len(params)
        elif head == ":functions":
            for fn in section[1:]:
                if fn == "-" or (isinstance(fn, str) and fn == "number"):
                    continue  # trailing "- number" type annotation
                if not isinstance(fn, list) or not fn or not isinstance(fn[0], str):
                    raise PDDLParseError(f"Malformed function declaration: {fn!r}")
                params = _parse_typed_list(fn[1:], f"function {fn[0]}")
                domain.functions[fn[0]] = len(params)
        elif head == ":action":
            domain.actions.append(_parse_action(section))
        elif head == ":durative-action":
            raise UnsupportedPDDLError(
                "durative-actions", f"durative action in domain {domain.name}"
            )
        elif head == ":derived":
            raise UnsupportedPDDLError(
                "derived-predicates", f"derived predicate in domain {domain.name}"
            )
        elif head == ":constraints":
            raise UnsupportedPDDLError(
                "constraints", f":constraints in domain {domain.name}"
            )
        # Unknown sections (e.g. PPDDL extensions we can ignore) fall through.
    return domain


def _parse_action(section: List[Sexpr]) -> PDDLActionDef:
    if len(section) < 2 or not isinstance(section[1], str):
        raise PDDLParseError("(:action ...) missing a name")
    name = section[1]
    parameters: TypedVars = []
    precondition: Optional[ConditionNode] = None
    effect: Optional[EffectNode] = None
    i = 2
    while i < len(section):
        key = section[i]
        if not isinstance(key, str) or not key.startswith(":"):
            raise PDDLParseError(f"Expected :keyword in action {name}, got {key!r}")
        if i + 1 >= len(section):
            raise PDDLParseError(f"Missing value for {key} in action {name}")
        value = section[i + 1]
        if key == ":parameters":
            if not isinstance(value, list):
                raise PDDLParseError(f"Malformed :parameters in action {name}")
            parameters = _parse_typed_list(value, f"action {name} parameters")
        elif key == ":precondition":
            if value != []:  # `()` means no precondition
                precondition = parse_condition(value, f"action {name} precondition")
        elif key == ":effect":
            effect = parse_effect(value, f"action {name} effect")
        elif key == ":duration":
            raise UnsupportedPDDLError(
                "durative-actions", f":duration in action {name}"
            )
        else:
            raise UnsupportedPDDLError(
                f"action-key:{key}", f"unsupported key in action {name}"
            )
        i += 2
    if effect is None:
        raise PDDLParseError(f"Action {name} has no :effect")
    return PDDLActionDef(name, parameters, precondition, effect)


# ============================================================================
#  Problem parsing
# ============================================================================


def parse_problem(text: str) -> PDDLProblem:
    top = read_sexprs(text)
    define = _find_define(top, "problem")
    if define is None:
        raise PDDLParseError("No (define (problem <name>) ...) found")
    header = define[1]
    assert isinstance(header, list)
    problem = PDDLProblem(name=str(header[1]), domain_name="")

    for section in define[2:]:
        head = _head(section)
        if head is None or not isinstance(section, list):
            raise PDDLParseError(f"Unexpected problem section: {section!r}")
        if head == ":domain":
            problem.domain_name = str(section[1])
        elif head == ":requirements":
            pass
        elif head == ":objects":
            for name, typ in _parse_typed_list(section[1:], ":objects"):
                problem.objects[name] = typ
        elif head == ":init":
            _parse_init(section[1:], problem)
        elif head == ":goal":
            if len(section) != 2:
                raise PDDLParseError("(:goal ...) takes one condition")
            problem.goal = parse_condition(section[1], "goal")
        elif head == ":metric":
            if len(section) < 3 or not isinstance(section[1], str):
                raise PDDLParseError("Malformed (:metric direction expr)")
            problem.metric = (section[1], _render(section[2]))
        elif head == ":goal-reward":
            problem.goal_reward = _parse_number(section[1], ":goal-reward")
        else:
            raise UnsupportedPDDLError(
                f"problem-section:{head}", f"in problem {problem.name}"
            )
    if problem.goal is None:
        raise UnsupportedPDDLError("no-goal", f"problem {problem.name} has no :goal")
    return problem


def _parse_init(entries: List[Sexpr], problem: PDDLProblem) -> None:
    for entry in entries:
        head = _head(entry)
        if head == "=":
            assert isinstance(entry, list)
            if len(entry) != 3 or not isinstance(entry[1], list):
                raise PDDLParseError(f"Malformed init function value: {entry!r}")
            if isinstance(entry[2], str):
                fn, args = _parse_atom(entry[1], ":init")
                problem.init_function_values[(fn, args)] = _parse_number(
                    entry[2], ":init"
                )
            else:
                raise UnsupportedPDDLError(
                    "object-fluents", f"non-numeric init value: {entry!r}"
                )
        elif head == "not":
            # Closed-world: explicit negative init facts are redundant.
            continue
        elif head == "probabilistic":
            raise UnsupportedPDDLError("probabilistic-init", "in :init")
        elif head == "at" and isinstance(entry, list) and len(entry) == 3 and isinstance(entry[1], str) and entry[1].replace(".", "").isdigit():
            raise UnsupportedPDDLError("timed-initial-literals", f"{entry!r} in :init")
        else:
            name, args = _parse_atom(entry, ":init")
            problem.init_literals.append(Literal(name, args))


def _render(sexpr: Sexpr) -> str:
    if isinstance(sexpr, str):
        return sexpr
    return "(" + " ".join(_render(s) for s in sexpr) + ")"

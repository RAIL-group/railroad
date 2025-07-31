# Re-import required dependencies due to kernel reset
from typing import Callable, List, Tuple, Dict, Set, Union, Optional, Sequence
import itertools
import heapq

Num = Union[float, int]
Binding = Dict[str, str]
Bindable = Callable[[Binding], Num]
OptCallable = Union[Num, Callable[..., float]]
OptExpr = Union[float, Tuple[Callable[..., float], List[str]]]


def _make_bindable(opt_expr: OptExpr) -> Bindable:
    if isinstance(opt_expr, Num):
        return lambda *args: opt_expr
    else:
        fn = opt_expr[0]
        args = opt_expr[1]
        return lambda b: fn(*[b.get(arg, arg) for arg in args])


class Fluent:
    __slots__ = ("name", "args", "negated", "_hash")

    def __init__(self, name: str, *args: str, negated: bool = False):
        if args:
            self.name = name
            self.args = args
            if "not" == name:
                raise ValueError("Use the 'negated' argument or ~Fluent to negate.")
            self.negated = negated
        else:
            if negated:
                raise ValueError(
                    "Cannot both pass a full string and negated=True. Use 'not' or ~Fluent."
                )
            split = name.split(" ")
            if split[0] == "not":
                self.negated = True
                split = split[1:]
            else:
                self.negated = False
            self.name = split[0]
            self.args = tuple(split[1:])

        self._hash = hash((self.name, self.args, self.negated))

    def __str__(self) -> str:
        prefix = "not " if self.negated else ""
        return f"{prefix}{self.name} {' '.join(self.args)}"

    def __repr__(self) -> str:
        return f"F({self})"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Fluent) and self._hash == other._hash

    def __invert__(self) -> "Fluent":
        return Fluent(self.name, *self.args, negated=not self.negated)


class ProbBranch:
    def __init__(self, prob: float, effects: tuple["GroundedEffect", ...]):
        self.prob = prob
        self.effects = effects

    def __eq__(self, other):
        return (
            isinstance(other, ProbBranch)
            and self.prob == other.prob
            and self.effects == other.effects
        )

    def __hash__(self):
        return hash((self.prob, tuple(self.effects)))


class GroundedEffect:
    __slots__ = (
        "time",
        "prob_effects",
        "resulting_fluents",
        "is_probabilistic",
        "_hash",
    )

    def __init__(
        self,
        time: float,
        *,
        resulting_fluents: frozenset[Fluent] = frozenset(),
        prob_effects: tuple["ProbBranch", ...] = tuple(),
    ):
        self.time = time
        self.prob_effects = prob_effects
        self.resulting_fluents = resulting_fluents
        self.is_probabilistic = bool(self.prob_effects)
        self._hash: int = hash((self.time, self.resulting_fluents, self.prob_effects))

    def __repr__(self):
        return f"GroundedEffect({self})"

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, GroundedEffect) and self._hash == other._hash

    def __str__(self):
        out_str = ""
        if self.is_probabilistic:
            parts = []
            for branch in self.prob_effects:
                effs = "; ".join(str(e) for e in branch.effects)
                parts.append(f"{branch.prob}: [{effs}]")
            out_str += f"probabilistic after {self.time:.3f}: {{ {', '.join(parts)} }}  "

        rfs = ", ".join(str(f) for f in self.resulting_fluents)
        return out_str + f"after {self.time:.3f}: {rfs}"

    def __lt__(self, other: "GroundedEffect") -> bool:
        return self.time < other.time


class Effect:
    def __init__(
        self,
        time: OptExpr,
        prob_effects: List[Tuple[OptExpr, List["Effect"]]] = list(),
        resulting_fluents: Set[Fluent] = set(),
    ):
        self.time = _make_bindable(time)
        self.prob_effects = [
            (_make_bindable(prob), effects) for prob, effects in prob_effects
        ]
        self.resulting_fluents = resulting_fluents
        self.is_probabilistic = bool(self.prob_effects)

    def _ground(self, binding: Binding) -> "GroundedEffect":
        # def evaluate(expr): return expr(binding) if callable(expr) else expr
        if self.is_probabilistic:
            grounded_prob_effects = tuple(
                (
                    prob(binding), tuple(e._ground(binding) for e in effect_list)
                )
                for prob, effect_list in self.prob_effects
            )
        else:
            grounded_prob_effects = tuple()

        grounded_time: float = self.time(binding)
        grounded_resulting_fluents = set(
            Fluent(
                f.name, *[binding.get(arg, arg) for arg in f.args], negated=f.negated
            )
            for f in self.resulting_fluents
        )

        return GroundedEffect(
            grounded_time,
            prob_effects=grounded_prob_effects,
            resulting_fluents=grounded_resulting_fluents,
        )


class Action:
    def __init__(
        self,
        preconditions: frozenset[Fluent],
        effects: List[GroundedEffect],
        name: Optional[str] = None,
    ):
        self.preconditions = frozenset(preconditions)
        self.effects = effects
        self.name = name or "anonymous"
        self._pos_precond = frozenset(f for f in self.preconditions if not f.negated)
        self._neg_precond_flipped = frozenset(
            ~f for f in self.preconditions if f.negated
        )

    def __str__(self):
        pre_str = ", ".join(str(p) for p in self.preconditions)
        eff_strs = []
        for eff in self.effects:
            eff_strs.append(f"    {str(eff)}")
        return (
            f"Action('{self.name}'\n  Preconditions: [{pre_str}]\n  Effects:\n    "
            + "\n    ".join(eff_strs)
            + ")"
        )

    def __repr__(self):
        return self.__str__()


class State:
    def __init__(
        self,
        time: float = 0,
        fluents: Optional[Union[frozenset[Fluent], Set[Fluent]]] = None,
        upcoming_effects: Optional[List[Tuple[float, GroundedEffect]]] = None,
    ):
        self.time = time
        if not fluents:
            self.fluents = frozenset()
        elif isinstance(fluents, frozenset):
            self.fluents = fluents
        else:
            self.fluents = frozenset(fluents)
        self.upcoming_effects = upcoming_effects or []
        self._hash = None

    def satisfies_precondition(self, action: Action, relax: bool = False) -> bool:
        if relax:
            return action._pos_precond <= self.fluents
        return action._pos_precond <= self.fluents and self.fluents.isdisjoint(
            action._neg_precond_flipped
        )

    def copy(self) -> "State":
        return State(
            time=self.time,
            fluents=self.fluents,
            upcoming_effects=list(self.upcoming_effects),
        )

    def set_time(self, new_time: float):
        self.time = new_time
        self._hash = None

    def update_fluents(
        self, new_fluents: Union[set[Fluent], frozenset[Fluent]], relax: bool = False
    ):
        self._hash = None

        positives = {f for f in new_fluents if not f.negated}
        if relax:
            self.fluents = frozenset(self.fluents | positives)
        else:
            flipped_negatives = {~f for f in new_fluents if f.negated}
            fluent_set = self.fluents - flipped_negatives | positives
            self.fluents = frozenset(fluent_set)

    def queue_effect(self, effect: GroundedEffect):
        self._hash = None
        heapq.heappush(self.upcoming_effects, (self.time + effect.time, effect))

    def pop_effect(self):
        self._hash = None
        heapq.heappop(self.upcoming_effects)

    def copy_and_zero_out_time(self):
        dt = self.time
        new_effects = [(t - dt, e) for t, e in self.upcoming_effects]
        return State(
            time=0,
            fluents=self.fluents,
            upcoming_effects=new_effects,
        )

    def __hash__(self) -> int:
        if not self._hash:
            self._hash = hash((self.time, self.fluents, tuple(self.upcoming_effects)))
        return self._hash
        # return hash((self.time, self.fluents, tuple(self.upcoming_effects)))

    def __eq__(self, other: object) -> bool:
        return hash(self) == hash(other)

    def __str__(self):
        upcoming = ", ".join(f"({t:.3f}, {effect})" for t, effect in self.upcoming_effects)
        return f"State<time={self.time:.3f}, fluents={self.fluents}, upcoming_effects={upcoming}>"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return self.time < other.time


class Operator:
    def __init__(
        self,
        name: str,
        parameters: List[Tuple[str, str]],
        preconditions: List[Fluent],
        effects: Sequence[Effect],
    ):
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions
        self.effects = effects

    def instantiate(self, objects_by_type: Dict[str, List[str]]) -> List[Action]:
        grounded_actions = []
        domains = [objects_by_type[typ] for _, typ in self.parameters]
        for assignment in itertools.product(*domains):
            binding = {var: obj for (var, _), obj in zip(self.parameters, assignment)}
            if len(set(binding.values())) != len(binding):
                continue
            grounded_actions.append(self._ground(binding))
        return grounded_actions

    def _ground(self, binding: Dict[str, str]) -> Action:
        def evaluate(value):
            return value(binding) if callable(value) else value

        grounded_preconditions = frozenset(
            self._substitute_fluent(f, binding) for f in self.preconditions
        )
        grounded_effects = [eff._ground(binding) for eff in self.effects]

        name_str = f"{self.name} " + " ".join(
            binding[var] for var, _ in self.parameters
        )
        return Action(grounded_preconditions, grounded_effects, name=name_str)

    def _substitute_fluent(self, fluent: Fluent, binding: Dict[str, str]) -> Fluent:
        grounded_args = tuple(binding.get(arg, arg) for arg in fluent.args)
        return Fluent(fluent.name, *grounded_args, negated=fluent.negated)


def get_action_by_name(actions: List[Action], name: str) -> Action:
    for action in actions:
        if action.name == name:
            return action
    raise ValueError(f"No action found with name: {name}")


def get_next_actions(state: State, all_actions: List[Action]) -> List[Action]:
    # Step 1: Extract all `free(...)` fluents
    free_robot_fluents = sorted([f for f in state.fluents if f.name == "free"], key=str)
    # neg_fluents = {~f for f in free_robot_fluents}
    neg_state = state.copy()
    neg_state.update_fluents({~f for f in free_robot_fluents})

    # Step 2: Check each robot individually
    for free_pred in free_robot_fluents:
        # Create a restricted version of the state
        temp_state = State(
            time=state.time,
            fluents=neg_state.fluents | {free_pred},
        )

        # Step 3: Check for applicable actions
        applicable = [a for a in all_actions if temp_state.satisfies_precondition(a)]
        if applicable:
            return applicable

    # Step 4: Otherwise, return any possible actions
    return [a for a in all_actions if state.satisfies_precondition(a)]


from mrppddl._bindings import GroundedEffect, Fluent, Action, State
from mrppddl._bindings import transition

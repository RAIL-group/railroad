# Re-import required dependencies due to kernel reset
from typing import Callable, List, Tuple, Dict, Set, Union, Sequence
import itertools

from mrppddl._bindings import GroundedEffect, Fluent, Action, State
from mrppddl._bindings import transition  # noqa: F401

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


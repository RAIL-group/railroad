# Re-import required dependencies due to kernel reset
from typing import Callable, List, Tuple, Dict, Set, Union, Optional
from queue import PriorityQueue
import itertools


TimeExpr = Union[float, Callable[[Dict[str, str]], float]]
ProbExpr = Union[float, Callable[[Dict[str, str]], float]]
Binding = Dict[str, str]


class Fluent(object):
    def __init__(self, name: str, *args: str, negated: bool = False):
        if args:
            self.name = name
            self.args = args
            if "not" in name[:4]:
                raise ValueError("Use the 'negated' argument or ~Fluent to negate.")
        else:
            if negated:
                raise ValueError("Cannot both pass a full string and negated=True. Use 'not' or ~Fluent.")
            split = name.split(" ")
            if split[0] == 'not':
                negated = True
                split = split[1:]
            self.name = split[0]
            self.args = tuple(split[1:])
        self.negated = negated

    def __str__(self) -> str:
        prefix = "not " if self.negated else ""
        return f"{prefix}{self.name} {' '.join(self.args)}"

    def __repr__(self) -> str:
        return f"Fluent<{self}>"

    def __hash__(self) -> int:
        return hash((self.name, self.args, self.negated))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Fluent) and
            self.name == other.name and
            self.args == other.args and
            self.negated == other.negated
        )

    def __invert__(self) -> 'Fluent':
        return Fluent(self.name, *self.args, negated=not self.negated)

    def positive(self) -> 'Fluent':
        return Fluent(self.name, *self.args, negated=False)


class ActiveFluents(object):
    def __init__(self, fluents: Optional[Set[Fluent]] = None):
        if fluents and any(fluent.negated for fluent in fluents):
            raise ValueError("All fluents in active fluents must be positive.")
        self.fluents: Set[Fluent] = set(fluents) if fluents else set()

    def update(self, fluents: Set[Fluent], relaxed: bool = False) -> 'ActiveFluents':
        """Apply fluents to the active set: add positives, remove targets of negations."""
        positives = {f for f in fluents if not f.negated}
        new_active_fluents = self.copy()

        if not relaxed:
            negatives = {f.positive() for f in fluents if f.negated}
            new_active_fluents.fluents -= negatives

        new_active_fluents.fluents |= positives
        return new_active_fluents

    def copy(self) -> 'ActiveFluents':
        return ActiveFluents(set(self.fluents))

    def __contains__(self, f: Fluent) -> bool:
        return f in self.fluents

    def __iter__(self):
        return iter(self.fluents)

    def __str__(self):
        return f"{{{', '.join(str(f) for f in sorted(self.fluents, key=str))}}}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ActiveFluents) and self.fluents == other.fluents

    def __hash__(self):
        return hash(frozenset(self.fluents))


class GroundedEffectType:
    def __init__(self, time: float, resulting_fluents: Set[Fluent]):
        self.time = time
    
    def __lt__(self, other: 'GroundedEffectType') -> bool:
        return self.time < other.time


class LiftedEffectType:
    def __init__(self, time: TimeExpr, resulting_fluents: Set[Fluent]):
        self.time = time
        self.resulting_fluents = resulting_fluents

    def _ground(self, binding: Binding) -> 'GroundedEffectType':
        grounded_time = self.time(binding) if callable(self.time) else self.time
        grounded_fluents = {
            Fluent(f.name, *[binding.get(arg, arg) for arg in f.args], negated=f.negated)
            for f in self.resulting_fluents
        }
        print(GroundedEffect(grounded_time, grounded_fluents))
        return GroundedEffect(grounded_time, grounded_fluents)


class GroundedEffect(GroundedEffectType):
    def __init__(self, time: float, resulting_fluents: Set[Fluent]):
        self.time = time
        self.resulting_fluents = resulting_fluents

    def __str__(self):
        rfs = ", ".join(str(f) for f in self.resulting_fluents)
        return f"after {self.time}: {rfs}"

    def __repr__(self):
        return f"GroundedEffect({self})"


class Effect(LiftedEffectType):
    def __init__(self, time: TimeExpr, resulting_fluents: Set[Fluent]):
        self.time = time
        self.resulting_fluents = resulting_fluents

class GroundedProbEffect(GroundedEffectType):
    def __init__(
        self,
        time: float,
        prob_effects: List[Tuple[float, List[GroundedEffectType]]],
        resulting_fluents: Set[Fluent] = set()
    ):
        self.time = time
        self.prob_effects = prob_effects
        self.resulting_fluents = resulting_fluents

    def __str__(self):
        parts = []
        for prob, effs in self.prob_effects:
            branch = "; ".join(str(e) for e in effs)
            parts.append(f"{prob}: [{branch}]")
        return f"probabilistic after {self.time}: {{ {', '.join(parts)} }}"

    def __repr__(self):
        return f"ProbEffects({self})"


class ProbEffect(LiftedEffectType):
    def __init__(
        self,
        time: TimeExpr,
        prob_effects: List[Tuple[ProbExpr, List[Effect]]],
        resulting_fluents: Set[Fluent] = set()
    ):
        self.time = time
        self.prob_effects = prob_effects
        self.resulting_fluents = resulting_fluents

    def _ground(self, binding: Binding) -> 'GroundedProbEffect':
        def evaluate(expr): return expr(binding) if callable(expr) else expr

        grounded_prob_effects = [
            (evaluate(prob), [e._ground(binding) for e in effect_list])
            for prob, effect_list in self.prob_effects
        ]

        grounded_time = evaluate(self.time)
        grounded_resulting_fluents = {
            Fluent(f.name, *[binding.get(arg, arg) for arg in f.args], negated=f.negated)
            for f in self.resulting_fluents
        }

        return GroundedProbEffect(grounded_time, grounded_prob_effects, grounded_resulting_fluents)


class Action:
    def __init__(self, preconditions: List[Fluent], 
                 effects: List[GroundedEffectType], name: Optional[str] = None):
        self.preconditions = preconditions
        self.effects = effects
        self.name = name or "anonymous"

    def __str__(self):
        pre_str = ", ".join(str(p) for p in self.preconditions)
        eff_strs = []
        for eff in self.effects:
            eff_strs.append(f"    {str(eff)}")
        return f"Action('{self.name}'\n  Preconditions: [{pre_str}]\n  Effects:\n    " + "\n    ".join(eff_strs) + ")"

    def __repr__(self):
        return self.__str__()


class State:
    def __init__(self, time: float = 0, active_fluents: Optional[Set[Fluent]] = None, upcoming_effects: Optional[PriorityQueue] = None):
        self.time = time
        self.active_fluents = ActiveFluents(active_fluents)
        self.upcoming_effects = upcoming_effects or PriorityQueue()

    def satisfies_precondition(self, action: Action) -> bool:
        for p in action.preconditions:
            if p.negated:
                if p.positive() in self.active_fluents:
                    return False
            else:
                if p not in self.active_fluents:
                    return False
        return True

    def copy(self) -> 'State':
        new_queue = PriorityQueue()
        new_queue.queue = [x for x in self.upcoming_effects.queue]
        return State(
            time=self.time,
            active_fluents=self.active_fluents.copy().fluents,
            upcoming_effects=new_queue
        )

    def zero_out_time(self):
        # FIXME: this isn't really done...
        copy = self.copy()
        copy.time = 0
        print(copy)
        return copy

    def __hash__(self) -> int:
        return hash(self.time) + hash(self.active_fluents)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and hash(self) == hash(other)

    def __str__(self):
        return f"State<time={self.time}, active_fluents={self.active_fluents}>"

    def __repr__(self):
        return self.__str__()


class Operator:
    def __init__(self, name: str, parameters: List[Tuple[str, str]], preconditions: List[Fluent], effects: List[LiftedEffectType]):
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

        grounded_preconditions = [self._substitute_fluent(f, binding) for f in self.preconditions]
        grounded_effects = [eff._ground(binding) for eff in self.effects]

        name_str = f"{self.name} " + " ".join(binding[var] for var, _ in self.parameters)
        return Action(grounded_preconditions, grounded_effects, name=name_str)

    def _substitute_fluent(self, fluent: Fluent, binding: Dict[str, str]) -> Fluent:
        grounded_args = tuple(binding.get(arg, arg) for arg in fluent.args)
        return Fluent(fluent.name, *grounded_args, negated=fluent.negated)


def transition(state: State, action: Action) -> List[Tuple[State, float]]:
    if not state.satisfies_precondition(action):
        raise ValueError("Precondition not satisfied for applying action")

    new_state = state.copy()
    for effect in action.effects:
        new_state.upcoming_effects.put((new_state.time + effect.time, effect))

    # Fixme: is this necessary or can I just pass it to outcomes?
    outcomes: Dict[State, float] = {}
    _advance_to_terminal(new_state, prob=1.0, outcomes=outcomes)
    return list(outcomes.items())


def _advance_to_terminal(state: State, prob: float, outcomes: Dict[State, float]) -> None:
    while not state.upcoming_effects.empty():
        scheduled_time, effect = state.upcoming_effects.queue[0]

        # Check if we're ready to yield this state
        if scheduled_time > state.time and any(f.name == "free" for f in state.active_fluents):
            outcomes[state] = prob
            return

        # Advance time if necessary
        if scheduled_time > state.time:
            state.time = scheduled_time

        # Apply effect
        state.upcoming_effects.get()
        state.active_fluents = state.active_fluents.update(effect.resulting_fluents)

        if isinstance(effect, GroundedProbEffect):
            for branch_prob, effects in effect.prob_effects:
                branched = state.copy()
                for e in effects:
                    branched.upcoming_effects.put((branched.time + e.time, e))
                _advance_to_terminal(branched, prob * branch_prob, outcomes)
            return  # stop after branching

    # No more effects; yield terminal state
    outcomes[state] = prob


def get_action_by_name(actions: List[Action], name: str) -> Action:
    for action in actions:
        if action.name == name:
            return action
    raise ValueError(f"No action found with name: {name}")


def get_next_actions(state: State, all_actions: List[Action]) -> List[Action]:
    # Step 1: Extract all `free(...)` fluents
    free_robot_fluents = sorted(
        [f for f in state.active_fluents if f.name == "free"],
        key=str
    )
    neg_active_fluents = state.active_fluents.update({~f for f in free_robot_fluents})

    # Step 2: Check each robot individually
    for free_pred in free_robot_fluents:
        # Create a restricted version of the state
        temp_state = State(time=state.time, active_fluents=neg_active_fluents.update({free_pred}).fluents)

        # Step 3: Check for applicable actions
        applicable = [a for a in all_actions if temp_state.satisfies_precondition(a)]
        if applicable:
            return applicable

    # Step 4: No applicable actions found
    return []

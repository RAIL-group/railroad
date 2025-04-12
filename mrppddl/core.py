# Re-import required dependencies due to kernel reset
from typing import Callable, List, Tuple, Dict, Set, Union, Optional, Sequence
from queue import PriorityQueue
import itertools


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

class Fluent(object):
    __slots__ = ('name', 'args', 'negated', '_hash', '_positive')

    def __init__(self, name: str, *args: str, negated: bool = False):
        if args:
            self.name = name
            self.args = args
            if "not" in name[:4]:
                raise ValueError("Use the 'negated' argument or ~Fluent to negate.")
            self.negated = negated
        else:
            if negated:
                raise ValueError("Cannot both pass a full string and negated=True. Use 'not' or ~Fluent.")
            split = name.split(" ")
            if split[0] == 'not':
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
        return f"Fluent<{self}>"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return hash(self) == hash(other)

    def __invert__(self) -> 'Fluent':
        return Fluent(self.name, *self.args, negated=not self.negated)


class ActiveFluents(object):
    def __init__(self, fluents: Optional[Union[frozenset[Fluent], Set[Fluent]]] = None):
        self.fluents: frozenset[Fluent] = frozenset(fluents) if fluents else frozenset()
        self._hash = hash(self.fluents)

    def update(self, fluents: Union[frozenset[Fluent], Set[Fluent]], relaxed: bool = False) -> 'ActiveFluents':
        """Apply fluents to the active set: add positives, remove targets of negations."""
        positives = {f for f in fluents if not f.negated}
        fluent_set = set(self.fluents)

        if not relaxed:
            flipped_negatives = {~f for f in fluents if f.negated}
            fluent_set -= flipped_negatives

        fluent_set |= positives
        return ActiveFluents(fluent_set)

    def __contains__(self, f: Fluent) -> bool:
        return f in self.fluents

    def __iter__(self):
        return iter(self.fluents)

    def __str__(self):
        return f"{{{', '.join(str(f) for f in sorted(self.fluents, key=str))}}}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other: object) -> bool:
        return hash(self) == hash(other)

    def __hash__(self):
        return self._hash


class GroundedEffectType:
    def __init__(self, time: float, resulting_fluents: Set[Fluent]):
        self.time = time
        self.resulting_fluents = resulting_fluents
    
    def __lt__(self, other: 'GroundedEffectType') -> bool:
        return self.time < other.time


class LiftedEffectType:
    def __init__(self, time: OptExpr, resulting_fluents: Set[Fluent]):
        self.time = _make_bindable(time)
        self.resulting_fluents = resulting_fluents

    def _ground(self, binding: Binding) -> 'GroundedEffectType':
        grounded_time = self.time(binding)
        grounded_fluents = frozenset(
            Fluent(f.name, *[binding.get(arg, arg) for arg in f.args], negated=f.negated)
            for f in self.resulting_fluents
        )
        return GroundedEffect(grounded_time, grounded_fluents)


class GroundedEffect(GroundedEffectType):
    def __init__(self, time: Num, resulting_fluents: frozenset[Fluent]):
        self.time = time
        self.resulting_fluents = resulting_fluents
        self._hash: int = hash((self.time, self.resulting_fluents))

    def __str__(self):
        rfs = ", ".join(str(f) for f in self.resulting_fluents)
        return f"after {self.time}: {rfs}"

    def __repr__(self):
        return f"GroundedEffect({self})"

    def __hash__(self):
        return self._hash


class Effect(LiftedEffectType):
    def __init__(self, time: OptExpr, resulting_fluents: Set[Fluent]):
        self.time = _make_bindable(time)
        self.resulting_fluents = resulting_fluents

class GroundedProbEffect(GroundedEffectType):
    __slots__ = ('time', 'prob_effects', 'resulting_fluents', '_hash')

    def __init__(
        self,
        time: float,
        prob_effects: tuple[tuple[float, tuple[GroundedEffectType, ...]], ...],
        resulting_fluents: frozenset[Fluent] = frozenset()
    ):
        self.time = time
        self.prob_effects = prob_effects
        self.resulting_fluents = resulting_fluents
        self._hash: int = hash((self.time, self.resulting_fluents, self.prob_effects))

    def __str__(self):
        parts = []
        for prob, effs in self.prob_effects:
            branch = "; ".join(str(e) for e in effs)
            parts.append(f"{prob}: [{branch}]")
        return f"probabilistic after {self.time}: {{ {', '.join(parts)} }}"

    def __repr__(self):
        return f"ProbEffects({self})"

    def __hash__(self):
        return self._hash

class ProbEffect(LiftedEffectType):
    def __init__(
        self,
        time: OptExpr,
        prob_effects: List[Tuple[OptExpr, List[Effect]]],

        resulting_fluents: Set[Fluent] = set()
    ):
        self.time = _make_bindable(time)
        self.prob_effects = [(_make_bindable(prob), effects) 
                             for prob, effects in prob_effects]
        self.resulting_fluents = resulting_fluents

    def _ground(self, binding: Binding) -> 'GroundedProbEffect':
        # def evaluate(expr): return expr(binding) if callable(expr) else expr
        grounded_prob_effects = tuple(
            (prob(binding), tuple(e._ground(binding) for e in effect_list))
            for prob, effect_list in self.prob_effects
        )
        # grounded_prob_effects = [
        #     (prob(binding), [e._ground(binding) for e in effect_list])
        #     for prob, effect_list in self.prob_effects
        # ]

        grounded_time: float = self.time(binding)
        grounded_resulting_fluents = frozenset(
            Fluent(f.name, *[binding.get(arg, arg) for arg in f.args], negated=f.negated)
            for f in self.resulting_fluents
        )

        return GroundedProbEffect(grounded_time, grounded_prob_effects, grounded_resulting_fluents)


class Action:
    def __init__(self, preconditions: frozenset[Fluent], 
                 effects: List[GroundedEffectType], name: Optional[str] = None):
        self.preconditions = frozenset(preconditions)
        self.effects = effects
        self.name = name or "anonymous"
        self._pos_precond = frozenset(f for f in self.preconditions if not f.negated)
        self._neg_precond_flipped = frozenset(~f for f in self.preconditions if f.negated)

    def __str__(self):
        pre_str = ", ".join(str(p) for p in self.preconditions)
        eff_strs = []
        for eff in self.effects:
            eff_strs.append(f"    {str(eff)}")
        return f"Action('{self.name}'\n  Preconditions: [{pre_str}]\n  Effects:\n    " + "\n    ".join(eff_strs) + ")"

    def __repr__(self):
        return self.__str__()


# class State:
#     def __init__(self, time: float = 0, active_fluents: Optional[Union[frozenset[Fluent], Set[Fluent]]] = None, upcoming_effects: Optional[PriorityQueue] = None):
#         self.time = time
#         self.active_fluents = ActiveFluents(active_fluents)
#         self.upcoming_effects = upcoming_effects or PriorityQueue()

#     def satisfies_precondition(self, action: Action) -> bool:
#         return (action._pos_precond <= self.active_fluents.fluents
#                 and self.active_fluents.fluents.isdisjoint(action._neg_precond_flipped))

#     def copy(self) -> 'State':  #noqa
#         new_queue = PriorityQueue()
#         new_queue.queue = [x for x in self.upcoming_effects.queue]
#         return State(
#             time=self.time,
#             active_fluents=self.active_fluents.fluents,
#             upcoming_effects=new_queue
#         )

#     def copy_and_zero_out_time(self):
#         dt = self.time
#         new_queue = PriorityQueue()
#         for time, effect in self.upcoming_effects.queue:
#             new_queue.put((time - dt, effect))
#         return State(
#             time=0,
#             active_fluents=set(self.active_fluents.fluents),
#             upcoming_effects=new_queue
#         )

#     def __hash__(self) -> int:
#         upcoming = tuple((t, effect) for t, effect in self.upcoming_effects.queue)
#         return hash((self.time, self.active_fluents, upcoming))

#     def __eq__(self, other: object) -> bool:
#         return hash(self) == hash(other)

#     def __str__(self):
#         upcoming = tuple((t, effect) for t, effect in self.upcoming_effects.queue)
#         return f"State<time={self.time}, active_fluents={self.active_fluents}, upcoming_effects={upcoming}>"

#     def __repr__(self):
#         return self.__str__()

import heapq

class State:
    def __init__(
        self,
        time: float = 0,
        active_fluents: Optional[Union[frozenset[Fluent], Set[Fluent]]] = None,
        upcoming_effects: Optional[List[Tuple[float, Union[GroundedEffectType]]]] = None
    ):
        self.time = time
        self.active_fluents = ActiveFluents(active_fluents)
        self.upcoming_effects = upcoming_effects or []

    def satisfies_precondition(self, action: Action) -> bool:
        return (action._pos_precond <= self.active_fluents.fluents
                and self.active_fluents.fluents.isdisjoint(action._neg_precond_flipped))

    def copy(self) -> 'State':
        return State(
            time=self.time,
            active_fluents=self.active_fluents.fluents,
            upcoming_effects=list(self.upcoming_effects)
        )

    def copy_and_zero_out_time(self):
        dt = self.time
        new_effects = [(t - dt, e) for t, e in self.upcoming_effects]
        return State(
            time=0,
            active_fluents=self.active_fluents.fluents,
            upcoming_effects=new_effects
        )

    def __hash__(self) -> int:
        upcoming = tuple((t, effect) for t, effect in self.upcoming_effects)
        return hash((self.time, self.active_fluents, upcoming))

    def __eq__(self, other: object) -> bool:
        return hash(self) == hash(other)

    def __str__(self):
        upcoming = tuple((t, effect) for t, effect in self.upcoming_effects)
        return f"State<time={self.time}, active_fluents={self.active_fluents}, upcoming_effects={upcoming}>"

    def __repr__(self):
        return self.__str__()


class Operator:
    def __init__(self, name: str, parameters: List[Tuple[str, str]], preconditions: List[Fluent], effects: Sequence[LiftedEffectType]):
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

        grounded_preconditions = frozenset(self._substitute_fluent(f, binding) for f in self.preconditions)
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
        heapq.heappush(new_state.upcoming_effects, (new_state.time + effect.time, effect))

    # Fixme: is this necessary or can I just pass it to outcomes?
    outcomes: Dict[State, float] = {}
    _advance_to_terminal(new_state, prob=1.0, outcomes=outcomes)
    return list(outcomes.items())


def _advance_to_terminal(state: State, prob: float, outcomes: Dict[State, float]) -> None:
    while state.upcoming_effects:
        scheduled_time, effect = state.upcoming_effects[0]

        # Check if we're ready to yield this state
        if scheduled_time > state.time and any(f.name == "free" for f in state.active_fluents):
            outcomes[state] = prob
            return

        # Advance time if necessary
        if scheduled_time > state.time:
            state.time = scheduled_time

        # Apply effect
        heapq.heappop(state.upcoming_effects)
        state.active_fluents = state.active_fluents.update(effect.resulting_fluents)

        if isinstance(effect, GroundedProbEffect):
            for branch_prob, effects in effect.prob_effects:
                branched = state.copy()
                for e in effects:
                    heapq.heappush(branched.upcoming_effects, (branched.time + e.time, e))
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

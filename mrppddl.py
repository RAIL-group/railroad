from typing import List, Tuple, Dict, Set, Union, Optional
from queue import PriorityQueue
import itertools


class Fluent:
    """
    Represents a logical fluent of the form (name arg1 arg2 ...).
    Can be negated using the `~` operator.
    """

    def __init__(self, name: str, *args: str):
        self.name: str = name
        self.args: Tuple[str, ...] = args

    def __str__(self) -> str:
        args_str = " ".join(self.args)
        return f"{self.name} {args_str}"

    def __repr__(self) -> str:
        return f"Fluent<{self}>"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Fluent) and hash(self) == hash(other)

    def __invert__(self) -> "Fluent":
        if self.name.startswith("not "):
            return Fluent(self.name[4:], *self.args)
        return Fluent(f"not {self.name}", *self.args)


class Effect:
    """
    Represents a deterministic effect occurring after a delay.
    """

    def __init__(self, time: float, resulting_fluents: Set[Fluent]):
        self.time: float = time
        self.resulting_fluents: Set[Fluent] = resulting_fluents


class ProbEffects:
    """
    Represents probabilistic effects, with each outcome associated with a probability
    and a list of Effects. The overall effect is scheduled after a delay.
    """

    def __init__(
        self,
        time: float,
        prob_effects: List[Tuple[float, List[Effect]]],
        resulting_fluents: Set[Fluent] = set(),
    ):
        self.time: float = time
        self.prob_effects: List[Tuple[float, List[Effect]]] = prob_effects
        self.resulting_fluents: Set[Fluent] = resulting_fluents


class Action:
    """
    A grounded action with concrete preconditions and effects.
    """

    def __init__(
        self, preconditions: List[Fluent], effects: List[Union[Effect, ProbEffects]]
    ):
        self.preconditions: List[Fluent] = preconditions
        self.effects: List[Union[Effect, ProbEffects]] = effects

    def __str__(self) -> str:
        pre_str = ", ".join(str(p) for p in self.preconditions)
        eff_strs = []
        for eff in self.effects:
            if isinstance(eff, Effect):
                rfs = ", ".join(str(f) for f in eff.resulting_fluents)
                eff_strs.append(f"after {eff.time}: {rfs}")
            elif isinstance(eff, ProbEffects):
                prob_lines = []
                for p, elist in eff.prob_effects:
                    outcomes = []
                    for e in elist:
                        rf = ", ".join(str(f) for f in e.resulting_fluents)
                        outcomes.append(f"after {e.time}: {rf}")
                    prob_lines.append(f"{p}: [{'; '.join(outcomes)}]")
                eff_strs.append(
                    f"probabilistic after {eff.time}: {{ {', '.join(prob_lines)} }}"
                )
        return (
            f"Action(\n  Preconditions: [{pre_str}]\n  Effects:\n    "
            + "\n    ".join(eff_strs)
            + "\n)"
        )

    def __repr__(self) -> str:
        return self.__str__()


class Operator:
    """
    An operator schema with typed parameters, abstract preconditions, and effects.
    Can be instantiated into grounded Actions using a dictionary of available objects by type.
    """

    def __init__(
        self,
        name: str,
        parameters: List[Tuple[str, str]],
        preconditions: List[Fluent],
        effects: List[Union[Effect, ProbEffects]],
    ):
        self.name: str = name
        self.parameters: List[Tuple[str, str]] = parameters
        self.preconditions: List[Fluent] = preconditions
        self.effects: List[Union[Effect, ProbEffects]] = effects

    def instantiate(self, objects_by_type: Dict[str, List[str]]) -> List[Action]:
        """
        Instantiate this operator into all valid grounded Actions
        given a dictionary mapping types to object names.
        Ensures all arguments are distinct.
        """
        grounded_actions: List[Action] = []
        domains = [objects_by_type[typ] for _, typ in self.parameters]
        for assignment in itertools.product(*domains):
            binding = {var: obj for (var, _), obj in zip(self.parameters, assignment)}
            if len(set(binding.values())) != len(binding):  # enforce distinctness
                continue
            grounded_actions.append(self._ground(binding))
        return grounded_actions

    def _ground(self, binding: Dict[str, str]) -> Action:
        grounded_preconditions = [
            self._substitute_fluent(f, binding) for f in self.preconditions
        ]
        grounded_effects = []
        for eff in self.effects:
            if isinstance(eff, Effect):
                grounded_fluents = {
                    self._substitute_fluent(f, binding) for f in eff.resulting_fluents
                }
                grounded_effects.append(Effect(eff.time, grounded_fluents))
            elif isinstance(eff, ProbEffects):
                grounded_prob_effects = []
                for prob, effect_list in eff.prob_effects:
                    grounded_list = [
                        Effect(
                            e.time,
                            {
                                self._substitute_fluent(f, binding)
                                for f in e.resulting_fluents
                            },
                        )
                        for e in effect_list
                    ]
                    grounded_prob_effects.append((prob, grounded_list))
                grounded_resulting_fluents = {
                    self._substitute_fluent(f, binding) for f in eff.resulting_fluents
                }
                grounded_effects.append(
                    ProbEffects(
                        eff.time, grounded_prob_effects, grounded_resulting_fluents
                    )
                )
        return Action(preconditions=grounded_preconditions, effects=grounded_effects)

    def _substitute_fluent(self, fluent: Fluent, binding: Dict[str, str]) -> Fluent:
        grounded_args = tuple(binding.get(arg, arg) for arg in fluent.args)
        return Fluent(fluent.name, *grounded_args)


class State:
    """
    Represents a timed state with active fluents and scheduled effects.
    """

    def __init__(
        self,
        time: float = 0,
        active_fluents: Optional[Set[Fluent]] = None,
        upcoming_effects: Optional[PriorityQueue] = None,
    ):
        self.time: float = time
        self.active_fluents: Set[Fluent] = active_fluents or set()
        self.upcoming_effects: PriorityQueue = upcoming_effects or PriorityQueue()

    def satisfies_precondition(self, action: Action) -> bool:
        return all(p in self.active_fluents for p in action.preconditions)

    def transition(self, action: Action) -> Dict["State", Tuple[float, float]]:
        if not self.satisfies_precondition(action):
            raise ValueError("Precondition not satisfied for applying action")
        next_state = self.copy()
        outcome_states: Dict[State, Tuple[float, float]] = {}
        for effect in action.effects:
            next_state.upcoming_effects.put((effect.time + next_state.time, effect))
        advance_state(next_state, outcome_states)
        return outcome_states

    def copy(self) -> "State":
        new_queue = PriorityQueue()
        new_queue.queue = [x for x in self.upcoming_effects.queue]
        return State(
            time=self.time,
            active_fluents=set(self.active_fluents),
            upcoming_effects=new_queue,
        )

    def __repr__(self) -> str:
        return f"State<time={self.time}, active_fluents={self.active_fluents}>"

    def __hash__(self) -> int:
        return hash(self.time) + sum(hash(f) for f in self.active_fluents)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and hash(self) == hash(other)


def advance_state(
    state: State, outcome_states: Dict[State, Tuple[float, float]], prob: float = 1.0
) -> None:
    while not state.upcoming_effects.empty():
        _, effect = state.upcoming_effects.get()
        state.time += effect.time
        state.active_fluents = add_fluents(
            state.active_fluents, effect.resulting_fluents
        )

        # Terminal condition example: presence of 'free' fluent
        for f in state.active_fluents:
            if f.name == "free":
                outcome_states[state] = (prob, state.time)

        if isinstance(effect, Effect):
            continue

        for p, effects in effect.prob_effects:
            prob_state = state.copy()
            for e in effects:
                prob_state.upcoming_effects.put((e.time + prob_state.time, e))
            advance_state(prob_state, outcome_states, prob=prob * p)


def add_fluents(base: Set[Fluent], new_fluents: Set[Fluent]) -> Set[Fluent]:
    # You could expand this to handle logical negation, etc.
    return base.union(new_fluents)


#### EXAMPLE

search_op = Operator(
    name="search",
    parameters=[
        ("?robot", "robot"),
        ("?loc_from", "location"),
        ("?loc_to", "location"),
        ("?object", "object"),
    ],
    preconditions=[
        Fluent("at", "?robot", "?loc_from"),
        ~Fluent("searched", "?loc_to", "?object"),
        Fluent("free", "?robot"),
        ~Fluent("found", "?object"),
    ],
    effects=[
        Effect(
            time=0,
            resulting_fluents={~Fluent("free", "?robot"), ~Fluent("found", "?object")},
        ),
        ProbEffects(
            time=5,
            prob_effects=[
                (
                    0.8,
                    [
                        Effect(
                            time=0,
                            resulting_fluents={
                                Fluent("at", "?loc_to", "?object"),
                                Fluent("found", "?object"),
                            },
                        ),
                        Effect(
                            time=3,
                            resulting_fluents={
                                Fluent("holding", "?robot", "?object"),
                                ~Fluent("at", "?loc_to", "?object"),
                                Fluent("free", "?robot"),
                            },
                        ),
                    ],
                ),
                (
                    0.2,
                    [
                        Effect(
                            time=0,
                            resulting_fluents={
                                Fluent("free", "?robot"),
                                ~Fluent("at", "?loc_to", "?object"),
                            },
                        )
                    ],
                ),
            ],
            resulting_fluents={
                Fluent("at", "?robot", "?loc_to"),
                ~Fluent("at", "?robot", "?loc_from"),
                Fluent("searched", "?loc_to", "?object"),
            },
        ),
    ],
)

objects_by_type = {
    "robot": ["r1"],
    "location": ["roomA", "roomB"],
    "object": ["medkit"],
}

grounded_search_actions = list(search_op.instantiate(objects_by_type))

for action in grounded_search_actions:
    print(action)
    print("-" * 40)

# Define fluents
r1 = "r1"
loc = "roomA"
fluent_free = Fluent("free", r1)
fluent_at = Fluent("at", r1, loc)

# Initial state with one fluent
initial_state = State(
    time=0, active_fluents={Fluent("at", "r1", "roomA"), Fluent("free", "roomA")}
)

# Define a grounded action with effects over time
action = Action(
    preconditions=[Fluent("at", "r1", "roomA"), Fluent("free", "roomA")],
    effects=[
        Effect(
            time=2, resulting_fluents={~Fluent("free", "roomA"), Fluent("busy", r1)}
        ),
        ProbEffects(
            time=5,
            prob_effects=[
                (
                    0.8,
                    [
                        Effect(
                            time=4,
                            resulting_fluents={
                                Fluent("charged", r1),
                                Fluent("free", r1),
                            },
                        )
                    ],
                ),
                (0.2, [Effect(time=2, resulting_fluents={Fluent("free", r1)})]),
            ],
        ),
    ],
)

# # Transition to new states and simulate time advancement
# if initial_state.satisfies_precondition(action):
#     print("Precondition satisfied.")
#     outcome_states = initial_state.transition(action)

#     # Print the outcome states
#     for state, (prob, t) in outcome_states.items():
#         print(f"Time: {t}, Probability: {prob}")
#         print(f"Final State: {state}")
#         print("-" * 40)
# else:
#     print("Preconditions not satisfied.")

print("HAAAAA")

from typing import List, Tuple, Dict


def transition(state: State, action: Action) -> List[Tuple[State, float]]:
    """
    Applies an action to a state and returns a list of terminal outcome states,
    each with associated probability.
    """
    if not state.satisfies_precondition(action):
        raise ValueError("Precondition not satisfied for applying action")

    # Make a copy to apply effects
    new_state = state.copy()
    for effect in action.effects:
        new_state.upcoming_effects.put((new_state.time + effect.time, effect))

    # Use helper to compute all terminal states
    outcomes: Dict[State, float] = {}
    _advance_to_terminal(new_state, prob=1.0, outcomes=outcomes)

    return list(outcomes.items())


def _advance_to_terminal(
    state: State, prob: float, outcomes: Dict[State, float]
) -> None:
    """
    Recursively apply all effects (including probabilistic ones) until terminal.
    Terminal = no more effects to apply.
    """
    while not state.upcoming_effects.empty():
        # Get next effect in order of execution time
        scheduled_time, effect = state.upcoming_effects.get()
        # Advance the state's clock
        state.time = scheduled_time
        # Apply deterministic fluents
        state.active_fluents = add_fluents(
            state.active_fluents, effect.resulting_fluents
        )

        if isinstance(effect, ProbEffects):
            for branch_prob, effects in effect.prob_effects:
                branched = state.copy()
                for e in effects:
                    branched.upcoming_effects.put((branched.time + e.time, e))
                _advance_to_terminal(branched, prob * branch_prob, outcomes)
            return  # Stop after branching
    # No effects left: terminal state
    outcomes[state] = prob


for outcome_state, prob in transition(initial_state, action):
    print(f"Probability: {prob:.2f}, Final Time: {outcome_state.time}")
    print(outcome_state)

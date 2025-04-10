# Re-import required dependencies due to kernel reset
from typing import List, Tuple, Dict, Set, Union, Optional
from queue import PriorityQueue
import itertools

# Redefine Fluent class
class Fluent:
    def __init__(self, name: str, *args: str):
        self.name = name
        self.args = args

    def __str__(self) -> str:
        return f"{self.name} {' '.join(self.args)}"

    def __repr__(self) -> str:
        return f"Fluent<{self}>"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Fluent) and hash(self) == hash(other)

    def __invert__(self) -> 'Fluent':
        if self.name.startswith('not '):
            return Fluent(self.name[4:], *self.args)
        return Fluent(f"not {self.name}", *self.args)

# Effect classes
class Effect:
    def __init__(self, time: float, resulting_fluents: Set[Fluent]):
        self.time = time
        self.resulting_fluents = resulting_fluents

class ProbEffects:
    def __init__(self, time: float, prob_effects: List[Tuple[float, List[Effect]]], resulting_fluents: Set[Fluent] = set()):
        self.time = time
        self.prob_effects = prob_effects
        self.resulting_fluents = resulting_fluents

# Action class
class Action:
    def __init__(self, preconditions: List[Fluent], effects: List[Union[Effect, ProbEffects]]):
        self.preconditions = preconditions
        self.effects = effects

    def __str__(self):
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
                eff_strs.append(f"probabilistic after {eff.time}: {{ {', '.join(prob_lines)} }}")
        return f"Action(\n  Preconditions: [{pre_str}]\n  Effects:\n    " + "\n    ".join(eff_strs) + "\n)"

    def __repr__(self):
        return self.__str__()

# Operator class
class Operator:
    def __init__(self, name: str, parameters: List[Tuple[str, str]], preconditions: List[Fluent], effects: List[Union[Effect, ProbEffects]]):
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
        grounded_preconditions = [self._substitute_fluent(f, binding) for f in self.preconditions]
        grounded_effects = []
        for eff in self.effects:
            if isinstance(eff, Effect):
                grounded_fluents = {self._substitute_fluent(f, binding) for f in eff.resulting_fluents}
                grounded_effects.append(Effect(eff.time, grounded_fluents))
            elif isinstance(eff, ProbEffects):
                grounded_prob_effects = []
                for prob, effect_list in eff.prob_effects:
                    grounded_list = [
                        Effect(e.time, {self._substitute_fluent(f, binding) for f in e.resulting_fluents})
                        for e in effect_list
                    ]
                    grounded_prob_effects.append((prob, grounded_list))
                grounded_resulting_fluents = {self._substitute_fluent(f, binding) for f in eff.resulting_fluents}
                grounded_effects.append(ProbEffects(eff.time, grounded_prob_effects, grounded_resulting_fluents))
        return Action(grounded_preconditions, grounded_effects)

    def _substitute_fluent(self, fluent: Fluent, binding: Dict[str, str]) -> Fluent:
        grounded_args = tuple(binding.get(arg, arg) for arg in fluent.args)
        return Fluent(fluent.name, *grounded_args)

# State class
class State:
    def __init__(self, time: float = 0, active_fluents: Optional[Set[Fluent]] = None, upcoming_effects: Optional[PriorityQueue] = None):
        self.time = time
        self.active_fluents = active_fluents or set()
        self.upcoming_effects = upcoming_effects or PriorityQueue()

    def satisfies_precondition(self, action: Action) -> bool:
        return all(p in self.active_fluents for p in action.preconditions)

    def copy(self) -> 'State':
        new_queue = PriorityQueue()
        new_queue.queue = [x for x in self.upcoming_effects.queue]
        return State(time=self.time, active_fluents=set(self.active_fluents), upcoming_effects=new_queue)

    def __hash__(self) -> int:
        return hash(self.time) + sum(hash(f) for f in self.active_fluents)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and hash(self) == hash(other)

    def __str__(self):
        return f"State<time={self.time}, active_fluents={self.active_fluents}>"

    def __repr__(self):
        return self.__str__()

# Utility to merge fluents
def add_fluents(base: Set[Fluent], new_fluents: Set[Fluent]) -> Set[Fluent]:
    return base.union(new_fluents)

# Transition logic
def transition(state: State, action: Action) -> List[Tuple[State, float]]:
    if not state.satisfies_precondition(action):
        raise ValueError("Precondition not satisfied for applying action")

    new_state = state.copy()
    for effect in action.effects:
        new_state.upcoming_effects.put((new_state.time + effect.time, effect))

    outcomes: Dict[State, float] = {}
    _advance_to_terminal(new_state, prob=1.0, outcomes=outcomes)
    return list(outcomes.items())

def _advance_to_terminal(state: State, prob: float, outcomes: Dict[State, float]) -> None:
    while not state.upcoming_effects.empty():
        scheduled_time, effect = state.upcoming_effects.get()
        state.time = scheduled_time
        state.active_fluents = add_fluents(state.active_fluents, effect.resulting_fluents)

        if isinstance(effect, ProbEffects):
            for branch_prob, effects in effect.prob_effects:
                branched = state.copy()
                for e in effects:
                    branched.upcoming_effects.put((branched.time + e.time, e))
                _advance_to_terminal(branched, prob * branch_prob, outcomes)
            return
    outcomes[state] = prob
# Now define and use the search operator in a test case

# Define the search operator again
search_op = Operator(
    name="search",
    parameters=[
        ("?robot", "robot"),
        ("?loc_from", "location"),
        ("?loc_to", "location"),
        ("?object", "object")
    ],
    preconditions=[
        Fluent("at", "?robot", "?loc_from"),
        ~Fluent("searched", "?loc_to", "?object"),
        Fluent("free", "?robot"),
        ~Fluent("found", "?object")
    ],
    effects=[
        Effect(
            time=0,
            resulting_fluents={
                ~Fluent("free", "?robot"),
                ~Fluent("found", "?object")
            }
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
                                Fluent("found", "?object")
                            }
                        ),
                        Effect(
                            time=3,
                            resulting_fluents={
                                Fluent("holding", "?robot", "?object"),
                                ~Fluent("at", "?loc_to", "?object"),
                                Fluent("free", "?robot")
                            }
                        )
                    ]
                ),
                (
                    0.2,
                    [
                        Effect(
                            time=0,
                            resulting_fluents={
                                Fluent("free", "?robot"),
                                ~Fluent("at", "?loc_to", "?object")
                            }
                        )
                    ]
                )
            ],
            resulting_fluents={
                Fluent("at", "?robot", "?loc_to"),
                ~Fluent("at", "?robot", "?loc_from"),
                Fluent("searched", "?loc_to", "?object")
            }
        )
    ],
)

# Ground a specific instance
objects_by_type = {
    "robot": ["r1"],
    "location": ["roomA", "roomB"],
    "object": ["medkit"]
}
search_action = search_op.instantiate(objects_by_type)[0]

# Define the initial state
initial_state = State(
    time=0,
    active_fluents={
        Fluent("at", "r1", "roomA"),
        Fluent("free", "r1")
    }
)
print(initial_state)
print(search_action)

# Execute transition
outcomes = transition(initial_state, search_action)

# Show results
import pandas as pd
import ace_tools as tools

rows = []
for s, p in outcomes:
    rows.append({
        "time": s.time,
        "probability": round(p, 2),
        "fluents": ", ".join(sorted(str(f) for f in s.active_fluents))
    })

df = pd.DataFrame(rows).sort_values("probability", ascending=False)
tools.display_dataframe_to_user(name="Search Outcomes", dataframe=df)

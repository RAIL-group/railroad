import copy
import itertools

# from queue import PriorityQueue


class Fluent():
    def __init__(self, name, *args):
        self.name = name
        self.args = args

    def __str__(self):
        args_str = " ".join(self.args)
        return f"{self.name} {args_str}"

    def __repr__(self):
        return f"Fluent<{self}>"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __invert__(self):
        if self.name[:4] == 'not ':
            return Fluent(f"{self.name[4:]}", *[a for a in self.args])
        return Fluent(f"not {self.name}", *[a for a in self.args])


class State():

    def __init__(self, time=0, active_fluents=None, upcoming_effects=None):
        self.time = time
        if active_fluents is None:
            active_fluents = set()
        if upcoming_effects is None:
            upcoming_effects = PriorityQueue()
        self.active_fluents = active_fluents
        self.upcoming_effects = upcoming_effects

    def satisfies_precondition(self, action):
        for precondition in action.preconditions:
            if precondition not in self.active_fluents:
                return False
        return True

    def transition(self, action):
        if not self.satisfies_precondition(action):
            raise ValueError("Precondition not satisfied for applying action")

        state = self.copy()
        outcome_states = {}

        # Copy action effects to upcoming effects in state
        for effect in action.effects:
            state.upcoming_effects.put((effect.time + state.time, effect))

        advance_state(state, outcome_states)
        return outcome_states

    def __repr__(self):
        return f"State<time={self.time}, active_fluents={self.active_fluents}>"

    def __hash__(self):
        return hash(self.time) + sum(hash(a_f) for a_f in self.active_fluents)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def copy(self):
        upcoming_effects = PriorityQueue()
        upcoming_effects.queue = [x for x in self.upcoming_effects.queue]
        return State(
            time=self.time,
            active_fluents=set(f for f in self.active_fluents),
            upcoming_effects=upcoming_effects
        )


def advance_state(state, outcome_states, prob=1.0):
    while not state.upcoming_effects.empty():
        _, effect = state.upcoming_effects.get()
        state.time += effect.time
        state.active_fluents = add_fluents(state.active_fluents,
                                           effect.resulting_fluents)
        for f in state.active_fluents:
            if f.name == 'free':
                outcome_states[state] = (prob, state.time)

        if isinstance(effect, Effect):
            continue

        for p, effects in effect.prob_effects:
            prob_state = state.copy()
            for effect in effects:
                prob_state.upcoming_effects.put((effect.time + prob_state.time, effect))
            advance_state(prob_state, outcome_states, prob=prob * p)


class Effect():
    """
    Effect(t: float, resulting_fluents: Set[Fluent])
    """
    def __init__(self, time, resulting_fluents):
        self.time = time
        self.resulting_fluents = resulting_fluents


class ProbEffects():
    """
    ProbEffects(
        t=5,
        prob_effects=[(0.8, List[Effect]),
                      (0.2, List[Effect])
        ],
        resulting_fluents=Set[String]
    )
    """
    def __init__(self, time, prob_effects, resulting_fluents=set()):
        self.time = time
        self.prob_effects = prob_effects
        self.resulting_fluents = resulting_fluents  # Propagate into all prob_effects

class Action():
    def __init__(self, preconditions, effects):
        self.preconditions = preconditions  # List[Fluent]
        self.effects = effects              # List[Effect or ProbEffects]

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


class Operator:
    def __init__(self, name, parameters, preconditions, effects):
        """
        name: str
        parameters: List of (var_name, type_name)
        preconditions: List of Fluent
        effects: List of Effect or ProbEffects
        """
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions
        self.effects = effects

    def instantiate(self, objects_by_type):
        """
        Returns all grounded Action instances this operator can produce,
        given typed object sets (dictionary of type_name -> list of object names).
        """
        domains = [objects_by_type[typ] for _, typ in self.parameters]
        for assignment in itertools.product(*domains):
            binding = {var: obj for (var, _), obj in zip(self.parameters, assignment)}
            if not self._all_distinct(binding):
                continue
            yield self._ground(binding)

    def _ground(self, binding):
        grounded_preconditions = [self._substitute_fluent(f, binding) for f in self.preconditions]
        grounded_effects = []
        for eff in self.effects:
            if isinstance(eff, Effect):
                grounded_fluents = set(self._substitute_fluent(f, binding) for f in eff.resulting_fluents)
                grounded_effects.append(Effect(eff.time, grounded_fluents))
            elif isinstance(eff, ProbEffects):
                grounded_prob_effects = []
                for prob, effect_list in eff.prob_effects:
                    grounded_list = []
                    for e in effect_list:
                        gfluents = set(self._substitute_fluent(f, binding) for f in e.resulting_fluents)
                        grounded_list.append(Effect(e.time, gfluents))
                    grounded_prob_effects.append((prob, grounded_list))
                grounded_effects.append(ProbEffects(eff.time, grounded_prob_effects))
        return Action(preconditions=grounded_preconditions, effects=grounded_effects)

    def _substitute_fluent(self, fluent, binding):
        grounded_args = tuple(binding.get(a, a) for a in fluent.args)
        return Fluent(fluent.name, *grounded_args)

    def _all_distinct(self, binding):
        vals = list(binding.values())
        return len(set(vals)) == len(vals)




# Types and objects
objects_by_type = {
    "robot": ["r1"],
    "location": ["l1", "l2"]
}

# Operator schema
move_op = Operator(
    name="move",
    parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
    preconditions=[
        Fluent("at", "?r", "?from"),
        Fluent("connected", "?from", "?to")
    ],
    effects=[
        Effect(0, {~Fluent("at", "?r", "?from"), Fluent("at", "?r", "?to")})
    ]
)

# Instantiate grounded actions
for action in move_op.instantiate(objects_by_type):
    print(action)

search_op = Operator(
    name="search",
    parameters=[("?robot", "robot"),
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

objects_by_type = {
    "robot": ["r1"],
    "location": ["roomA", "roomB"],
    "object": ["medkit"]
}

grounded_search_actions = list(search_op.instantiate(objects_by_type))

for action in grounded_search_actions:
    print(action)
    print("-" * 40)

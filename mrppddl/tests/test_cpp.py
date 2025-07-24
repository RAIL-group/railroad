import pytest
from mrppddl._bindings import GroundedEffect, Fluent, Action, State, transition

F = Fluent

def test_cpp_fluent_equality():
    assert Fluent("at", "r1", "roomA") == Fluent("at", "r1", "roomA")
    assert Fluent("at", "r1", "roomA") == Fluent("at r1 roomA")
    assert not Fluent("at", "r1", "roomA") == Fluent("at", "r1", "roomB")
    assert not Fluent("at r1 roomA") == Fluent("at r1 roomB")
    assert not Fluent("at r1 roomA") == Fluent("at r1 rooma")

    # Test Negation
    assert Fluent("not at r1 roomA") == ~Fluent("at r1 roomA")
    assert Fluent("at r1 roomA") == ~Fluent("not at r1 roomA")
    assert not Fluent("at", "r1", "roomA") == ~Fluent("at", "r1", "roomA")
    assert not Fluent("at", "r1", "roomA") == ~Fluent("at r1 roomA")
    assert not Fluent("at", "r1", "roomA") == ~Fluent("at r1 roomA")


def test_cpp_fluent_equality_hash():

    assert hash(F("at", "r1", "roomA")) == hash(F("at", "r1", "roomA"))
    assert hash(F("at", "r1", "roomA")) == hash(F("at r1 roomA"))
    assert not hash(F("at", "r1", "roomA")) == hash(F("at", "r1", "roomB"))
    assert not hash(F("at r1 roomA")) == hash(F("at r1 roomB"))
    assert not hash(F("at r1 roomA")) == hash(F("at r1 rooma"))

    # Test Negation
    assert hash(F("not at r1 roomA")) == hash(~F("at r1 roomA"))
    assert hash(F("at r1 roomA")) == hash(~F("not at r1 roomA"))
    assert not hash(F("at", "r1", "roomA")) == hash(~F("at", "r1", "roomA"))
    assert not hash(F("at", "r1", "roomA")) == hash(~F("at r1 roomA"))
    assert not hash(F("at", "r1", "roomA")) == hash(~F("at r1 roomA"))

def test_cpp_effect_instantiation():
    f = Fluent("at r1 roomA")
    e = GroundedEffect(2.5, {f})
    pe = GroundedEffect(3.0, prob_effects=[(0.5, [e]), (0.5, [e])])
    assert len(pe.prob_effects) == 2
    for prob, e in pe.prob_effects:
        print(e)
        assert prob == 0.5
        assert len(e) == 1

def test_cpp_effect_hashing():
    f1 = F("at r1 roomA")
    f2 = F("at r1 roomB")
    e1 = GroundedEffect(2.5, {f1, f2})
    e1_alt = GroundedEffect(2.5, {f1, f2})
    e1_reordered = GroundedEffect(2.5, {f2, f1})
    e1_time_change = GroundedEffect(2.0, {f1, f2})
    e2 = GroundedEffect(2.0, {f2, f1})
    assert hash(e1) == hash(e1_alt)
    assert hash(e1) == hash(e1_reordered)
    assert not hash(e1) == hash(e1_time_change)
    assert not hash(e1) == hash(e2)
    # Doubled for caching
    assert hash(e1) == hash(e1_alt)
    assert hash(e1) == hash(e1_reordered)
    assert not hash(e1) == hash(e1_time_change)
    assert not hash(e1) == hash(e2)

    e1 = GroundedEffect(2.5, {f1})
    e2 = GroundedEffect(2.5, {f2})
    pea = GroundedEffect(3.0, prob_effects=[(0.5, [e1]), (0.5, [e2])])
    peb = GroundedEffect(3.0, prob_effects=[(0.5, [e2]), (0.5, [e1])])
    pe2 = GroundedEffect(3.0, prob_effects=[(0.5, [e2]), (0.5, [e2])])
    assert hash(pea) == hash(peb), "Failed to generate order-independent hash for prob effects."
    assert not hash(pea) == hash(pe2), "Failed to gen different hash for different prob effects."

    e1 = GroundedEffect(2.5, {f1})
    e2 = GroundedEffect(2.5, {f2})
    pea = GroundedEffect(3.0, prob_effects=[(0.5, [e1]), (0.5, [e2])])
    peb = GroundedEffect(3.5, prob_effects=[(0.5, [e1]), (0.5, [e2])])
    assert not hash(pea) == hash(peb), "Failed: has does not consider time of effect."

    e1 = GroundedEffect(2.5, {f1})
    e2 = GroundedEffect(2.5, {f2})
    pea = GroundedEffect(3.0, prob_effects=[(0.5, [e1]), (0.5, [e2])])
    peb = GroundedEffect(3.0, prob_effects=[(0.4, [e1]), (0.6, [e2])])
    assert not hash(pea) == hash(peb), "Failed: has does not consider probability of effects."

    e1 = GroundedEffect(2.0, {f1})
    e2 = GroundedEffect(5.0, {f1})
    assert not hash(e1) == hash(e2)
    pea = GroundedEffect(3.0, prob_effects=[(1.0, [e1])])
    peb = GroundedEffect(3.0, prob_effects=[(1.0, [e2])])
    assert not hash(pea) == hash(peb), "Failed: has does not consider time of probabilitstic effects."

    e1 = GroundedEffect(2.0, {f1})
    e2 = GroundedEffect(5.0, {f2})
    assert not hash(e1) == hash(e2)
    pea = GroundedEffect(3.0, prob_effects=[(0.4, [e1]), (0.6, [e2])])
    peb = GroundedEffect(3.0, prob_effects=[(0.4, [e2]), (0.6, [e1])])
    assert not hash(pea) == hash(peb), "Failed: has does not consider associativity of probabilitstic effects."

def test_cpp_action_instantiation():
    a1 = Action(preconditions={Fluent("at roomA r1")},
               effects=[GroundedEffect(2.5, {Fluent("at roomB r1"), Fluent("not at roomA r1")})],
               name="move r1 roomA roomB")
    a2 = Action(preconditions={Fluent("at roomB r1")},
               effects=[GroundedEffect(2.5, {Fluent("at roomA r1"), Fluent("not at roomB r1")})],
               name="move r1 roomB roomA")
    s = State(fluents={Fluent("at roomA r1")})
    assert s.satisfies_precondition(a1)
    assert not s.satisfies_precondition(a2)

    out = transition(s, a1)


# DEBUG

import itertools
from typing import List, Tuple, Sequence, Dict, Set
from mrppddl._bindings import get_next_actions, astar, make_goal_test
from mrppddl.core import _make_bindable, OptExpr, Binding
from mrppddl.helper import _make_callable, OptCallable

class LiftedEffectType:
    def __init__(self, time: OptExpr, resulting_fluents: Set[Fluent]):
        self.time = _make_bindable(time)
        self.resulting_fluents = resulting_fluents

    def _ground(self, binding: Binding) -> "GroundedEffect":
        grounded_time = self.time(binding)
        grounded_fluents = frozenset(
            Fluent(
                f.name, *[binding.get(arg, arg) for arg in f.args], negated=f.negated
            )
            for f in self.resulting_fluents
        )
        return GroundedEffect(grounded_time, grounded_fluents)

class Effect(LiftedEffectType):
    def __init__(self, time: OptExpr, resulting_fluents: Set[Fluent]):
        self.time = _make_bindable(time)
        self.resulting_fluents = resulting_fluents

class ProbEffect(LiftedEffectType):
    def __init__(
        self,
        time: OptExpr,
        prob_effects: List[Tuple[OptExpr, List[Effect]]],
        resulting_fluents: Set[Fluent] = set(),
    ):
        self.time = _make_bindable(time)
        self.prob_effects = [
            (_make_bindable(prob), effects) for prob, effects in prob_effects
        ]
        self.resulting_fluents = resulting_fluents

    def _ground(self, binding: Binding) -> "GroundedEffect":
        grounded_prob_effects = tuple(
            (prob(binding), tuple(e._ground(binding) for e in effect_list))
            for prob, effect_list in self.prob_effects
        )

        grounded_time: float = self.time(binding)
        grounded_resulting_fluents = frozenset(
            Fluent(
                f.name, *[binding.get(arg, arg) for arg in f.args], negated=f.negated
            )
            for f in self.resulting_fluents
        )

        return GroundedEffect(
            grounded_time, grounded_resulting_fluents, grounded_prob_effects
        )

class OperatorCpp:
    def __init__(
        self,
        name: str,
        parameters: List[Tuple[str, str]],
        preconditions: List[Fluent],
        effects: Sequence[LiftedEffectType],
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


def construct_move_operator(move_time: OptCallable):
    move_time = _make_callable(move_time)
    return OperatorCpp(
        name="move_visit",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r"), F("not visited ?to")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?r ?from")}),
            Effect(time=(move_time, ["?r", "?from", "?to"]),
                   resulting_fluents={F("free ?r"), F("at ?r ?to"), F("visited ?to")},
            ),
        ],
    )

def get_action_by_name(actions: List[Action], name: str) -> Action:
    for action in actions:
        if action.name == name:
            return action
    raise ValueError(f"No action found with name: {name}")

# @pytest.mark.skip()
def test_cpp_astar_move():

    # Get all actions
    objects_by_type = {
        "robot": ["r1", "r2"],
        "location": ["start", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        # "location": ["start", "a", "b"],
    }
    import random
    random.seed(8616)
    move_op = construct_move_operator(lambda *args: 5.0 + random.random())
    all_actions = move_op.instantiate(objects_by_type)

    # Initial state
    initial_state = State(
        time=0,
        fluents={
            F("at r1 start"), F("free r1"),
            F("at r2 start"), F("free r2"),
            F("visited start"),
        })
    # s = initial_state
    # s = transition(s, get_action_by_name(all_actions, "move r1 start a"))[0][0]
    # print(s)
    # s = transition(s, get_action_by_name(all_actions, "move r2 start b"))[0][0]
    # print(s)
    # s = transition(s, get_action_by_name(all_actions, "move r2 b c"))[0][0]
    # print(s)
    goal_fn = make_goal_test({
        F("at r1 a"),
        F("visited a"), 
        F("visited b"),
        F("visited c"),
        F("visited d"),
        # F("visited e"),
    })
    goal_fluents = {
        F("at r1 a"),
        F("visited a"),
        F("visited b"),
        F("visited c"),
        F("visited d"),
        F("visited e"),
    }
    # from mrppddl.planner import astar
    # def goal_fn(fluents):
    #     return F("at r1 a") in fluents
    from time import time
    tstart = time()
    path = astar(initial_state, all_actions, goal_fluents)
    print(time() - tstart)
    s = initial_state
    for action in path:
        print(s)
        print(action.name)
        assert s.satisfies_precondition(action)
        s = transition(s, action)[0][0]
    print("State")
    print(s)
    raise ValueError()

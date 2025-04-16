from mrppddl._bindings import GroundedEffectType, Fluent, Action, State, transition

GroundedEffect = GroundedEffectType


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

def test_cpp_effect_instantiation():
    f = Fluent("at r1 roomA")
    e = GroundedEffect(2.5, {f})
    pe = GroundedEffect(3.0, prob_effects=[(0.5, [e]), (0.5, [e])])
    assert len(pe.prob_effects) == 2
    for prob, e in pe.prob_effects:
        assert prob == 0.5


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
    print(out)
    raise ValueError()

from .mrppddl import Fluent, ActiveFluents, State, Action


def test_fluent_equality():
    assert Fluent("at", "r1", "roomA") == Fluent("at", "r1", "roomA")
    assert not Fluent("at", "r1", "roomA") == Fluent("at", "r1", "roomB")


def test_active_fluents_update_1():
    f1 = Fluent("at", "r1", "roomA")
    f2 = Fluent("free", "r1")
    f3 = Fluent("holding", "r1", "medkit")

    af = ActiveFluents({f1, f2})

    # Apply update with positive fluent and no conflict
    af = af.update({f3})
    assert f3 in af

    # Apply update with negation of f2 (should remove f2)
    af = af.update({~f2})
    assert f2 not in af
    assert f3 in af

    # Apply update with both positive and negated fluent
    af = af.update({~f3, f2})
    assert f3 not in af
    assert f2 in af


def test_active_fluents_update_2():
    af = ActiveFluents({
        Fluent('at', 'robot1', 'bedroom'),
        Fluent('free', 'robot1'),
    })

    upcoming_fluents = {
        ~Fluent('free', 'robot1'),
        ~Fluent('at', 'robot1', 'bedroom'),
        Fluent('at', 'robot1', 'kitchen'),
        ~Fluent('found', 'fork'),
    }

    af = af.update(upcoming_fluents)
    expected = ActiveFluents({
        Fluent('at', 'robot1', 'kitchen'),
    })
    assert af == expected, f"Unexpected result: {af}"

    # Now re-add a positive fluent
    af = af.update({Fluent('free', 'robot1')})
    expected = ActiveFluents({
        Fluent('free', 'robot1'),
        Fluent('at', 'robot1', 'kitchen'),
    })
    assert af == expected, f"Unexpected result after re-adding: {af}"

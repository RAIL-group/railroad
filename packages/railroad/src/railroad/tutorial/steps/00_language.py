"""Step 00 -- the language.

Fluents, states, and what a transition actually does. No planner, no
environment: just the state semantics that everything else is built on.

The point to watch for: dispatching an action does not advance the clock.
`transition` runs the world forward only until somebody is free to act again,
which is what makes concurrency fall out of the state representation rather
than out of the planner.
"""

from railroad.core import (
    Action,
    Effect,
    Fluent as F,
    GroundedEffect,
    Operator,
    State,
    get_action_by_name,
    get_next_actions,
    transition,
)


def rule(title: str) -> None:
    print(f"\n── {title} " + "─" * max(4, 58 - len(title)))


def row(label: str, value: object, note: str = "") -> None:
    text = f"  {label:<18}{value}"
    print(f"{text:<58}{note}" if note else text)


def check(expression: str, value: object) -> None:
    print(f"  {expression:<44}{value}")


def show(fluents) -> str:
    return ", ".join(sorted(str(f) for f in fluents)) or "(none)"


def show_queue(state: State) -> str:
    if not state.upcoming_effects:
        return "(none)"
    return "  ".join(
        f"t={t:.1f} {{{show(effect.resulting_fluents)}}}"
        for t, effect in state.upcoming_effects
    )


def demo() -> None:
    rule("fluents")
    check('F("at r1 roomA") == F("at","r1","roomA")',
          F("at r1 roomA") == F("at", "r1", "roomA"))
    check('~F("at r1 roomA") == F("not at r1 roomA")',
          ~F("at r1 roomA") == F("not at r1 roomA"))

    state = State(0.0, {F("at r1 roomA"), F("free r1"),
                        F("at r2 roomA"), F("free r2")})
    rule("a state is fluents + a clock + a queue")
    row("time", state.time)
    row("fluents", show(state.fluents))
    row("upcoming", show_queue(state))

    # A durative action is a list of effects at times, not a single instant:
    # the robot stops being free and stops being anywhere at t=0, and both
    # facts are restored at the destination when the move lands.
    move_r1 = Action(
        preconditions={F("at r1 roomA"), F("free r1")},
        effects=[
            GroundedEffect(0.0, {~F("free r1"), ~F("at r1 roomA")}),
            GroundedEffect(5.0, {F("free r1"), F("at r1 roomB")}),
        ],
        name="move r1 roomA roomB",
    )
    rule("one grounded action")
    for effect in move_r1.effects:
        row(f"t={effect.time:.1f}", show(effect.resulting_fluents))
    row("applicable?", state.satisfies_precondition(move_r1))

    # Lifted operators ground out to exactly that, one per legal binding.
    move = Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
            Effect(time=5.0, resulting_fluents={F("free ?r"), F("at ?r ?to")}),
        ],
    )
    actions = move.instantiate({"robot": ["r1", "r2"],
                               "location": ["roomA", "roomB"]})
    rule("the lifted operator, grounded")
    for action in actions:
        row("", action.name)

    rule("dispatch r1 -- the clock does not move")
    after_r1, _prob = transition(state, get_action_by_name(actions, "move r1 roomA roomB"))[0]
    row("time", after_r1.time, "<- r2 is still free, so nothing has to happen yet")
    row("fluents", show(after_r1.fluents))
    row("upcoming", show_queue(after_r1))

    rule("dispatch r2 -- now nobody is free")
    ready = [a.name for a in get_next_actions(after_r1, actions)]
    row("available", ", ".join(ready))
    after_r2, _prob = transition(after_r1, get_action_by_name(actions, "move r2 roomA roomB"))[0]
    row("time", after_r2.time, "<- transition ran the world forward to here")
    row("fluents", show(after_r2.fluents))
    row("upcoming", show_queue(after_r2))

    # Effects may branch. transition then returns a distribution over states.
    search = Action(
        preconditions={F("at r1 roomA"), F("free r1")},
        effects=[
            GroundedEffect(0.0, {~F("free r1")}),
            GroundedEffect(
                3.0,
                {F("free r1"), F("searched roomA cup")},
                prob_effects=[
                    (0.8, [GroundedEffect(0.0, {F("found cup"), F("at cup roomA")})]),
                    (0.2, []),
                ],
            ),
        ],
        name="search r1 roomA cup",
    )
    # One robot this time. With r2 around, transition would stop at t=0 with the
    # search still queued -- somebody is free, so the world need not move yet.
    solo = State(0.0, {F("at r1 roomA"), F("free r1")})
    rule("a probabilistic action returns a distribution")
    for outcome, prob in transition(solo, search):
        found = F("found cup") in outcome.fluents
        row(f"p={prob:.2f}", f"t={outcome.time:.1f}",
            f"found cup: {'yes' if found else 'no'}")


if __name__ == "__main__":
    demo()

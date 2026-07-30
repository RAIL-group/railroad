"""Tests for planner-side relevance projection (railroad.core + planner).

The planner may drop fluents that nothing reads, because at call time it holds
the whole problem: actions, goal, and the state's queued effects. These tests
pin the readers it must never drop and check that projection does not change
which action MCTS picks.
"""

import pytest

from railroad._bindings import Fluent as F, GroundedEffect, State
from railroad.core import (
    Action,
    Effect,
    Operator,
    ground_operators,
    project_action,
    project_state,
    relevant_predicates,
)
from railroad.planner import MCTSPlanner, seed_planner_rng


def _search_domain(n_locs=12, robots=("r1",)):
    """Ring-connected search domain with two kinds of unread fluent.

    `connected` is static (no effect touches it); `log` is dynamic write-only
    (an effect adds it, nothing ever reads it) — only relevance analysis can
    drop the second.

    With one robot a transition always drains the effect queue (it runs to a
    state where a robot is free, and there is only one). Pass a second robot
    to get states that carry in-flight effects, which is what a concurrent
    domain looks like and what projecting queued effects is about.
    """
    locs = [f"L{i}" for i in range(n_locs)]
    connected = {
        F(f"connected {a} {locs[(i + j) % n_locs]}")
        for i, a in enumerate(locs)
        for j in (1, 2)
    } | {
        F(f"connected {locs[(i + j) % n_locs]} {a}")
        for i, a in enumerate(locs)
        for j in (1, 2)
    }
    move = Operator(
        "move",
        [("?r", "robot"), ("?f", "location"), ("?t", "location")],
        [F("at ?r ?f"), F("free ?r"), F("connected ?f ?t")],
        [
            Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?f")}),
            Effect(time=2.0, resulting_fluents={F("free ?r"), F("at ?r ?t")}),
        ],
    )
    search = Operator(
        "search",
        [("?r", "robot"), ("?l", "location")],
        [F("at ?r ?l"), F("free ?r")],
        [
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(
                time=3.0,
                resulting_fluents={F("free ?r"), F("log ?l")},
                prob_effects=[
                    (0.3, [Effect(time=0, resulting_fluents={F("found Knife")})]),
                    (0.7, []),
                ],
            ),
        ],
    )
    dynamic = {f for i, r in enumerate(robots)
               for f in (F(f"at {r} {locs[i % n_locs]}"), F(f"free {r}"))}
    actions = ground_operators(
        [move, search],
        {"robot": set(robots), "location": set(locs)},
        dynamic | connected,
    ).actions
    return actions, dynamic, connected


def test_relevant_predicates_covers_every_reader():
    """Preconditions, branch conditions, the goal and core names are kept."""
    action = Action(
        {F("pre x"), ~F("neg x")},
        [
            GroundedEffect(
                1.0,
                {F("written x")},
                cond_effects=[({F("cond x")}, [GroundedEffect(0.0, {F("deep x")})])],
            )
        ],
        name="a x",
    )
    relevant = relevant_predicates([action], F("goalpred x"))

    assert {"pre", "neg", "cond", "goalpred"} <= relevant
    # The core reads these by name, not through any precondition.
    assert {"free", "waiting", "at", "found"} <= relevant
    # Written but never read anywhere: droppable.
    assert "written" not in relevant
    assert "deep" not in relevant


def test_relevant_predicates_reads_queued_effect_conditions():
    """A `when` condition on a state-carried effect is a reader too.

    This is the blind spot that makes the same analysis unsound in the
    environment: grounding sees only operators, never the effects a state
    happens to be carrying.
    """
    action = Action({F("go")}, [GroundedEffect(1.0, {F("done")})], name="a")
    queued = GroundedEffect(
        2.0,
        {F("started")},
        cond_effects=[({F("armed")}, [GroundedEffect(0.0, {F("fired")})])],
    )

    assert "armed" not in relevant_predicates([action], F("done"))
    assert "armed" in relevant_predicates([action], F("done"), [(2.0, queued)])


def test_project_action_strips_write_only_adds_recursively():
    """Irrelevant adds go, including inside probabilistic and `when` branches.

    Conditions themselves are untouched: their predicates are relevant by
    construction.
    """
    action = Action(
        {F("go")},
        [
            GroundedEffect(
                1.0,
                {F("keep me"), F("drop me")},
                prob_effects=[
                    (1.0, [GroundedEffect(0.0, {F("keep deep"), F("drop deep")})])
                ],
                cond_effects=[
                    ({F("cond")}, [GroundedEffect(0.0, {F("keep c"), F("drop c")})])
                ],
            )
        ],
        name="a",
    )
    projected = project_action(action, {"go", "keep", "cond"})

    effect = projected.effects[0]
    assert set(effect.resulting_fluents) == {F("keep me")}
    assert set(effect.prob_effects[0].effects[0].resulting_fluents) == {F("keep deep")}
    branch = effect.cond_effects[0]
    assert set(branch.conditions) == {F("cond")}  # condition preserved
    assert set(branch.effects[0].resulting_fluents) == {F("keep c")}
    # Preconditions and identity are untouched.
    assert set(projected.preconditions) == {F("go")}
    assert projected.name == "a"


def test_project_state_keeps_time_and_queued_effects():
    queued = GroundedEffect(1.0, {F("later")})
    state = State(4.0, {F("keep x"), F("drop x")}, [(5.0, queued)])
    projected = project_state(state, {"keep"})

    assert projected.time == 4.0
    assert set(projected.fluents) == {F("keep x")}
    assert len(projected.upcoming_effects) == 1


def test_project_state_strips_writes_from_queued_effects():
    """An in-flight effect must not re-add what the projection just dropped.

    Projecting only the fluent set leaves the queued effect free to
    reintroduce the irrelevant fluent the moment it fires — and a state with
    queued effects is the normal case in a concurrent domain, not an edge one.
    """
    deep = GroundedEffect(0.0, {F("keep deep"), F("drop deep")})
    queued = GroundedEffect(
        1.0,
        {F("keep top"), F("drop top")},
        [(1.0, [deep])],
        [({F("keep cond")}, [GroundedEffect(0.0, {F("drop c"), F("keep c")})])],
    )
    state = State(4.0, {F("keep x")}, [(5.0, queued)])
    projected = project_state(state, {"keep"})

    assert projected.time == 4.0
    (scheduled_time, effect), = projected.upcoming_effects
    assert scheduled_time == 5.0  # times untouched, so the heap stays valid
    assert effect.time == 1.0
    assert set(effect.resulting_fluents) == {F("keep top")}
    # Branches are rebuilt whole: dropping one would change the outcome
    # distribution, not just the fluents in it.
    assert effect.prob_effects[0].prob == 1.0
    assert set(effect.prob_effects[0].effects[0].resulting_fluents) == {F("keep deep")}
    branch = effect.cond_effects[0]
    assert set(branch.conditions) == {F("keep cond")}  # condition preserved
    assert set(branch.effects[0].resulting_fluents) == {F("keep c")}


def test_project_state_returns_the_same_state_when_nothing_drops():
    """Identity is preserved so a no-op projection costs nothing downstream."""
    queued = GroundedEffect(1.0, {F("keep later")})
    state = State(4.0, {F("keep x")}, [(5.0, queued)])
    assert project_state(state, {"keep"}) is state


def test_projection_is_a_bisimulation_over_states_with_queued_effects():
    """The invariant projection actually has to satisfy, checked exhaustively.

    Comparing which action MCTS picks is the wrong instrument: projection
    changes state hashes, so a finite-budget search may break ties differently
    — that is true of projecting the fluent set alone and predates projecting
    the queued effects. What must hold is semantic:

        transition(S, a)  restricted to relevant predicates
          ==  transition(project(S), project(a))

    including successor times and probabilities. This drives it over states
    built by firing random action prefixes, so they carry real in-flight
    effects with probabilistic and conditional branches, and over every
    applicable action in each.
    """
    import random

    from railroad.core import transition

    actions, dynamic, connected = _search_domain(n_locs=6, robots=("r1", "r2"))
    goal = F("found Knife")
    rng = random.Random(0)

    def observable(state, names):
        return (round(state.time, 9),
                frozenset(f for f in state.fluents if f.name in names))

    def outcomes(state, action, names):
        out = {}
        for succ, prob in transition(state, action):
            key = observable(succ, names)
            out[key] = round(out.get(key, 0.0) + prob, 9)
        return out

    compared = with_queued = 0
    for _ in range(60):
        state = State(0.0, dynamic | connected, [])
        for _ in range(rng.randrange(0, 4)):
            applicable = [a for a in actions if state.satisfies_precondition(a)]
            if not applicable:
                break
            succs = transition(state, rng.choice(applicable))
            state = rng.choices([s for s, _ in succs],
                                weights=[p for _, p in succs], k=1)[0]

        relevant = relevant_predicates(actions, goal, state.upcoming_effects)
        projected = project_state(state, relevant)
        if state.upcoming_effects:
            with_queued += 1

        for action in actions:
            if not state.satisfies_precondition(action):
                continue
            compared += 1
            assert outcomes(state, action, relevant) == outcomes(
                projected, project_action(action, relevant), relevant
            ), action.name

    # `log` is written by the search effect and read by nothing, so the queued
    # effects really are being rewritten rather than passed through.
    assert "log" not in relevant_predicates(actions, goal)
    assert with_queued > 10 and compared > 100


def test_projection_does_not_change_the_chosen_action():
    """Projection is bisimulation-preserving, so the decision is identical."""
    actions, dynamic, connected = _search_domain()
    state = State(0.0, dynamic | connected, [])
    goal = F("found Knife")

    choices = []
    for project in (False, True):
        seed_planner_rng(0)
        planner = MCTSPlanner(actions, project_irrelevant=project)
        choices.append(planner(state, goal, max_iterations=800, max_depth=20))
    assert choices[0] == choices[1]


def test_action_relevance_cache_matches_the_current_action_list():
    """The cached action scan must never outlive the actions it scanned.

    The scan is cached because it is a pure function of the converted actions
    and costs ~65ms on a large problem. The one thing that reassigns those
    actions mid-life is a goal carrying a negative fluent the mapping has not
    seen, which re-converts every action — so that path has to drop the cache
    too, and this pins the invariant on both sides of it.
    """
    actions, dynamic, connected = _search_domain()
    state = State(0.0, dynamic | connected, [])
    planner = MCTSPlanner(actions, project_irrelevant=True)

    planner(state, F("found Knife"), max_iterations=50)
    assert planner._actions_relevant == relevant_predicates(planner._converted_actions)

    # A goal-only negative fluent: extends the mapping, re-converts the actions.
    planner(state, ~F("searched loc0"), max_iterations=50)
    assert planner._actions_relevant == relevant_predicates(planner._converted_actions)


def test_projection_keeps_goal_over_an_otherwise_unread_predicate():
    """`log` is written and never read — unless the goal asks for it."""
    actions, dynamic, connected = _search_domain()
    state = State(0.0, dynamic | connected, [])

    assert "log" not in relevant_predicates(actions, F("found Knife"))
    assert "log" in relevant_predicates(actions, F("log L0"))

    seed_planner_rng(0)
    planner = MCTSPlanner(actions)
    # Solvable only because the goal predicate survives projection.
    assert planner(state, F("log L0"), max_iterations=800) == "search r1 L0"


def test_projected_planner_still_solves_a_negated_goal():
    """The not-* bookkeeping fluents are preconditions, so they survive."""
    hold = Operator(
        "grab", [("?x", "item")], [F("free r1"), ~F("held ?x")],
        [
            Effect(time=0, resulting_fluents={~F("free r1")}),
            Effect(time=1.0, resulting_fluents={F("free r1"), F("held ?x")}),
        ],
    )
    drop = Operator(
        "drop", [("?x", "item")], [F("free r1"), F("held ?x")],
        [
            Effect(time=0, resulting_fluents={~F("free r1")}),
            Effect(time=1.0, resulting_fluents={F("free r1"), ~F("held ?x")}),
        ],
    )
    actions = ground_operators(
        [hold, drop], {"item": {"cup"}}, {F("free r1"), F("held cup")}
    ).actions
    state = State(0.0, {F("free r1"), F("held cup")}, [])

    seed_planner_rng(0)
    planner = MCTSPlanner(actions)
    assert planner(state, ~F("held cup"), max_iterations=400) == "drop cup"


def test_found_reservation_preserves_at_implies_found():
    """`found` must survive projection even when nothing syntactically reads it.

    Object-search domains (procthor_search) state goals as `at <obj> <loc>` and
    leave `found <obj>` implicit, relying on the FF heuristic's
    `at_implies_found` augmentation. After the negative-precondition
    conversion the applicability test lives on `not-found`, so `found` is read
    by nothing but that augmentation — and the augmentation guards on
    reachability, so dropping `found` would not raise, it would quietly
    produce a weaker heuristic.
    """
    from railroad import operators as ops
    from railroad._bindings import LiteralGoal
    from railroad.core import (
        convert_goal_to_positive_preconditions,
        convert_state_to_positive_preconditions,
        ff_heuristic,
    )

    actions = ground_operators(
        [
            ops.construct_move_operator_blocking(lambda r, a, b: 5.0),
            ops.construct_search_operator(lambda r, l, o: 0.5, 10.0),
            ops.construct_pick_operator_blocking(10.0),
            ops.construct_place_operator_blocking(10.0),
        ],
        {"robot": {"r1"}, "location": {"start_loc", "kitchen", "table"},
         "object": {"Knife", "tomato_21"}},
        {F("revealed start_loc"), F("at r1 start_loc"), F("free r1")},
    ).actions

    goal = LiteralGoal(F("at Knife table"))  # `found Knife` deliberately implicit
    state = State(0.0, {
        F("revealed start_loc"), F("at r1 start_loc"), F("free r1"),
        F("found tomato_21"), F("at tomato_21 kitchen"),
    }, [])

    planner = MCTSPlanner(actions)
    planner._ensure_mapping_includes_goal(goal)
    conv_state = convert_state_to_positive_preconditions(
        state, planner._current_mapping
    )
    conv_goal = convert_goal_to_positive_preconditions(
        goal, planner._current_mapping
    )
    conv_actions = planner._converted_actions

    relevant = relevant_predicates(conv_actions, conv_goal)
    assert "found" in relevant  # reserved, though the goal never mentions it
    assert F("found tomato_21") in project_state(conv_state, relevant).fluents

    def h(names):
        return ff_heuristic(
            project_state(conv_state, names),
            conv_goal,
            [project_action(a, names) for a in conv_actions],
        )

    # Dropping `found` loses the "must find it before its location is
    # established" term: no error, just a silently cheaper estimate.
    assert h(relevant) > h(relevant - {"found"})

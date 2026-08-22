from railroad.core import (
    Effect, Fluent, Operator, State, get_action_by_name, transition,
)
from railroad.operators import _to_numeric, construct_move_visited_operator
from railroad.operators import construct_search_and_pick_operator
from railroad import operators
from railroad._bindings import get_next_actions
from railroad.planner import MCTSPlanner, get_usable_actions

import pytest
import random

F = Fluent


def test_no_op_offered_only_when_only_option():
    move_op = operators.construct_move_operator_blocking(lambda r, a, b: 1.0)
    no_op = operators.construct_no_op_operator(no_op_time=5.0)
    objects = {"robot": {"r1"}, "location": {"a", "b", "c"}}
    move_actions = list(move_op.instantiate(objects))
    no_op_actions = list(no_op.instantiate(objects))

    state = State(0.0, {F("at r1 a"), F("free r1")})

    # With moves available, the planner must not be offered the wait action.
    names = {x.name for x in get_next_actions(state, move_actions + no_op_actions)}
    assert any(n.startswith("move r1 a") for n in names)
    assert not any(n.startswith("no_op") for n in names)

    # When waiting is the only applicable action, it is offered.
    only_wait = {x.name for x in get_next_actions(state, no_op_actions)}
    assert only_wait == {"no_op r1"}


def test_pruning_unavailable_actions():
    """Pruning is per-robot, and keeps more than the currently-applicable moves.

    `len(after) < len(before)` used to be the whole assertion, which any
    pruning at all satisfies. What actually happens is narrower and worth
    pinning: r2 and r3 are neither placed nor free, so every one of their
    actions goes; r1 keeps *all* of its 132, including moves from locations it
    is not standing in. Pruning drops robots that cannot act, not actions that
    do not currently apply.
    """
    objects_by_type = {
        "robot": ["r1", "r2", "r3"],
        "location": ["start", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
    }
    random.seed(8616)
    move_op = construct_move_visited_operator(lambda *args: 5.0 + random.random())
    all_actions = move_op.instantiate(objects_by_type)
    assert len(all_actions) == 3 * 12 * 11

    initial_state = State(
        time=0,
        fluents={F("at r1 start"), F("free r1"), F("visited start")},
    )
    names = {a.name for a in get_usable_actions(initial_state, all_actions)}

    assert all(n.startswith("move r1 ") for n in names)
    assert len(names) == 12 * 11


@pytest.mark.parametrize(
    "initial_fluents",
    [
        {F("at r1 start"), F("free r1"), F("visited start")},
        {
            F("at r1 start"),
            F("free r1"),
            F("at r2 start"),
            F("free r2"),
            F("visited start"),
        },
        {
            F("at r1 start"),
            F("free r1"),
            F("at r2 start"),
            F("free r2"),
            F("at r3 start"),
            F("free r3"),
            F("visited start"),
        },
    ],
    ids=["one robot", "two robots", "three robots"],
)
def test_planner_mcts_move_visit_multirobot(initial_fluents):
    # Get all actions
    objects_by_type = {
        "robot": ["r1", "r2", "r3"],
        "location": ["start", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
    }
    random.seed(8616)
    move_op = construct_move_visited_operator(lambda *args: 5.0 + 5 * random.random())
    all_actions = move_op.instantiate(objects_by_type)

    # Initial state
    initial_state = State(time=0, fluents=initial_fluents)
    goal = (
        F("visited a") &
        F("visited b") &
        F("visited c") &
        F("visited d") &
        F("visited e")
    )

    state = initial_state
    all_actions = get_usable_actions(initial_state, all_actions)
    mcts = MCTSPlanner(all_actions)
    for _ in range(15):
        if goal.evaluate(state.fluents):
            break
        action_name = mcts(state, goal, 2000, c=10)
        if action_name == "NONE":
            break
        action = get_action_by_name(all_actions, action_name)

        state = transition(state, action)[0][0]
    assert goal.evaluate(state.fluents)


# `attempts` is the sample count behind the >=80% claim below. The roomA_prob=1.0
# rows are degenerate -- roomA strictly dominates the 0.4 alternatives, so there is
# no distribution left to sample and 20 draws only cost time. The rows that carry
# the statistical claim keep their full sample.
@pytest.mark.parametrize(("roomA_prob", "num_robots", "attempts"), [
    (1.0, 1, 5),
    (0.8, 1, 20),
    (0.6, 1, 20),
    (1.0, 2, 5),
    (0.8, 2, 20),
    (0.6, 2, 20),
])
def test_mcts_search_picks_more_likely_location(roomA_prob, num_robots, attempts):
    # Define objects
    objects_by_type = {
        "robot": ["r1", "r2"],
        "location": ["start", "roomA", "roomB"],
        "object": ["cup", "bowl"],
    }

    # Parametrized search probability model
    def object_search_prob(robot, search_loc, obj):
        if search_loc == "roomA":
            return roomA_prob
        else:
            return 0.4  # same as your original default for non-roomA

    # Ground actions
    search_actions = construct_search_and_pick_operator(
        object_search_prob, 5.0, 3
    ).instantiate(objects_by_type)

    # Initial state
    if num_robots == 1:
        initial_state = State(
            time=0,
            fluents={Fluent("at r1 start"), Fluent("free r1"),})
        goal = Fluent("found bowl")
    elif num_robots == 2:
        initial_state = State(
            time=0,
            fluents={
                Fluent("at r1 start"),
                Fluent("at r2 start"),
                Fluent("free r1"),
                Fluent("free r2"),
            })
        goal = Fluent("found cup") & Fluent("found bowl")
    else:
        raise ValueError(f"num_robots {num_robots} unsupported.")
    all_actions = search_actions
    mcts = MCTSPlanner(all_actions)

    # Run MCTS N times and collect chosen actions
    selected_actions = []
    num_planning_attempts = attempts
    for _ in range(num_planning_attempts):
        action = mcts(initial_state, goal, max_iterations=10000, c=100, heuristic_multiplier=2)
        selected_actions.append(action)

    # Count how many selected actions mention roomA
    roomA_count = sum("roomA" in str(action) for action in selected_actions)

    # We expect roomA to appear in at least 80% of planning attempts
    assert (
        roomA_count >= 0.8 * num_planning_attempts
    ), f"Expected roomA in at least 80% actions, got {roomA_count}/{num_planning_attempts} for roomA_prob={roomA_prob}"


def test_mcts_planner_lambdas_propagate_to_heuristic():
    """Lambdas set on MCTSPlanner construction should drive both the search
    heuristic and the .heuristic() helper. We verify by configuring three
    planners (h_add-only, h_max-only, h_ff-only) over the same problem and
    asserting the helper returns the expected mixed values."""
    objects_by_type = {
        "robot": ["r1"],
        "location": ["start", "target"],
    }
    move_op = construct_move_visited_operator(lambda *args: 5.0)
    all_actions = move_op.instantiate(objects_by_type)

    initial_state = State(
        time=0,
        fluents={F("at r1 start"), F("free r1"), F("visited start")},
    )
    # AND goal with two fluents whose optimistic costs differ:
    #   visited target  -> needs move(start->target), optimistic_cost = 5
    #   at r1 target    -> same move, optimistic_cost = 5
    goal = F("visited target") & F("at r1 target")

    mcts_add = MCTSPlanner(all_actions, lambda_add=1.0, lambda_max=0.0, lambda_ff=0.0)
    mcts_max = MCTSPlanner(all_actions, lambda_add=0.0, lambda_max=1.0, lambda_ff=0.0)
    mcts_ff  = MCTSPlanner(all_actions, lambda_add=0.0, lambda_max=0.0, lambda_ff=1.0)

    h_add = mcts_add.heuristic(initial_state, goal)
    h_max = mcts_max.heuristic(initial_state, goal)
    h_ff  = mcts_ff.heuristic(initial_state, goal)

    # Both fluents share the single move action: optimistic_cost = 5 each.
    # h_add sums them (10), h_max maxes (5), h_ff sums unique action durations (5).
    assert h_add == 10.0
    assert h_max == 5.0
    assert h_ff == 5.0

    # Read-back of stored weights via the C++ binding.
    assert mcts_max._cpp_planner.lambda_max == 1.0
    assert mcts_ff._cpp_planner.lambda_ff == 1.0


def _fragile_delivery_actions():
    """Two items; dropping the fragile one unpadded breaks it irreversibly.

    The goal needs the vase unbroken, so `drop-off vase` before `pad vase` is
    a dead end the delete-relaxation *can* see (h = inf).
    """
    from railroad.core import Effect, Operator

    pad = Operator(
        name="pad",
        parameters=[("?r", "robot"), ("?x", "item")],
        preconditions=[F("free ?r"), F("holding ?x")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(time=1.0, resulting_fluents={F("padded ?x"), F("free ?r")}),
        ],
    )
    drop = Operator(
        name="drop-off",
        parameters=[("?r", "robot"), ("?x", "item")],
        preconditions=[F("free ?r"), F("holding ?x")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(
                time=1.0,
                resulting_fluents={
                    ~F("holding ?x"), F("delivered ?x"), F("free ?r"),
                },
                cond_effects=[
                    ({F("fragile ?x"), ~F("padded ?x")},
                     [Effect(time=0, resulting_fluents={F("broken ?x")})]),
                ],
            ),
        ],
    )
    objects = {"robot": {"r1"}, "item": {"vase", "brick"}}
    return [a for op in (pad, drop) for a in op.instantiate(objects)]


def _drive_to_goal(planner, state, goal, actions, max_steps=10, seed=0):
    rng = random.Random(seed)
    plan = []
    for _ in range(max_steps):
        if goal.evaluate(state.fluents):
            return True, plan
        name = planner(state, goal, max_iterations=2000, c=100)
        if name == "NONE":
            return False, plan
        successors = transition(state, get_action_by_name(actions, name))
        state = successors[
            rng.choices(range(len(successors)),
                        weights=[p for _, p in successors], k=1)[0]
        ][0]
        plan.append(name)
    return goal.evaluate(state.fluents), plan


@pytest.mark.parametrize("penalty, should_solve", [(None, False), (1e4, True)])
def test_dead_end_penalty_steers_mcts_away_from_dead_ends(penalty, should_solve):
    """A flat failure cost makes MCTS avoid a dead end it otherwise seeks.

    With `dead_end_penalty=None` the reward for an h = inf state is clamped
    to -(time + cost), which is *higher* than any reachable state's reward --
    so the search is drawn to the dead end and drops the vase unpadded.
    """
    actions = _fragile_delivery_actions()
    initial = State(
        0.0,
        {F("free r1"), F("holding vase"), F("holding brick"), F("fragile vase")},
        [],
    )
    goal = F("delivered vase") & F("delivered brick") & ~F("broken vase")

    planner = MCTSPlanner(actions, dead_end_penalty=penalty)
    solved, plan = _drive_to_goal(planner, initial, goal, actions)

    assert solved is should_solve, plan
    if should_solve:
        assert plan.index("pad r1 vase") < plan.index("drop-off r1 vase")


def test_dead_end_penalty_is_flat_not_added_to_elapsed_cost():
    """The penalty replaces the branch's accrued cost rather than adding to it.

    Two dead ends reached at different depths must be scored identically, so
    the search expresses no preference for failing quickly. Reaching one via
    an extra `pad brick` step (pure delay, no progress) must not change the
    action chosen at the root.
    """
    actions = _fragile_delivery_actions()
    goal = F("delivered vase") & F("delivered brick") & ~F("broken vase")
    base = {F("free r1"), F("holding vase"), F("holding brick"), F("fragile vase")}

    planner = MCTSPlanner(actions, dead_end_penalty=1e4)
    early = State(0.0, set(base), [])
    # Same problem, but the clock has already advanced a long way.
    late = State(50.0, set(base), [])

    assert planner(early, goal, max_iterations=2000, c=100) == "pad r1 vase"
    assert planner(late, goal, max_iterations=2000, c=100) == "pad r1 vase"


# ---------------------------------------------------------------------------
# Feasibility (was test_feasible_actions.py)
# ---------------------------------------------------------------------------


def _no_revisit_move_op(move_time):
    """`construct_move_visited_operator` *plus* a `not visited ?to` precondition.

    The production builder deliberately omits that precondition -- its docstring
    says so, to allow revisits -- which is exactly why this test declares its
    own: forbidding revisits is what makes the feasible set small enough to
    assert exactly.
    """
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r"), F("not visited ?to")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(
                time=(_to_numeric(move_time), ["?r", "?from", "?to"]),
                resulting_fluents={
                    F("free ?r"),
                    F("not at ?r ?from"),
                    F("at ?r ?to"),
                    F("visited ?to"),
                },
            ),
        ],
    )


def test_feasible_set_shrinks_to_exactly_one_as_a_robot_commits():
    """Exact sets at each step, not just "fewer than before".

    Two robots can reach roomA; once r1 commits, r2's move is the only feasible
    action left, and the planner must choose it.
    """
    state = State(
        time=0,
        fluents={
            F("free r1"), F("free r2"),
            F("visited r1_loc"), F("visited r2_loc"),
            F("at r1 r1_loc"), F("at r2 r2_loc"),
        },
    )
    objects_by_type = {
        "robot": ["r1", "r2"],
        "location": ["r1_loc", "r2_loc", "roomA"],
    }
    all_actions = _no_revisit_move_op(lambda r, a, b: 5.0).instantiate(objects_by_type)

    assert {a.name for a in all_actions if state.satisfies_precondition(a)} == {
        "move r1 r1_loc roomA",
        "move r2 r2_loc roomA",
    }

    next_state = transition(
        state, get_action_by_name(all_actions, "move r1 r1_loc roomA")
    )[0][0]

    # r1 is committed, r2 is untouched, and roomA is not visited until arrival.
    assert F("free r1") not in next_state.fluents
    assert F("free r2") in next_state.fluents
    assert F("visited roomA") not in next_state.fluents

    assert [a.name for a in all_actions if next_state.satisfies_precondition(a)] == [
        "move r2 r2_loc roomA"
    ]

    action_name = MCTSPlanner(all_actions)(
        next_state, F("visited roomA"),
        max_iterations=1000, c=10, max_depth=10, heuristic_multiplier=2,
    )
    assert action_name == "move r2 r2_loc roomA"

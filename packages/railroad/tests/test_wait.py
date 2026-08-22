import pytest

from railroad.core import (
    Fluent,
    State,
    transition,
    get_action_by_name,
    GroundedEffect,
    Action,
    Operator,
    Effect,
)
from railroad.operators import construct_move_operator, construct_move_visited_operator, construct_wait_operator
from railroad.planner import MCTSPlanner, get_usable_actions
import random

F = Fluent


def test_wait_for_transition():
    state = State(0, {F("free r1"), F("free r2")})
    # Action 1 is for robot 1: work quickly
    action_1 = Action(name="work r1",
                      preconditions={F("free r1")},
                      effects=[
                          GroundedEffect(0, {F("not free r1")}),
                          GroundedEffect(1.0, {F("free r1")})
                      ])
    # Action 2 is for robot 2: work slowly
    action_2 = Action(name="work r2",
                      preconditions={F("free r2")},
                      effects=[
                          GroundedEffect(0, {F("not free r2")}),
                          GroundedEffect(2.0, {F("free r2")})
                      ])
    # Action 3 is for robot 1: wait for r2
    action_3 = Action(name="wait r1 r2",
                      preconditions={F("free r1"), F("not free r2")},
                      effects=[
                          GroundedEffect(0, {F("not free r1"), F("waiting r1 r2")}),
                      ])

    state = transition(state, action_1)[0][0]
    assert state.time == 0

    state = transition(state, action_2)[0][0]
    assert state.time == 1

    state = transition(state, action_3)[0][0]
    assert state.time == 2
    assert F("free r1") in state.fluents
    assert F("waiting r1 r2") not in state.fluents
    assert F("free r2") in state.fluents
    assert F("free r3") not in state.fluents

@pytest.mark.parametrize(
    "initial_fluents",
    [
        {
            F("at r1 start"),
            F("free r1"),
            F("at r2 start"),
            F("free r2"),
            F("at r3 start"),
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
    ids=["two robots", "three robots"],
)
@pytest.mark.slow
def test_planner_mcts_move_visit_wait_multirobot(initial_fluents):
    # Get all actions
    objects_by_type = {
       "robot": [f.args[0] for f in initial_fluents
                 if f.name.split()[0] == 'free'],
        "location": ["start", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
    }
    random.seed(8616)
    move_op = construct_move_visited_operator(lambda *args: 5.0 + random.random())
    wait_op = construct_wait_operator()
    all_actions = move_op.instantiate(objects_by_type) + wait_op.instantiate(objects_by_type)


    # Initial state
    initial_state = State(time=0, fluents=initial_fluents)
    goal = (
        F("at r1 start") &
        F("at r2 start") &
        F("at r3 start") &
        F("visited a") &
        F("visited b") &
        F("visited c") &
        F("visited d") &
        F("visited e")
    )
    all_actions = get_usable_actions(initial_state, all_actions)

    state = initial_state
    mcts = MCTSPlanner(all_actions)
    for _ in range(25):
        if goal.evaluate(state.fluents):
            break
        action_name = mcts(state, goal, 10000, c=5)
        if action_name == "NONE":
            break
        action = get_action_by_name(all_actions, action_name)

        state = transition(state, action)[0][0]

    assert goal.evaluate(state.fluents)


# Distances between the four locations. r1 starts far from the couch (5s), r2
# close (2s), so r2 necessarily arrives first and has to wait.
_COUCH_DISTANCES = {
    ("l1", "l3"): 5.0, ("l3", "l1"): 5.0,
    ("l2", "l3"): 2.0, ("l3", "l2"): 2.0,
    ("l1", "l2"): 3.0, ("l2", "l1"): 3.0,
    ("l1", "l4"): 7.0, ("l4", "l1"): 7.0,
    ("l2", "l4"): 4.0, ("l4", "l2"): 4.0,
    ("l3", "l4"): 3.0, ("l4", "l3"): 3.0,
}

_COUCH_OBJECTS = {
    "robot": ["r1", "r2"],
    "couch": ["couch1"],
    "location": ["l1", "l2", "l3", "l4"],
}


def _couch_actions():
    """Every grounding of the two-robot couch-carrying domain.

    Shared by both couch tests. The fixed-plan test below used to hand-write
    six `Action` objects that were exactly these groundings -- two copies of
    one domain in one file, with nothing keeping them in step.
    """
    operators = [
        construct_move_operator(
            lambda r, frm, to: _COUCH_DISTANCES.get((frm, to), 1.0)),
        construct_wait_operator(),
        construct_lift_couch_operator(lift_time=1.0),
        construct_move_couch_operator(move_time=3.0),
        construct_put_down_couch_operator(put_down_time=1.0),
    ]
    return {a.name: a
            for op in operators
            for a in op.instantiate(_COUCH_OBJECTS)}


def _couch_initial_state():
    return State(time=0, fluents={
        F("at r1 l1"), F("at r2 l2"), F("at couch1 l3"),
        F("on-floor couch1"), F("free r1"), F("free r2"),
    })


#: (action, time after it resolves, fluents that must hold, fluents that must not)
_COUCH_PLAN = [
    ("move r1 l1 l3", 0.0,
     set(), {"free r1", "at r1 l1"}),
    # r2 arrives at t=2; r1 is still in transit.
    ("move r2 l2 l3", 2.0,
     {"free r2", "at r2 l3"}, {"free r1"}),
    # Waiting advances time to r1's arrival and then releases r2.
    ("wait r2 r1", 5.0,
     {"free r1", "at r1 l3", "free r2"}, {"waiting r2 r1"}),
    # The primary carrier goes free after the lift; the helper stays committed.
    ("lift-couch-together r1 r2 couch1 l3", 6.0,
     {"carrying-primary r1 couch1", "carrying-helper r2 couch1", "free r1"},
     {"on-floor couch1", "free r2"}),
    ("move-couch r1 r2 couch1 l3 l4", 9.0,
     {"at r1 l4", "at r2 l4", "at couch1 l4", "free r1"}, {"free r2"}),
    ("put-down-couch-together r1 r2 couch1 l4", 10.0,
     {"on-floor couch1", "at couch1 l4", "free r1", "free r2"},
     {"carrying-primary r1 couch1", "carrying-helper r2 couch1"}),
]


def test_couch_carry_with_wait():
    """A fixed plan through the couch domain, checked step by step.

    Two robots must carry a couch together. r2 reaches it first and waits;
    they lift, carry and put down as a pair. The point of the fixed plan (as
    opposed to `test_couch_carry_with_operators_and_planner`, which lets the
    planner find it) is that every intermediate *time* is pinned, which is what
    makes the wait and the concurrent execution observable.
    """
    actions = _couch_actions()
    state = _couch_initial_state()

    for name, expected_time, present, absent in _COUCH_PLAN:
        state = transition(state, actions[name])[0][0]
        assert state.time == expected_time, f"wrong time after {name!r}"
        for fluent in present:
            assert F(fluent) in state.fluents, f"{fluent!r} missing after {name!r}"
        for fluent in absent:
            assert F(fluent) not in state.fluents, f"{fluent!r} present after {name!r}"

    # Goal: the couch is at l4 and back on the floor.
    assert F("at couch1 l4") in state.fluents
    assert F("on-floor couch1") in state.fluents


def construct_lift_couch_operator(lift_time: float = 1.0):
    """
    Operator for two robots to lift a couch together.
    Both robots must be at the same location as the couch and free.
    After lifting, r1 becomes the primary carrier (free after lift_time),
    r2 becomes the helper (stays not-free until put-down).
    """
    return Operator(
        name="lift-couch-together",
        parameters=[
            ("?r1", "robot"),
            ("?r2", "robot"),
            ("?c", "couch"),
            ("?loc", "location"),
        ],
        preconditions=[
            F("at ?r1 ?loc"),
            F("at ?r2 ?loc"),
            F("at ?c ?loc"),
            F("on-floor ?c"),
            F("free ?r1"),
            F("free ?r2"),
            ~F("= ?r1 ?r2"),  # Different robots
        ],
        effects=[
            Effect(
                time=0,
                resulting_fluents={
                    ~F("on-floor ?c"),
                    ~F("free ?r1"),
                    ~F("free ?r2"),
                    F("carrying-primary ?r1 ?c"),
                    F("carrying-helper ?r2 ?c"),
                },
            ),
            Effect(
                time=lift_time,
                resulting_fluents={
                    F("free ?r1"),  # Primary becomes free
                    # Helper stays not-free
                },
            ),
        ],
    )


def construct_move_couch_operator(move_time: float = 3.0):
    """
    Operator for two robots to move a couch together.
    Primary robot must be free, both must be carrying the couch.
    """
    return Operator(
        name="move-couch",
        parameters=[
            ("?r1", "robot"),
            ("?r2", "robot"),
            ("?c", "couch"),
            ("?from", "location"),
            ("?to", "location"),
        ],
        preconditions=[
            F("carrying-primary ?r1 ?c"),
            F("carrying-helper ?r2 ?c"),
            F("at ?r1 ?from"),
            F("at ?r2 ?from"),
            F("at ?c ?from"),
            F("free ?r1"),
            ~F("= ?r1 ?r2"),
        ],
        effects=[
            Effect(
                time=0,
                resulting_fluents={
                    ~F("free ?r1"),
                    ~F("at ?r1 ?from"),
                    ~F("at ?r2 ?from"),
                    ~F("at ?c ?from"),
                },
            ),
            Effect(
                time=move_time,
                resulting_fluents={
                    F("free ?r1"),
                    F("at ?r1 ?to"),
                    F("at ?r2 ?to"),
                    F("at ?c ?to"),
                },
            ),
        ],
    )


def construct_put_down_couch_operator(put_down_time: float = 1.0):
    """
    Operator for two robots to put down a couch together.
    Both robots become free after put_down_time.
    """
    return Operator(
        name="put-down-couch-together",
        parameters=[
            ("?r1", "robot"),
            ("?r2", "robot"),
            ("?c", "couch"),
            ("?loc", "location"),
        ],
        preconditions=[
            F("carrying-primary ?r1 ?c"),
            F("carrying-helper ?r2 ?c"),
            F("at ?r1 ?loc"),
            F("at ?r2 ?loc"),
            F("at ?c ?loc"),
            F("free ?r1"),
            ~F("= ?r1 ?r2"),
        ],
        effects=[
            Effect(
                time=0,
                resulting_fluents={
                    ~F("free ?r1"),
                },
            ),
            Effect(
                time=put_down_time,
                resulting_fluents={
                    F("on-floor ?c"),
                    ~F("carrying-primary ?r1 ?c"),
                    ~F("carrying-helper ?r2 ?c"),
                    F("free ?r1"),
                    F("free ?r2"),
                },
            ),
        ],
    )


def test_couch_carry_with_operators_and_planner():
    """
    Complete test of couch-carrying with operators and MCTS planner.

    Scenario:
    - r1 starts at l1 (far from couch)
    - r2 starts at l2 (closer to couch)
    - couch1 is at l3 (on floor)
    - Goal: couch1 at l4 (on floor)

    The planner should discover that:
    1. Both robots need to move to l3
    2. One robot needs to wait for the other
    3. They lift the couch together
    4. They move it to l4
    5. They put it down
    """

    # Define object types
    all_actions = list(_couch_actions().values())


    # Initial state
    initial_state = State(
        time=0,
        fluents={
            F("at r1 l1"),
            F("at r2 l2"),
            F("at couch1 l3"),
            F("on-floor couch1"),
            F("free r1"),
            F("free r2"),
        }
    )

    # Goal
    goal = F("at couch1 l4") & F("on-floor couch1")

    # Filter to only usable actions
    usable_actions = get_usable_actions(initial_state, all_actions)

    # Run MCTS planner
    state = initial_state
    mcts = MCTSPlanner(usable_actions)

    max_steps = 20
    for step in range(max_steps):
        if goal.evaluate(state.fluents):
            break

        # Get next action from MCTS
        action_name = mcts(
            state,
            goal,
            max_iterations=5000,
            max_depth=20,
            c=100
        )

        if action_name == "NONE":
            break

        # Execute action
        action = get_action_by_name(usable_actions, action_name)
        next_states = transition(state, action)
        state = next_states[0][0]  # Take first (deterministic) outcome

    # Verify goal was achieved
    assert goal.evaluate(state.fluents), f"Goal not achieved after {max_steps} steps"

    # Verify both robots are free at the end
    assert F("free r1") in state.fluents, "r1 should be free at the end"
    assert F("free r2") in state.fluents, "r2 should be free at the end"

    # Verify couch is at l4 and on floor
    assert F("at couch1 l4") in state.fluents
    assert F("on-floor couch1") in state.fluents


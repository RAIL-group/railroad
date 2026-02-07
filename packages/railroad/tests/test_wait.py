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
            print("Goal found!")
            break
        action_name = mcts(state, goal, 10000, c=5)
        if action_name == "NONE":
            break
        action = get_action_by_name(all_actions, action_name)

        state = transition(state, action)[0][0]

        print(action_name, state, goal.evaluate(state.fluents))
    assert goal.evaluate(state.fluents)


def test_couch_carry_with_wait():
    """
    Test scenario: Two robots must move a couch together.
    - r1 starts at l1 (5 seconds away from couch)
    - r2 starts at l2 (2 seconds away from couch)
    - couch1 is at l3 (on floor)
    - Goal: couch1 at l4 (on floor)

    Expected behavior:
    1. Both robots move to l3 (r2 arrives first)
    2. r2 waits for r1 to arrive
    3. Once both at l3, they lift the couch together
    4. They move the couch to l4
    5. They put down the couch at l4
    """

    # Define initial state
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

    # Define actions
    # Action 1: r1 moves from l1 to l3 (takes 5 seconds)
    move_r1_l1_l3 = Action(
        name="move r1 l1 l3",
        preconditions={F("at r1 l1"), F("free r1")},
        effects=[
            GroundedEffect(0, {~F("free r1"), ~F("at r1 l1")}),
            GroundedEffect(5.0, {F("free r1"), F("at r1 l3")}),
        ]
    )

    # Action 2: r2 moves from l2 to l3 (takes 2 seconds)
    move_r2_l2_l3 = Action(
        name="move r2 l2 l3",
        preconditions={F("at r2 l2"), F("free r2")},
        effects=[
            GroundedEffect(0, {~F("free r2"), ~F("at r2 l2")}),
            GroundedEffect(2.0, {F("free r2"), F("at r2 l3")}),
        ]
    )

    # Action 3: r2 waits for r1
    wait_r2_r1 = Action(
        name="wait r2 r1",
        preconditions={F("free r2"), ~F("free r1")},
        effects=[
            GroundedEffect(0, {~F("free r2"), F("waiting r2 r1")}),
        ]
    )

    # Action 4: lift couch together (both robots must be at l3, both free)
    lift_couch = Action(
        name="lift-couch-together r1 r2 couch1 l3",
        preconditions={
            F("at r1 l3"),
            F("at r2 l3"),
            F("at couch1 l3"),
            F("on-floor couch1"),
            F("free r1"),
            F("free r2"),
        },
        effects=[
            GroundedEffect(0, {
                ~F("on-floor couch1"),
                ~F("free r1"),
                ~F("free r2"),
                F("carrying-primary r1 couch1"),
                F("carrying-helper r2 couch1"),
            }),
            GroundedEffect(1.0, {
                F("free r1"),  # Primary robot becomes free after lift
                # Note: helper robot stays not-free until put-down
            }),
        ]
    )

    # Action 5: move couch from l3 to l4 (takes 3 seconds)
    move_couch = Action(
        name="move-couch r1 r2 couch1 l3 l4",
        preconditions={
            F("carrying-primary r1 couch1"),
            F("carrying-helper r2 couch1"),
            F("at r1 l3"),
            F("at r2 l3"),
            F("at couch1 l3"),
            F("free r1"),
        },
        effects=[
            GroundedEffect(0, {
                ~F("free r1"),
                ~F("at r1 l3"),
                ~F("at r2 l3"),
                ~F("at couch1 l3"),
            }),
            GroundedEffect(3.0, {
                F("free r1"),
                F("at r1 l4"),
                F("at r2 l4"),
                F("at couch1 l4"),
            }),
        ]
    )

    # Action 6: put down couch at l4
    put_down_couch = Action(
        name="put-down-couch-together r1 r2 couch1 l4",
        preconditions={
            F("carrying-primary r1 couch1"),
            F("carrying-helper r2 couch1"),
            F("at r1 l4"),
            F("at r2 l4"),
            F("at couch1 l4"),
            F("free r1"),
        },
        effects=[
            GroundedEffect(0, {
                ~F("free r1"),
            }),
            GroundedEffect(1.0, {
                F("on-floor couch1"),
                ~F("carrying-primary r1 couch1"),
                ~F("carrying-helper r2 couch1"),
                F("free r1"),
                F("free r2"),
            }),
        ]
    )

    # Execute the sequence
    state = initial_state
    print(f"\nInitial state (t={state.time}): {sorted(str(f) for f in state.fluents)}")

    # Step 1: r1 starts moving to l3
    state = transition(state, move_r1_l1_l3)[0][0]
    print(f"\nAfter 'move r1 l1 l3' (t={state.time}):")
    print("  - r1 is moving (not free)")
    assert state.time == 0
    assert F("free r1") not in state.fluents
    assert F("at r1 l1") not in state.fluents

    # Step 2: r2 starts moving to l3
    state = transition(state, move_r2_l2_l3)[0][0]
    print(f"\nAfter 'move r2 l2 l3' (t={state.time}):")
    print("  - r2 is moving (not free)")
    print("  - Time advances to t=2 when r2 arrives at l3")
    assert state.time == 2  # Time advances to when r2 finishes
    assert F("free r2") in state.fluents
    assert F("at r2 l3") in state.fluents
    assert F("free r1") not in state.fluents  # r1 still moving

    # Step 3: r2 waits for r1
    state = transition(state, wait_r2_r1)[0][0]
    print(f"\nAfter 'wait r2 r1' (t={state.time}):")
    print("  - r2 is waiting (not free)")
    print("  - Time advances to t=5 when r1 arrives at l3")
    assert state.time == 5  # Time advances to when r1 finishes
    assert F("free r1") in state.fluents
    assert F("at r1 l3") in state.fluents
    assert F("free r2") in state.fluents  # r2 becomes free when r1 arrives
    assert F("waiting r2 r1") not in state.fluents

    # Step 4: Both robots lift the couch
    state = transition(state, lift_couch)[0][0]
    print(f"\nAfter 'lift-couch-together r1 r2 couch1 l3' (t={state.time}):")
    print("  - Couch is no longer on floor")
    print("  - r1 is primary carrier (becomes free after 1 second)")
    print("  - r2 is helper (stays not-free)")
    assert state.time == 6  # 5 + 1 second lift time
    assert F("on-floor couch1") not in state.fluents
    assert F("carrying-primary r1 couch1") in state.fluents
    assert F("carrying-helper r2 couch1") in state.fluents
    assert F("free r1") in state.fluents
    assert F("free r2") not in state.fluents  # Helper stays not-free

    # Step 5: Move couch to l4
    state = transition(state, move_couch)[0][0]
    print(f"\nAfter 'move-couch r1 r2 couch1 l3 l4' (t={state.time}):")
    print("  - Couch and both robots at l4")
    assert state.time == 9  # 6 + 3 seconds travel
    assert F("at r1 l4") in state.fluents
    assert F("at r2 l4") in state.fluents
    assert F("at couch1 l4") in state.fluents
    assert F("free r1") in state.fluents
    assert F("free r2") not in state.fluents  # Helper still not-free

    # Step 6: Put down couch
    state = transition(state, put_down_couch)[0][0]
    print(f"\nAfter 'put-down-couch-together r1 r2 couch1 l4' (t={state.time}):")
    print("  - Couch is on floor at l4")
    print("  - Both robots are free")
    assert state.time == 10  # 9 + 1 second put-down time
    assert F("on-floor couch1") in state.fluents
    assert F("at couch1 l4") in state.fluents
    assert F("carrying-primary r1 couch1") not in state.fluents
    assert F("carrying-helper r2 couch1") not in state.fluents
    assert F("free r1") in state.fluents
    assert F("free r2") in state.fluents

    # Verify goal is achieved
    print("\nGoal achieved: couch1 is at l4 and on floor")
    print(f"Total time: {state.time} seconds")
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
    objects_by_type = {
        "robot": ["r1", "r2"],
        "couch": ["couch1"],
        "location": ["l1", "l2", "l3", "l4"],
    }

    # Define travel times for different routes
    def get_move_time(robot, from_loc, to_loc):
        # Define distances
        distances = {
            ("l1", "l3"): 5.0,  # r1 far from couch
            ("l2", "l3"): 2.0,  # r2 close to couch
            ("l3", "l1"): 5.0,
            ("l3", "l2"): 2.0,
            ("l1", "l2"): 3.0,
            ("l2", "l1"): 3.0,
            ("l1", "l4"): 7.0,
            ("l4", "l1"): 7.0,
            ("l2", "l4"): 4.0,
            ("l4", "l2"): 4.0,
            ("l3", "l4"): 3.0,
            ("l4", "l3"): 3.0,
        }
        return distances.get((from_loc, to_loc), 1.0)

    # Create operators
    operators = [
        construct_move_operator(get_move_time),
        construct_wait_operator(),
        construct_lift_couch_operator(lift_time=1.0),
        construct_move_couch_operator(move_time=3.0),
        construct_put_down_couch_operator(put_down_time=1.0)
    ]

    # Instantiate all actions
    all_actions = [act for op in operators
                   for act in op.instantiate(objects_by_type)]

    print(f"\nTotal actions instantiated: {len(all_actions)}")

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
    print(f"Usable actions: {len(usable_actions)}")

    # Run MCTS planner
    state = initial_state
    mcts = MCTSPlanner(usable_actions)

    print(f"\n{'='*60}")
    print("Starting MCTS Planning for Couch Carry Task")
    print(f"{'='*60}")
    print(f"Initial state (t={state.time}):")
    for fluent in sorted(str(f) for f in state.fluents):
        print(f"  {fluent}")

    max_steps = 20
    for step in range(max_steps):
        if goal.evaluate(state.fluents):
            print(f"\n{'='*60}")
            print("GOAL ACHIEVED!")
            print(f"{'='*60}")
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
            print("\nPlanner returned NONE - no valid action found")
            break

        # Execute action
        action = get_action_by_name(usable_actions, action_name)
        next_states = transition(state, action)
        state = next_states[0][0]  # Take first (deterministic) outcome

        print(f"\nStep {step + 1}: {action_name}")
        print(f"  Time: {state.time}")
        print("  Key fluents:")

        # Print relevant fluents
        for fluent_str in sorted(str(f) for f in state.fluents):
            if any(keyword in fluent_str for keyword in
                   ["at ", "free", "carrying", "waiting", "on-floor"]):
                print(f"    {fluent_str}")

    # Verify goal was achieved
    assert goal.evaluate(state.fluents), f"Goal not achieved after {max_steps} steps"
    print(f"\nFinal time: {state.time} seconds")

    # Verify both robots are free at the end
    assert F("free r1") in state.fluents, "r1 should be free at the end"
    assert F("free r2") in state.fluents, "r2 should be free at the end"

    # Verify couch is at l4 and on floor
    assert F("at couch1 l4") in state.fluents
    assert F("on-floor couch1") in state.fluents

    print("\nAll assertions passed!")

import pytest
from mrppddl.core import (
    Fluent,
    Operator,
    Effect,
    State,
    transition,
    preprocess_actions_for_relaxed_planning,
)
from mrppddl._bindings import ff_heuristic

F = Fluent


def construct_move_operator(move_cost: float = 5.0):
    """Construct a simple move operator with fixed cost."""
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
            Effect(
                time=move_cost,
                resulting_fluents={F("free ?r"), F("at ?r ?to")},
            ),
        ],
    )


def construct_pick_operator(pick_cost: float = 2.0):
    """Construct a pick operator with negative precondition (hand not full)."""
    return Operator(
        name="pick",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[
            F("at ?r ?loc"),
            F("at ?obj ?loc"),
            F("free ?r"),
            ~F("hand_full ?r"),  # Negative precondition
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?obj ?loc")}),
            Effect(
                time=pick_cost,
                resulting_fluents={
                    F("free ?r"),
                    F("holding ?r ?obj"),
                    F("hand_full ?r"),  # Hand becomes full after picking
                },
            ),
        ],
    )


def construct_place_operator(place_cost: float = 2.0):
    """Construct a place operator that empties the hand."""
    return Operator(
        name="place",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[
            F("at ?r ?loc"),
            F("holding ?r ?obj"),
            F("free ?r"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r"), ~F("holding ?r ?obj")}),
            Effect(
                time=place_cost,
                resulting_fluents={
                    F("free ?r"),
                    F("at ?obj ?loc"),
                    ~F("hand_full ?r"),  # Hand becomes empty after placing
                },
            ),
        ],
    )


def test_simple_move_and_pick():
    """Test a simple planning task: move to a location and pick up an object."""
    # Define operators
    move_op = construct_move_operator(move_cost=5.0)
    pick_op = construct_pick_operator(pick_cost=2.0)

    # Define objects
    objects_by_type = {
        "robot": ["r1"],
        "location": ["start", "target"],
        "object": ["box"],
    }

    # Instantiate all actions
    all_actions = []
    all_actions.extend(move_op.instantiate(objects_by_type))
    all_actions.extend(pick_op.instantiate(objects_by_type))

    # Initial state: robot at start, object at target location, hand not full
    initial_state = State(
        time=0,
        fluents={
            F("at r1 start"),
            F("free r1"),
            F("at box target"),
            # Note: hand_full is not in fluents, meaning ~F("hand_full r1") is satisfied
        }
    )

    # Goal: robot is holding the box
    goal_fluents = {F("holding r1 box")}

    def is_goal(state):
        return all(gf in state.fluents for gf in goal_fluents)

    # Test that goal is not satisfied initially
    assert not is_goal(initial_state)

    # Execute plan manually to verify it works
    state = initial_state

    # Step 1: Move from start to target (cost: 5.0)
    move_action = next(a for a in all_actions if a.name == "move r1 start target")
    state = transition(state, move_action)[0][0]
    assert F("at r1 target") in state.fluents
    assert state.time == 5.0

    # Step 2: Pick the box (cost: 2.0)
    pick_action = next(a for a in all_actions if a.name == "pick r1 target box")
    state = transition(state, pick_action)[0][0]
    assert F("holding r1 box") in state.fluents
    assert F("hand_full r1") in state.fluents  # Hand is now full
    assert state.time == 7.0

    # Verify goal is satisfied
    assert is_goal(state)


def test_ff_heuristic_move_and_pick():
    """Test that FF heuristic computes correct value for move-and-pick task.

    This test will verify that the relaxed planning graph properly handles
    negative preconditions by converting them to positive preconditions.
    """
    # Define operators
    move_op = construct_move_operator(move_cost=5.0)
    pick_op = construct_pick_operator(pick_cost=2.0)

    # Define objects
    objects_by_type = {
        "robot": ["r1"],
        "location": ["start", "target"],
        "object": ["box"],
    }

    # Instantiate all actions
    all_actions = []
    all_actions.extend(move_op.instantiate(objects_by_type))
    all_actions.extend(pick_op.instantiate(objects_by_type))

    # Initial state: robot at start, object at target location
    initial_state = State(
        time=0,
        fluents={
            F("at r1 start"),
            F("free r1"),
            F("at box target"),
            # hand_full is not in fluents, so ~F("hand_full r1") is true
        }
    )

    # Goal: robot is holding the box
    goal_fluents = {F("holding r1 box")}

    # Compute FF heuristic
    h_value = ff_heuristic(initial_state, goal_fluents, all_actions)

    # Expected cost: move (5.0) + pick (2.0) = 7.0
    # This assumes the heuristic properly handles the negative precondition ~F("hand_full")
    assert h_value == 7.0, f"Expected heuristic value 7.0, got {h_value}"


def test_ff_heuristic_already_at_goal():
    """Test FF heuristic when already at the target location."""
    # Define operators
    move_op = construct_move_operator(move_cost=5.0)
    pick_op = construct_pick_operator(pick_cost=2.0)

    # Define objects
    objects_by_type = {
        "robot": ["r1"],
        "location": ["start", "target"],
        "object": ["box"],
    }

    # Instantiate all actions
    all_actions = []
    all_actions.extend(move_op.instantiate(objects_by_type))
    all_actions.extend(pick_op.instantiate(objects_by_type))

    # Initial state: robot already at target location with box
    initial_state = State(
        time=0,
        fluents={
            F("at r1 target"),
            F("free r1"),
            F("at box target"),
            # hand_full is not in fluents
        }
    )

    # Goal: robot is holding the box
    goal_fluents = {F("holding r1 box")}

    # Compute FF heuristic
    h_value = ff_heuristic(initial_state, goal_fluents, all_actions)

    # Expected cost: only pick (2.0), no move needed
    assert h_value == 2.0, f"Expected heuristic value 2.0, got {h_value}"


def test_ff_heuristic_with_hand_full():
    """Test FF heuristic when hand is initially full - requires place then pick.

    This test specifically exposes issues with negative precondition handling.
    The pick operator requires ~F("hand_full"), which is initially violated.
    The robot must place the dummy object first to empty the hand.

    Uses preprocessing to convert negative preconditions to positive ones.
    """
    # Define operators (no move operator needed for this test)
    pick_op = construct_pick_operator(pick_cost=2.0)
    place_op = construct_place_operator(place_cost=2.0)

    # Define objects
    objects_by_type = {
        "robot": ["r1"],
        "location": ["target"],
        "object": ["dummy_obj", "box"],
    }

    # Instantiate all actions
    all_actions = []
    all_actions.extend(pick_op.instantiate(objects_by_type))
    all_actions.extend(place_op.instantiate(objects_by_type))

    # Initial state: robot at target, already holding dummy_obj (hand is full!)
    initial_state = State(
        time=0,
        fluents={
            F("at r1 target"),
            F("free r1"),
            F("holding r1 dummy_obj"),  # Already holding an object
            F("hand_full r1"),  # Hand is full - blocks picking!
            F("at box target"),  # Target box is at same location
        }
    )

    # Preprocess actions and state to convert negative preconditions to positive
    converted_actions, converted_state, mapping = preprocess_actions_for_relaxed_planning(
        all_actions, initial_state
    )

    # Goal: robot is holding the box
    goal_fluents = {F("holding r1 box")}

    # Compute FF heuristic with converted actions and state
    h_value = ff_heuristic(converted_state, goal_fluents, converted_actions)

    # Expected cost: place dummy_obj (2.0) + pick box (2.0) = 4.0
    # The heuristic MUST recognize that place is needed before pick
    # because pick has precondition ~F("hand_full r1") -> F("not-hand_full r1")
    assert h_value == 4.0, f"Expected heuristic value 4.0, got {h_value}"


def test_place_then_pick_execution():
    """Verify that the place-then-pick plan actually executes correctly."""
    # Define operators
    pick_op = construct_pick_operator(pick_cost=2.0)
    place_op = construct_place_operator(place_cost=2.0)

    # Define objects
    objects_by_type = {
        "robot": ["r1"],
        "location": ["target"],
        "object": ["dummy_obj", "box"],
    }

    # Instantiate all actions
    all_actions = []
    all_actions.extend(pick_op.instantiate(objects_by_type))
    all_actions.extend(place_op.instantiate(objects_by_type))

    # Initial state: robot holding dummy_obj, box at target
    initial_state = State(
        time=0,
        fluents={
            F("at r1 target"),
            F("free r1"),
            F("holding r1 dummy_obj"),
            F("hand_full r1"),
            F("at box target"),
        }
    )

    # Goal: robot is holding the box
    def is_goal(state):
        return F("holding r1 box") in state.fluents

    # Verify goal not initially satisfied
    assert not is_goal(initial_state)

    state = initial_state

    # Step 1: Place dummy_obj (cost: 2.0)
    place_action = next(a for a in all_actions if a.name == "place r1 target dummy_obj")
    state = transition(state, place_action)[0][0]
    assert F("at dummy_obj target") in state.fluents
    assert F("hand_full r1") not in state.fluents  # Hand is now empty
    assert F("holding r1 dummy_obj") not in state.fluents
    assert state.time == 2.0

    # Step 2: Pick box (cost: 2.0)
    pick_action = next(a for a in all_actions if a.name == "pick r1 target box")
    state = transition(state, pick_action)[0][0]
    assert F("holding r1 box") in state.fluents
    assert F("hand_full r1") in state.fluents  # Hand is full again
    assert state.time == 4.0

    # Verify goal is satisfied
    assert is_goal(state)

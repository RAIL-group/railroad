import pytest
from mrppddl.core import (
    Fluent,
    Operator,
    Effect,
    State,
    transition,
    preprocess_actions_for_relaxed_planning,
    extract_negative_preconditions,
    create_positive_fluent_mapping,
    convert_action_to_positive_preconditions,
    convert_action_effects,
)
from mrppddl._bindings import ff_heuristic, Action, GroundedEffect, GoalFn

F = Fluent


# ============================================================================
# Unit Tests for Preprocessing Functions
# ============================================================================


def test_convert_action_replaces_negative_precondition():
    """Test that negative preconditions are replaced with positive 'not-' versions."""
    # Manually create an action with a negative precondition
    preconditions = {
        F("at r1 location"),
        F("free r1"),
        ~F("hand_full r1"),  # Negative precondition
    }
    effects = [GroundedEffect(time=2.0, resulting_fluents={F("done")})]
    action = Action(preconditions, effects, name="test_action")

    # Verify the action has a negative precondition
    assert len(action._neg_precond_flipped) == 1
    assert F("hand_full r1") in action._neg_precond_flipped

    # Extract negative preconditions using our function
    neg_fluents = extract_negative_preconditions([action])
    assert F("hand_full r1") in neg_fluents

    # Create mapping for negative preconditions
    mapping = create_positive_fluent_mapping(neg_fluents)
    assert F("hand_full r1") in mapping

    # Convert the action
    converted_action = convert_action_to_positive_preconditions(action, mapping)

    # Verify negative precondition was replaced with mapped positive version
    assert len(converted_action._neg_precond_flipped) == 0
    assert mapping[F("hand_full r1")] in converted_action.preconditions
    assert ~F("hand_full r1") not in converted_action.preconditions

    # Verify positive preconditions are preserved
    assert F("at r1 location") in converted_action.preconditions
    assert F("free r1") in converted_action.preconditions


def test_effect_adds_positive_fluent():
    """Test that when F("P") is in effects, ~F("not-P") is also added."""
    # Create an action that adds F("P") in its effects
    preconditions = {F("at r1 location")}
    effects = [
        GroundedEffect(time=2.0, resulting_fluents={F("P")})  # Adds P
    ]
    action = Action(preconditions, effects, name="test_action")

    # Create mapping for P (simulate that ~F("P") appears as negative precondition somewhere)
    mapping = create_positive_fluent_mapping({F("P")})

    # Verify mapping was created correctly
    assert F("P") in mapping
    not_P = mapping[F("P")]  # This is F("not-P")

    # Convert the action's effects
    converted_action = convert_action_effects(action, mapping)

    # Expected: effects should now include both F("P") and ~F("not-P")
    # This maintains consistency: if P becomes true, not-P must become false
    effect_fluents = converted_action.effects[0].resulting_fluents
    assert F("P") in effect_fluents
    assert ~not_P in effect_fluents  # Should have ~F("not-P")


def test_effect_removes_positive_fluent():
    """Test that when ~F("P") is in effects, F("not-P") is also added."""
    # Create an action that removes F("P") (i.e., has ~F("P") in effects)
    preconditions = {F("at r1 location")}
    effects = [
        GroundedEffect(time=2.0, resulting_fluents={~F("P")})  # Removes P
    ]
    action = Action(preconditions, effects, name="test_action")

    # Create mapping for P
    mapping = create_positive_fluent_mapping({F("P")})

    # Verify mapping was created correctly
    assert F("P") in mapping
    not_P = mapping[F("P")]  # This is F("not-P")

    # Convert the action's effects
    converted_action = convert_action_effects(action, mapping)

    # Expected: effects should now include both ~F("P") and F("not-P")
    # This maintains consistency: if P becomes false, not-P must become true
    effect_fluents = converted_action.effects[0].resulting_fluents
    assert ~F("P") in effect_fluents
    assert not_P in effect_fluents  # Should have F("not-P")


def test_probabilistic_effect_conversion():
    """Test that probabilistic effects are correctly converted."""
    # Create an action with probabilistic effects (like a search action)
    preconditions = {F("at r1 location"), ~F("found obj")}

    # Create probabilistic effect: 60% chance of finding, 40% chance of not finding
    prob_effects = [
        (0.6, [  # Success branch
            GroundedEffect(time=0.0, resulting_fluents={F("found obj")}),
            GroundedEffect(time=2.0, resulting_fluents={F("holding r1 obj")})
        ]),
        (0.4, [  # Failure branch
            GroundedEffect(time=0.0, resulting_fluents={F("searched location")})
        ])
    ]

    effects = [
        GroundedEffect(
            time=5.0,
            resulting_fluents={F("at r1 location")},
            prob_effects=prob_effects
        )
    ]
    action = Action(preconditions, effects, name="search")

    # Create mapping for "found"
    mapping = create_positive_fluent_mapping({F("found obj")})
    not_found = mapping[F("found obj")]  # F("not-found obj")

    # Convert the action's effects
    converted_action = convert_action_effects(action, mapping)

    # Verify the main effect is converted
    main_effect = converted_action.effects[0]
    assert F("at r1 location") in main_effect.resulting_fluents

    # Verify probabilistic branches are preserved
    assert main_effect.is_probabilistic
    assert len(main_effect.prob_effects) == 2

    # Check success branch - should have added ~F("not-found obj")
    success_branch = main_effect.prob_effects[0]
    assert success_branch.prob == 0.6
    success_effects = success_branch.effects

    # First effect in success branch adds "found obj", should also add ~F("not-found obj")
    first_success_effect = success_effects[0]
    assert F("found obj") in first_success_effect.resulting_fluents
    assert ~not_found in first_success_effect.resulting_fluents  # Should have ~F("not-found obj")


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


def construct_search_operator(find_prob: float = 1.0, search_cost: float = 10.0):
    """Construct a search operator with probabilistic effects."""
    return Operator(
        name="search",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[
            F("at ?r ?loc"),
            F("free ?r"),
            ~F("revealed ?loc"),
            ~F("searched ?loc ?obj"),
            ~F("found ?obj")
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r"), F("lock-search ?loc")}),
            Effect(
                time=search_cost,
                resulting_fluents={
                    F("free ?r"),
                    F("searched ?loc ?obj"),
                    ~F("lock-search ?loc")
                },
                prob_effects=[
                    (find_prob, [Effect(time=0, resulting_fluents={F("found ?obj"), F("at ?obj ?loc")})]),
                    (1.0 - find_prob, [])
                ]
            )
        ]
    )


@pytest.mark.parametrize("find_prob", [1.0, 0.5, 0.2, 0.0])
def test_ff_heuristic_with_probabilistic_search(find_prob):
    """Test that FF heuristic handles probabilistic search actions correctly.

    This test verifies the fix for the bug where the heuristic only examined
    the first outcome of probabilistic actions, potentially missing the success
    case where an object is found. The heuristic should consider ALL possible
    outcomes in relaxed planning.

    Regression test for: heuristic returning infinity when goal requires
    finding an object via probabilistic search action.
    """
    # Define operators
    move_op = construct_move_operator(move_cost=5.0)
    search_op = construct_search_operator(find_prob=find_prob, search_cost=10.0)
    pick_op = construct_pick_operator(pick_cost=5.0)
    place_op = construct_place_operator(place_cost=5.0)

    # Define objects
    objects_by_type = {
        "robot": ["r1"],
        "location": ["start", "desk_4", "garbagecan_5"],
        "object": ["pencil_17"],
    }

    # Instantiate all actions
    all_actions = []
    all_actions.extend(move_op.instantiate(objects_by_type))
    all_actions.extend(search_op.instantiate(objects_by_type))
    all_actions.extend(pick_op.instantiate(objects_by_type))
    all_actions.extend(place_op.instantiate(objects_by_type))

    # Initial state: robot at start, object location unknown
    initial_state = State(
        time=0,
        fluents={
            F("at r1 start"),
            F("free r1"),
            # Note: revealed start means we can't search there
            # pencil_17 location is unknown - must search to find it
        }
    )

    # Goal: pencil at garbagecan_5
    goal_fluents = {F("at pencil_17 garbagecan_5")}

    # Compute FF heuristic
    h_value = ff_heuristic(initial_state, goal_fluents, all_actions)

    # The heuristic should return a finite value, not infinity
    # Before the fix, this would return inf because the heuristic only
    # looked at succs[0], which could be the failure branch of the search
    if find_prob > 0:
        assert h_value != float('inf'), (
            "Heuristic returned infinity! The fix for probabilistic outcomes "
            "may not be working correctly."
        )
    else:
        assert h_value == float('inf'), (
            "Heuristic should be infinite, since there is no solution"
        )


def test_convert_state_with_upcoming_effects():
    """Test that convert_state_to_positive_preconditions handles upcoming effects.

    This test verifies that when a state has upcoming effects, those effects
    are properly converted to maintain consistency with the negative-to-positive
    precondition mapping.
    """
    from mrppddl.core import convert_state_to_positive_preconditions

    # Hand-code a simple mapping
    neg_to_pos_mapping = {
        F("hand_full r1"): F("not-hand_full r1"),
        F("found obj"): F("not-found obj")
    }

    # Create a state with upcoming effects that include fluents in the mapping
    # Deterministic effect: adds hand_full (should also add ~not-hand_full)
    det_effect = GroundedEffect(
        time=2.0,
        resulting_fluents={F("hand_full r1"), F("holding r1 obj")}
    )

    # Probabilistic effect: branches that add/remove mapped fluents
    prob_effect = GroundedEffect(
        time=5.0,
        resulting_fluents={F("searched location")},
        prob_effects=[
            (0.6, [GroundedEffect(time=0.0, resulting_fluents={F("found obj")})]),
            (0.4, [GroundedEffect(time=0.0, resulting_fluents={F("nothing")})])
        ]
    )

    initial_state = State(
        time=0,
        fluents={F("at r1 start"), F("free r1")},
        upcoming_effects=[(2.0, det_effect), (5.0, prob_effect)]
    )

    # Convert the state
    converted_state = convert_state_to_positive_preconditions(initial_state, neg_to_pos_mapping)

    # Check that upcoming effects are converted
    assert len(converted_state.upcoming_effects) == 2

    # Check deterministic effect: should have ~F("not-hand_full r1") added
    det_time, det_converted = converted_state.upcoming_effects[0]
    assert det_time == 2.0
    assert F("hand_full r1") in det_converted.resulting_fluents
    assert ~F("not-hand_full r1") in det_converted.resulting_fluents
    assert F("holding r1 obj") in det_converted.resulting_fluents

    # Check probabilistic effect: should have ~F("not-found obj") in success branch
    prob_time, prob_converted = converted_state.upcoming_effects[1]
    assert prob_time == 5.0
    assert F("searched location") in prob_converted.resulting_fluents
    assert prob_converted.is_probabilistic

    success_branch = prob_converted.prob_effects[0]
    assert success_branch.prob == 0.6
    success_effect = success_branch.effects[0]
    assert F("found obj") in success_effect.resulting_fluents
    assert ~F("not-found obj") in success_effect.resulting_fluents


def test_goal_fn_goal_count():
    """Test GoalFn's goal_count method counts how many goal fluents are achieved."""
    # Create a goal function with 3 goal fluents
    goal_fluents = {
        F("at robot kitchen"),
        F("holding robot cup"),
        F("clean cup")
    }

    goal_fn = GoalFn(goal_fluents)

    # Test case 1: No goals achieved
    active_fluents_0 = {F("at robot bedroom"), F("free robot")}
    assert goal_fn.goal_count(active_fluents_0) == 0
    assert not goal_fn(active_fluents_0)

    # Test case 2: 1 goal achieved
    active_fluents_1 = {F("at robot kitchen"), F("free robot")}
    assert goal_fn.goal_count(active_fluents_1) == 1
    assert not goal_fn(active_fluents_1)

    # Test case 3: 2 goals achieved
    active_fluents_2 = {F("at robot kitchen"), F("holding robot cup"), F("dirty cup")}
    assert goal_fn.goal_count(active_fluents_2) == 2
    assert not goal_fn(active_fluents_2)

    # Test case 4: All 3 goals achieved
    active_fluents_3 = {F("at robot kitchen"), F("holding robot cup"), F("clean cup")}
    assert goal_fn.goal_count(active_fluents_3) == 3
    assert goal_fn(active_fluents_3)

    # Test case 5: All goals achieved plus extra fluents
    active_fluents_4 = {
        F("at robot kitchen"),
        F("holding robot cup"),
        F("clean cup"),
        F("visited bedroom"),
        F("searched drawer")
    }
    assert goal_fn.goal_count(active_fluents_4) == 3
    assert goal_fn(active_fluents_4)

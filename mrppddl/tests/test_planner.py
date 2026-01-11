from mrppddl.core import Fluent, State, transition, get_action_by_name
from mrppddl.helper import construct_move_visited_operator
from mrppddl.helper import construct_search_operator
from mrppddl.planner import MCTSPlanner, get_usable_actions

import pytest
import random

F = Fluent


def test_pruning_unavailable_actions():
    initial_state = State(time=0, fluents=set())
    objects_by_type = {
        "robot": ["r1", "r2", "r3"],
        "location": ["start", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
    }
    random.seed(8616)
    move_op = construct_move_visited_operator(lambda *args: 5.0 + random.random())
    all_actions = move_op.instantiate(objects_by_type)

    initial_state = State(time=0, fluents={F("at r1 start"), F("free r1"),
                                        F("visited start")}, )
    num_actions_before = len(all_actions)
    all_actions = get_usable_actions(initial_state, all_actions)

    assert len(all_actions) < num_actions_before

    

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
            print("Goal found!")
            break
        action_name = mcts(state, goal, 2000, c=10)
        if action_name == "NONE":
            break
        action = get_action_by_name(all_actions, action_name)

        state = transition(state, action)[0][0]
        print(action_name, state, goal.evaluate(state.fluents))
    assert goal.evaluate(state.fluents)


@pytest.mark.parametrize(("roomA_prob", "num_robots"), [
    (1.0, 1),
    (0.8, 1),
    (0.6, 1),
    (1.0, 2),
    (0.8, 2),
    (0.6, 2),
])
def test_mcts_search_picks_more_likely_location(roomA_prob, num_robots):
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
    search_actions = construct_search_operator(
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
    num_planning_attempts = 20
    for _ in range(num_planning_attempts):
        action = mcts(initial_state, goal, max_iterations=10000, c=100)
        selected_actions.append(action)

    # Count how many selected actions mention roomA
    roomA_count = sum("roomA" in str(action) for action in selected_actions)

    # We expect roomA to appear in at least 80% of planning attempts
    assert (
        roomA_count >= 0.8 * num_planning_attempts
    ), f"Expected roomA in at least 80% actions, got {roomA_count}/{num_planning_attempts} for roomA_prob={roomA_prob}"


def test_basic_planning():
    """Test basic planning functionality."""
    # Simple test setup
    objects_by_type = {
        "robot": ["r1"],
        "location": ["start", "a", "b"],
    }
    move_op = construct_move_visited_operator(lambda *args: 5.0)
    all_actions = move_op.instantiate(objects_by_type)

    initial_state = State(
        time=0,
        fluents={F("at r1 start"), F("free r1"), F("visited start")}
    )
    goal = F("visited a")

    # Create planner
    mcts = MCTSPlanner(all_actions)

    # Run planner
    action_name = mcts(initial_state, goal, max_iterations=100, c=1.414)

    # Verify we got a valid result (either an action name or "NONE")
    assert isinstance(action_name, str)
    assert len(action_name) > 0

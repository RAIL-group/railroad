import pytest
from interruption.utilities import (
    RandomVariableType,
    _check_num_rooms,
    _check_scene_room_types,
    filter_procthor_scenes,
    get_action_cost,
    get_augmented_task_dist,
    get_next_state,
    get_task_arrival_prob,
)
from railroad.core import Fluent, State, get_next_actions
from railroad.operators.core import construct_move_operator, construct_pick_operator


@pytest.mark.parametrize("action_cost", [1, 3, 5])
def test_get_action_cost_pick(action_cost):
    pick_op = construct_pick_operator(action_cost)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"pantry", "refrigerator"},
        "object": {"turkey", "bread"}
    }

    pick_actions = pick_op.instantiate(objects_by_type)

    for action in pick_actions:
        assert get_action_cost(action) == action_cost

def test_get_next_state():
    move_op = construct_move_operator(4)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
    }

    move_actions = move_op.instantiate(objects_by_type)
    assert len(move_actions) == 2

    initial_state = State(
        time=0,
        fluents={Fluent("at robot1 kitchen"), Fluent("free robot1")}
    )

    applicable_actions = get_next_actions(initial_state, move_actions)
    assert len(applicable_actions) == 1
    next_state, interruption_prob = get_next_state(initial_state, applicable_actions[0])
    assert next_state.time == 4
    assert next_state.fluents == {Fluent("at robot1 living_room"), Fluent("free robot1")}
    assert interruption_prob == 0


@pytest.mark.parametrize(
    argnames="rv_type, arrival_prob, time_between_arrivals, action_time, sol",
    argvalues=[
        (RandomVariableType.DISCRETE, 0.1, 100, 1, 0.1),
        (RandomVariableType.DISCRETE, 0.1, -1, 4, 0.1),
        (RandomVariableType.DISCRETE, 0.1, -1, 100, 0.1),
        (RandomVariableType.CONTINUOUS, -1, 5, 10, 0.86466472),
        (RandomVariableType.CONTINUOUS, -1, 30, 30, 0.63212056),
        (RandomVariableType.CONTINUOUS, -1, 60, 30, 0.39346934),
    ]
)
def test_get_task_arrival_prob(rv_type, arrival_prob, time_between_arrivals, action_time, sol):
    arrival_prob = get_task_arrival_prob(rv_type, arrival_prob, time_between_arrivals, action_time)
    assert arrival_prob == pytest.approx(sol)


@pytest.mark.parametrize(
    argnames="rooms, num_rooms, sol",
    argvalues=[
        ([{"room1": 1}, {"room2": 2}], {1}, False),
        ([{"room1": 1}, {"room2": 2}], {2}, True),
        ([{"room1": 1}, {"room2": 2}], {1, 2}, True),
        ([{"room1": 1}, {"room2": 2}], None, True),
    ]
)
def test_check_num_rooms(rooms, num_rooms, sol):
    assert _check_num_rooms(rooms, num_rooms) == sol


@pytest.mark.parametrize(
    argnames="rooms, room_types, sol",
    argvalues=[
        ([{"roomType": "Kitchen"}], {"Bedroom"}, False),
        ([{"roomType": "Bedroom"}], {"Bedroom"}, True),
        ([{"roomType": "Kitchen"}, {"roomType": "Bedroom"}], {"Kitchen"}, True),
        ([{"roomType": "Kitchen"}, {"roomType": "Bedroom"}], None, True),
    ]
)
def test_check_scene_room_types(rooms, room_types, sol):
    assert _check_scene_room_types(rooms, room_types) == sol


@pytest.mark.parametrize(
    argnames="num_rooms, room_types, locations, objects, sol",
    argvalues=[
        (None, None, None, None, 10000),
        ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, None, None, None, 10000),
        ({1}, {"Kitchen"}, None, None, 482),
        ({1}, {"Kitchen"}, {"sidetable"}, None, 33),
        ({1}, {"Kitchen"}, None, {"coffeemachine"}, 120),
    ]
)
def test_filter_procthor_scenes(num_rooms, room_types, locations, objects, sol):
    assert len(filter_procthor_scenes(num_rooms, room_types, locations, objects)) == sol


@pytest.mark.parametrize(
    argnames="current_task, task_dist, sol",
    argvalues=[
        (
            Fluent("at spoon bathroom"),
            ([Fluent("at cup bathroom")], [1.0]),
            ([Fluent("at spoon bathroom") & Fluent("at cup bathroom")], [1.0])
        ),
        (
            Fluent("at spoon bathroom"),
            ([Fluent("at cup bathroom"), Fluent("at knife bathroom")], [0.5, 0.5]),
            (
                [
                    Fluent("at spoon bathroom") & Fluent("at cup bathroom"),
                    Fluent("at spoon bathroom") & Fluent("at knife bathroom")
                ],
                [0.5, 0.5]
            )
        ),
    ]
)
def test_get_augmented_task_dist(current_task, task_dist, sol):
    assert get_augmented_task_dist(current_task, task_dist) == sol

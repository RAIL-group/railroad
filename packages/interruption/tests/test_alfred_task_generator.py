from interruption.alfred_task_generator import get_task_list
import pytest


@pytest.mark.parametrize(
        'objects, locations',
        [(
            [
                'knife', 'egg', 'egg', 'tomato', 'tomato', 
                'peppershaker', 'spoon', 'apple', 'spraybottle', 
                'tomato', 'pencil', 'potato', 'tomato', 
                'pen', 'papertowelroll', 'pan'
            ],
            [
                'fridge', 'stool', 'stool', 'start', 
                'garbagecan', 'countertop', 'shelvingunit'
            ]
        )]
)
def test_get_task_list_ordering(objects, locations):
    task_list = get_task_list(locations, objects)
    for _ in range(1000):
        assert task_list == get_task_list(locations, objects)

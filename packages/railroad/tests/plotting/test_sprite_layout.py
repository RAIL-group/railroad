import math

import pytest

from railroad.plotting.sprites import assign_slots, fan_offset, ring_radius


def test_fan_offsets_are_distinct_and_keep_the_radius():
    assert fan_offset(0, 1, 6) == (0, -6)
    offsets = [fan_offset(index, 3, 6) for index in range(3)]
    assert len({(round(x, 6), round(y, 6)) for x, y in offsets}) == 3
    assert [math.hypot(x, y) for x, y in offsets] == pytest.approx([6] * 3)


def test_slots_are_stable_across_input_order_and_locations():
    first = {"mug": ["shelf", "counter"], "apple": ["shelf"], "pan": ["sink"]}
    slots = assign_slots(first)
    assert slots == assign_slots(dict(reversed(first.items())))
    assert slots[("apple", "shelf")] == (0, 2)
    assert slots[("mug", "shelf")] == (1, 2)
    assert slots[("mug", "counter")] == (0, 1)


def test_radius_uses_scene_resolution_and_expands_for_crowds():
    assert ring_radius(1, 0.05) == pytest.approx(6)
    assert ring_radius(1, 0.25) == pytest.approx(1.2)
    assert ring_radius(1, None) == pytest.approx(1)
    assert ring_radius(20, 0.05) > ring_radius(3, 0.05)

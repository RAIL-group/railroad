"""Fan-out of sprites that share a location.

ProcTHOR snaps every object to its container's cell, so without a fan the
sprites land exactly on top of one another.
"""

import math

import pytest

from railroad.dashboard._sprites import assign_slots, fan_offset, ring_radius


def test_a_lone_sprite_sits_above_the_cell():
    """Leaving the location marker and its label readable underneath."""
    assert fan_offset(0, 1, 6.0) == (0.0, -6.0)


def test_group_members_get_distinct_offsets():
    offsets = [fan_offset(i, 3, 6.0) for i in range(3)]
    assert len({(round(x, 6), round(y, 6)) for x, y in offsets}) == 3
    for x, y in offsets:
        assert math.hypot(x, y) == pytest.approx(6.0)


def test_slots_are_independent_of_input_order():
    forwards = assign_slots({"mug": ["shelf"], "apple": ["shelf"], "pan": ["sink"]})
    backwards = assign_slots({"pan": ["sink"], "apple": ["shelf"], "mug": ["shelf"]})
    assert forwards == backwards
    assert forwards[("apple", "shelf")] == (0, 2)
    assert forwards[("mug", "shelf")] == (1, 2)
    assert forwards[("pan", "sink")] == (0, 1)


def test_an_object_resting_at_two_locations_gets_a_slot_at_each():
    slots = assign_slots({"mug": ["shelf", "counter"], "apple": ["shelf"]})
    assert slots[("mug", "shelf")] == (1, 2)
    assert slots[("mug", "counter")] == (0, 1)


def test_radius_follows_the_scene_resolution():
    """The fan is sized in metres, so it looks the same on any map."""
    fine = ring_radius(1, 0.05)
    coarse = ring_radius(1, 0.25)
    assert fine == pytest.approx(6.0)
    assert coarse == pytest.approx(1.2)
    assert ring_radius(1, None) == pytest.approx(1.0)


def test_a_crowded_ring_grows_to_keep_sprites_apart():
    assert ring_radius(20, 0.05) > ring_radius(3, 0.05)
    assert ring_radius(3, 0.05) == pytest.approx(ring_radius(1, 0.05))

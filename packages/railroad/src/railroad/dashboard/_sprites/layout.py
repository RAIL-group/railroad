"""Deterministic placement of several sprites sharing one location.

ProcTHOR snaps every object to its container's grid cell, so co-located objects
are *exactly* coincident and one would hide the others entirely. Slots are
assigned once from the whole timeline rather than per frame: recomputing group
membership as objects are picked up would make the survivors jump.
"""

from __future__ import annotations

import math

RING_RADIUS_M = 0.30
"""Fan-out radius in metres, converted through the scene's cell resolution.

Sprites are sized in points and the map in cells, so a radius fixed in cells
would look enormous on a small map and vanish on a large one. Metres is the one
unit both ends agree on, and the scene already reports it.
"""

RING_RADIUS_CELLS_FALLBACK = 1.0
"""Used when the scene does not report a resolution (symbolic environments)."""

SPRITE_DIAMETER_M = 0.25
"""Nominal sprite width, in the same units as the radius.

Only used to keep a crowded ring from overlapping itself; the sprite's real size
is set in points, which is deliberately not a quantity this module can see.
"""


def assign_slots(
    rest_locations: dict[str, list[str]],
) -> dict[tuple[str, str], tuple[int, int]]:
    """Map ``(object, location)`` to ``(slot, group_size)``.

    Args:
        rest_locations: object name -> every location it ever rests at.

    Slots come from the sorted set of objects that *ever* rest at a location, so
    they do not depend on dict ordering and an object leaving simply vacates its
    slot rather than reshuffling its neighbours.
    """
    members: dict[str, set[str]] = {}
    for obj, locations in rest_locations.items():
        for loc in locations:
            members.setdefault(loc, set()).add(obj)

    slots: dict[tuple[str, str], tuple[int, int]] = {}
    for loc, objs in members.items():
        ordered = sorted(objs)
        for index, obj in enumerate(ordered):
            slots[(obj, loc)] = (index, len(ordered))
    return slots


def ring_radius(group_size: int, resolution: float | None) -> float:
    """Fan-out radius in grid cells for a group of *group_size* sprites."""
    if resolution:
        radius = RING_RADIUS_M / resolution
    else:
        radius = RING_RADIUS_CELLS_FALLBACK
    if group_size > 1:
        # Keep arc spacing at least one sprite wide, so a crowded location
        # spreads rather than stacking. Rarely binding once the selection is
        # restricted to objects the plan actually touches.
        diameter = SPRITE_DIAMETER_M / resolution if resolution else 1.0
        radius = max(radius, group_size * diameter / (2 * math.pi))
    return radius


def fan_offset(slot: int, group_size: int, radius: float) -> tuple[float, float]:
    """Offset from a location's cell for the *slot*-th of *group_size* sprites.

    A lone sprite sits directly above the cell rather than on it, so the black
    location marker and its label stay readable underneath. y increases
    downwards in this frame, so "above" is negative.
    """
    if group_size <= 1:
        return (0.0, -radius)
    theta = -math.pi / 2 + 2 * math.pi * slot / group_size
    return (radius * math.cos(theta), radius * math.sin(theta))

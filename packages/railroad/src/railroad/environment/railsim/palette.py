"""Scene color palette: albedos lifted verbatim from the Unity materials.

Colors are sRGB triples (the shader decodes to linear). A palette is a
plain dict so experiments can override any subset of entries -- or add new
ones for future object types -- without touching the scene code:

    map_data, start, goal = make_office(seed=2005,
                                        palette={'floor': (0.8, 0.1, 0.1)})
"""

from __future__ import annotations

from typing import Mapping

Color = tuple[float, float, float]
Palette = dict[str, Color]

DEFAULT_PALETTE: Palette = {
    'wall': (0.582, 0.5791, 0.5763),           # wall-base.mat
    'wall_hallway': (0.6902, 0.7346, 0.9010),  # wall-hallway.mat (light blue)
    'wall_room': (1.0, 0.8977, 0.4481),        # wall-classroom.mat (warm gold)
    'floor': (0.7083, 0.8007, 0.8679),         # floor.mat (light blue-gray)
    'ceiling': (0.8491, 0.8040, 0.6368),       # ceiling.mat (warm cream)
    'breadcrumb': (0.3587, 0.9057, 0.2520),    # breadcrumb-green.mat
    'table': (0.3585, 0.2105, 0.0592),         # table.mat (brown)
    'light_fixture': (0.217, 0.217, 0.217),    # lightFixture.mat
}


def resolve_palette(overrides: Mapping[str, Color] | None = None) -> Palette:
    """The default palette updated with ``overrides`` (extra keys allowed,
    so callers can introduce colors for new object types)."""
    palette = dict(DEFAULT_PALETTE)
    if overrides:
        palette.update(overrides)
    return palette

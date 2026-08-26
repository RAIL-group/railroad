"""Search pulses: an expanding ring showing that a robot is sensing.

A search is the one action a robot spends real time on while standing still,
so on the map it is indistinguishable from waiting. The ring is that missing
signal: it leaves the robot small and opaque, widens to `PULSE_RADIUS_M` and
fades out as the search runs, so its size reads as how far through the action
the robot is.

Like the object glyphs next door, the whole thing comes out of the snapshots
the dashboard already keeps -- no new capture hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

Coord = tuple[float, float]

PULSE_RADIUS_M = 1.0
"""How far a ring travels from the robot by the end of the search, in metres."""

PULSE_MIN_FRACTION = 0.12
"""Radius the ring starts at, as a fraction of its final one.

Starting from exactly zero spends the opening frames on a ring too small to
resolve, which reads as the pulse arriving late rather than as it growing.
"""

PULSE_LINEWIDTH_PT = 1.6
"""Stroke width of the ring, in points."""

SEARCH_ACTION_PREFIX = "search"
"""Prefix an action's head token must have to be drawn as a search.

A prefix rather than an equality so `search-frontier` counts alongside
`search`; keying on the name is the same convention the rest of the
object-search machinery uses for `searched`, `found` and `free`.
"""


@dataclass(frozen=True)
class SearchWindow:
    """When one robot was busy searching."""

    robot: str
    start: float
    end: float


def _holds_free(fluents: Iterable[Any], robot: str) -> bool:
    for fluent in fluents:
        args: Sequence[str] = getattr(fluent, "args", ())
        if getattr(fluent, "name", None) == "free" and args and args[0] == robot:
            return True
    return False


def search_windows(
    *,
    actions_taken: Iterable[tuple[str, float]],
    history: Iterable[dict],
    known_robots: set[str],
    horizon: float,
) -> list[SearchWindow]:
    """The intervals each robot spent searching, from the plan's own record.

    A search dispatch takes the robot's `free` away and its completion gives it
    back, so the window is bounded by the next snapshot in which the robot is
    free again. That is exact rather than approximate: `act()` returns whenever
    *a* robot frees, so the search's own completion is always a snapshot -- and
    nothing can free the robot earlier, because it is busy. Keying on `free`
    rather than on `searched` also covers `search-frontier`, which records its
    result under a different predicate.

    A search still in flight when the plan ended has no such snapshot, and runs
    to *horizon*: the ring is then cut short with the video, which is what
    happened to the search.
    """
    entries = sorted(
        (float(entry["time"]), entry.get("fluents", ()))
        for entry in history
    )
    windows: list[SearchWindow] = []
    for action, dispatched in actions_taken:
        parts = action.split()
        if not parts or not parts[0].startswith(SEARCH_ACTION_PREFIX):
            continue
        robot = next((part for part in parts[1:] if part in known_robots), None)
        if robot is None:
            continue
        start = float(dispatched)
        end = next(
            (
                time
                for time, fluents in entries
                if time > start and _holds_free(fluents, robot)
            ),
            float(horizon),
        )
        if end > start:
            windows.append(SearchWindow(robot, start, end))
    return windows


def by_robot(windows: Iterable[SearchWindow]) -> dict[str, list[SearchWindow]]:
    grouped: dict[str, list[SearchWindow]] = {}
    for window in windows:
        grouped.setdefault(window.robot, []).append(window)
    return grouped


def sample(
    windows: Iterable[SearchWindow], times: Any, *, radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ring radius and opacity at each of *times*, for one robot's windows.

    Both are zero outside every window, which is what hides the artist. A
    robot cannot search twice at once -- it is not free in between -- so the
    windows never overlap and the order they are applied in does not matter.
    """
    query = np.asarray(times, dtype=float).reshape(-1)
    radii = np.zeros_like(query)
    alpha = np.zeros_like(query)
    for window in windows:
        span = window.end - window.start
        if span <= 0:
            continue
        live = (query >= window.start) & (query < window.end)
        progress = (query[live] - window.start) / span
        radii[live] = radius * (
            PULSE_MIN_FRACTION + (1.0 - PULSE_MIN_FRACTION) * progress
        )
        alpha[live] = 1.0 - progress
    return radii, alpha


def pulse_radius(ax: Any, resolution: float | None) -> float:
    """`PULSE_RADIUS_M` in the data units the axes are drawn in.

    Plots are drawn in grid cells, so the metre figure only means something
    where the scene reports a cell size. Without one -- a symbolic domain laid
    out in whatever coordinates its locations were given -- fall back to the
    screen-derived length the sprite fan already uses, so the ring stays
    legible at a scale no metre constant can predict.
    """
    if resolution:
        return PULSE_RADIUS_M / float(resolution)
    from .sprites import ring_radius, sprite_extent

    return ring_radius(1, sprite_extent(ax))


def make_ring(ax: Any, center: Coord, *, zorder: float, animated: bool = False) -> Any:
    """A white, unfilled ring parked at *center*, hidden until a search runs.

    Parked on the robot rather than at the origin because `add_patch` folds the
    patch into the axes' data limits, and a stray point at (0, 0) would drag
    the view of a symbolic plot out to meet it.
    """
    from matplotlib.patches import Circle

    ring = Circle(
        (float(center[0]), float(center[1])), 0.0,
        fill=False, edgecolor="white", linewidth=PULSE_LINEWIDTH_PT,
        zorder=zorder, visible=False,
    )
    ring.set_animated(animated)
    ax.add_patch(ring)
    return ring


def update_ring(ring: Any, center: Any, radius: float, alpha: float) -> None:
    ring.set_center((float(center[0]), float(center[1])))
    ring.set_radius(float(radius))
    ring.set_alpha(float(alpha))


__all__ = [
    "PULSE_RADIUS_M",
    "SearchWindow",
    "by_robot",
    "make_ring",
    "pulse_radius",
    "sample",
    "search_windows",
    "update_ring",
]

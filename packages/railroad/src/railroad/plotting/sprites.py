"""Object sprites for trajectory plots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np

from .emoji import GlyphProvider, get_glyph_provider

Coord = tuple[float, float]
RING_RADIUS_M = 0.30
SPRITE_DIAMETER_M = 0.25
SPRITE_POINTS = 18.0


@dataclass(frozen=True)
class Anchor:
    time: float
    kind: Literal["rest", "ride"]
    xy: Coord | None = None
    loc: str | None = None
    robot: str | None = None

    @property
    def target(self) -> tuple[str, str]:
        return self.kind, self.loc or self.robot or ""


@dataclass(frozen=True)
class ObjectTimeline:
    name: str
    found_time: float
    anchors: tuple[Anchor, ...]

    @property
    def rest_locations(self) -> list[str]:
        return [a.loc for a in self.anchors if a.kind == "rest" and a.loc]


def _args(fluent: Any) -> Sequence[str]:
    return getattr(fluent, "args", ())


def select_objects(
    *,
    goal_fluents: Iterable[Any],
    actions_taken: Iterable[tuple[str, float]],
    history: Iterable[dict],
    entity_positions: dict[str, list],
    known_robots: set[str],
    objects_by_type: dict[str, set[str]] | None = None,
    glyph_objects: Iterable[str] | None = None,
) -> set[str]:
    tracked = set(entity_positions) - known_robots
    if glyph_objects is not None:
        return set(glyph_objects) & tracked

    candidates = {arg for fluent in goal_fluents for arg in _args(fluent)}
    candidates.update(
        arg for action, _time in actions_taken for arg in action.split()[1:]
    )
    candidates.update(
        args[1]
        for entry in history
        for fluent in entry.get("fluents", ())
        if getattr(fluent, "name", None) == "holding"
        for args in [_args(fluent)]
        if len(args) >= 2
    )
    if objects_by_type and "object" in objects_by_type:
        candidates &= objects_by_type["object"]
    return candidates & tracked


def _collapse(anchors: list[Anchor]) -> tuple[Anchor, ...]:
    collapsed: list[Anchor] = []
    start = 0
    while start < len(anchors):
        end = start
        while (
            end + 1 < len(anchors) and anchors[end + 1].target == anchors[start].target
        ):
            end += 1
        collapsed.append(anchors[start])
        if end > start:
            collapsed.append(anchors[end])
        start = end + 1
    return tuple(collapsed)


def build_timelines(
    *,
    history: Sequence[dict],
    entity_positions: dict[str, list],
    selected: Iterable[str],
    env_coords: dict[str, Coord],
) -> dict[str, ObjectTimeline]:
    timelines = {}
    for obj in sorted(selected):
        positions = entity_positions.get(obj, [])
        stored = {loc: xy for _time, loc, xy in positions if xy is not None}
        anchors: list[Anchor] = []
        found_time = first_at_time = None

        for entry in history:
            time = float(entry["time"])
            at_loc = holder = None
            for fluent in entry.get("fluents", ()):
                name, args = getattr(fluent, "name", None), _args(fluent)
                if name == "at" and len(args) >= 2 and args[0] == obj:
                    at_loc = args[1]
                elif name == "holding" and len(args) >= 2 and args[1] == obj:
                    holder = args[0]
                elif name == "found" and args and args[0] == obj and found_time is None:
                    found_time = time
            if at_loc is not None:
                xy = stored.get(at_loc) or env_coords.get(at_loc)
                if xy is not None:
                    first_at_time = time if first_at_time is None else first_at_time
                    anchors.append(Anchor(time, "rest", xy=xy, loc=at_loc))
            elif holder is not None:
                anchors.append(Anchor(time, "ride", robot=holder))

        last_time = anchors[-1].time if anchors else float("-inf")
        for time, loc, xy in positions:
            resolved = xy or env_coords.get(loc)
            if time > last_time and resolved is not None:
                anchors.append(Anchor(float(time), "rest", xy=resolved, loc=loc))
        if not anchors:
            continue
        anchors.sort(key=lambda anchor: anchor.time)
        found_time = found_time if found_time is not None else first_at_time
        timelines[obj] = ObjectTimeline(
            obj,
            found_time if found_time is not None else anchors[0].time,
            _collapse(anchors),
        )
    return timelines


def sample(
    timeline: ObjectTimeline,
    times: Any,
    robot_xy: dict[str, Any],
    *,
    fade: float,
    offset_for: Callable[[Anchor], Coord] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(times, dtype=float).reshape(-1)
    anchors = timeline.anchors
    fallback = next(
        (anchor.xy for anchor in anchors if anchor.xy is not None), (0.0, 0.0)
    )
    tracks = np.broadcast_to(fallback, (len(anchors), len(query), 2)).copy()
    for index, anchor in enumerate(anchors):
        source = (
            anchor.xy if anchor.kind == "rest" else robot_xy.get(anchor.robot or "")
        )
        if source is not None:
            tracks[index] = np.asarray(source, dtype=float)
    offsets = np.asarray(
        [offset_for(anchor) if offset_for else (0.0, 0.0) for anchor in anchors]
    )
    tracks += offsets[:, None, :]

    anchor_times = np.fromiter((a.time for a in anchors), dtype=float)
    lower = np.clip(
        np.searchsorted(anchor_times, query, side="right") - 1, 0, len(anchors) - 1
    )
    upper = np.minimum(lower + 1, len(anchors) - 1)
    rows = np.arange(len(query))
    position = tracks[lower, rows]

    span = anchor_times[upper] - anchor_times[lower]
    with np.errstate(divide="ignore", invalid="ignore"):
        progress = np.where(span > 0, (query - anchor_times[lower]) / span, 0.0)
    progress = np.clip(progress, 0.0, 1.0)
    same_target = np.fromiter(
        (a.target == b.target for a, b in zip(anchors, anchors[1:] + anchors[-1:])),
        dtype=bool,
    )
    blending = (upper != lower) & ~same_target[lower]
    if blending.any():
        weight = (progress[blending] ** 2 * (3 - 2 * progress[blending]))[:, None]
        start = tracks[lower[blending], rows[blending]]
        end = tracks[upper[blending], rows[blending]]
        position[blending] = start * (1 - weight) + end * weight

    alpha = (
        np.clip((query - timeline.found_time) / fade, 0.0, 1.0)
        if fade > 0
        else (query >= timeline.found_time).astype(float)
    )
    return position, alpha


def assign_slots(
    rest_locations: dict[str, list[str]],
) -> dict[tuple[str, str], tuple[int, int]]:
    members: dict[str, set[str]] = {}
    for obj, locations in rest_locations.items():
        for location in locations:
            members.setdefault(location, set()).add(obj)
    slots: dict[tuple[str, str], tuple[int, int]] = {}
    for location, objects in members.items():
        ordered = sorted(objects)
        slots.update(
            {
                (obj, location): (index, len(ordered))
                for index, obj in enumerate(ordered)
            }
        )
    return slots


def ring_radius(group_size: int, resolution: float | None) -> float:
    radius = RING_RADIUS_M / resolution if resolution else 1.0
    if group_size > 1:
        diameter = SPRITE_DIAMETER_M / resolution if resolution else 1.0
        radius = max(radius, group_size * diameter / (2 * math.pi))
    return radius


def fan_offset(slot: int, group_size: int, radius: float) -> Coord:
    if group_size <= 1:
        return 0.0, -radius
    angle = -math.pi / 2 + 2 * math.pi * slot / group_size
    return radius * math.cos(angle), radius * math.sin(angle)


def make_sprite(
    ax: Any,
    rgba: Any,
    xy: Coord,
    *,
    zorder: float,
    size_points: float = SPRITE_POINTS,
    animated: bool = False,
) -> tuple[Any, Any]:
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    image = OffsetImage(rgba, zoom=size_points / rgba.shape[0], dpi_cor=True)
    box = AnnotationBbox(
        image, xy, frameon=False, pad=0.0, zorder=zorder, annotation_clip=True
    )
    box.set_animated(animated)
    ax.add_artist(box)
    return box, image


def update_sprite(box: Any, image: Any, rgba: Any, xy: Coord, alpha: float) -> None:
    box.xy = box.xybox = (float(xy[0]), float(xy[1]))
    if alpha >= 1.0:
        image.set_data(rgba)
    else:
        faded = rgba.astype(float) / 255.0
        faded[..., 3] *= alpha
        image.set_data(faded)


__all__ = [
    "Anchor",
    "GlyphProvider",
    "ObjectTimeline",
    "assign_slots",
    "build_timelines",
    "fan_offset",
    "get_glyph_provider",
    "make_sprite",
    "ring_radius",
    "sample",
    "select_objects",
    "update_sprite",
]

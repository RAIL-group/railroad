"""When and where each plan-relevant object should be drawn.

The dashboard already snapshots the whole fluent set once per planner step. That
is enough to reconstruct an object's whole story -- found, resting, carried,
put down -- because ``Environment.act`` returns when a robot frees, so the
interesting transitions always land on a snapshot boundary.

The one subtlety is that ``pick`` deletes ``at obj L`` at dispatch and only adds
``holding r obj`` when it completes, and ``place`` mirrors it. Across those
windows the object is in *neither* fluent, so "no fluent" must never be read as
"invisible" -- it is exactly the span the sprite should be travelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np

Coord = tuple[float, float]


@dataclass(frozen=True)
class Anchor:
    """Somewhere an object is pinned, from ``time`` until the next anchor."""

    time: float
    kind: Literal["rest", "ride"]
    xy: Coord | None = None
    """Grid cell of the location, for ``rest`` anchors."""
    loc: str | None = None
    """Location name, for ``rest`` anchors -- the key slot assignment uses."""
    robot: str | None = None
    """Carrier, for ``ride`` anchors."""

    @property
    def target(self) -> tuple[str, str]:
        """What the anchor pins to, for collapsing runs of identical anchors."""
        return (self.kind, self.loc or self.robot or "")


@dataclass(frozen=True)
class ObjectTimeline:
    name: str
    found_time: float
    anchors: tuple[Anchor, ...]

    @property
    def rest_locations(self) -> list[str]:
        return [a.loc for a in self.anchors if a.kind == "rest" and a.loc]

    @property
    def rest_positions(self) -> list[Coord]:
        return [a.xy for a in self.anchors if a.kind == "rest" and a.xy is not None]


def _fluent_args(fluent: Any) -> Sequence[str]:
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
    """The objects a plan is actually *about*.

    Searching a location reveals every object that was truly there, not just the
    one being looked for, and each of those lands in ``entity_positions``. Drawing
    all of them would bury the plan under a receptacle's worth of clutter, so an
    object earns a sprite only by appearing in the goal, in an executed action, or
    in a robot's hand.
    """
    tracked = set(entity_positions) - known_robots
    if glyph_objects is not None:
        return set(glyph_objects) & tracked

    candidates: set[str] = set()
    for fluent in goal_fluents:
        candidates.update(_fluent_args(fluent))
    for action_name, _start in actions_taken:
        # Grounded action names are the only handle available: the dashboard is
        # handed the name, never the Action, and Action exposes no argument list.
        # The environment parses them the same way in its own action filters.
        candidates.update(action_name.split()[1:])
    for entry in history:
        for fluent in entry.get("fluents", ()):  # type: ignore[union-attr]
            if getattr(fluent, "name", None) == "holding":
                args = _fluent_args(fluent)
                if len(args) >= 2:
                    candidates.add(args[1])

    if objects_by_type and "object" in objects_by_type:
        candidates &= set(objects_by_type["object"])
    # entity_positions is keyed by the first argument of `at`, so intersecting
    # with it drops locations for free -- and drops goal objects that were never
    # found, which have no position to draw at anyway.
    return candidates & tracked


def _collapse(anchors: list[Anchor]) -> tuple[Anchor, ...]:
    """Keep the first and last of each run of identically-targeted anchors.

    Dropping the interior keeps the timeline small; keeping the *last* is what
    makes a tween span only the pick itself rather than stretching back to
    whenever the object first arrived at the shelf.
    """
    collapsed: list[Anchor] = []
    start = 0
    while start < len(anchors):
        end = start
        while end + 1 < len(anchors) and anchors[end + 1].target == anchors[start].target:
            end += 1
        collapsed.append(anchors[start])
        if end != start:
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
    """Reconstruct each selected object's anchors from the fluent snapshots."""
    timelines: dict[str, ObjectTimeline] = {}
    for obj in sorted(selected):
        positions = entity_positions.get(obj, [])
        # Coordinates recorded alongside the fluent win over the shared lookup:
        # that is where an environment reports a location it placed itself.
        stored: dict[str, Coord] = {
            loc: coords for _t, loc, coords in positions if coords is not None
        }

        def resolve(loc: str) -> Coord | None:
            return stored.get(loc) or env_coords.get(loc)

        anchors: list[Anchor] = []
        found_time: float | None = None
        first_at_time: float | None = None
        for entry in history:
            time = float(entry["time"])
            at_loc: str | None = None
            holder: str | None = None
            for fluent in entry.get("fluents", ()):
                name = getattr(fluent, "name", None)
                args = _fluent_args(fluent)
                if name == "at" and len(args) >= 2 and args[0] == obj:
                    at_loc = args[1]
                elif name == "holding" and len(args) >= 2 and args[1] == obj:
                    holder = args[0]
                elif name == "found" and len(args) >= 1 and args[0] == obj:
                    if found_time is None:
                        found_time = time
            if at_loc is not None:
                xy = resolve(at_loc)
                if xy is None:
                    continue
                if first_at_time is None:
                    first_at_time = time
                anchors.append(Anchor(time, "rest", xy=xy, loc=at_loc))
            elif holder is not None:
                anchors.append(Anchor(time, "ride", robot=holder))

        # Positions recorded after the last snapshot come from
        # finalize_trajectories, which reads the pending `at` out of the
        # upcoming effects -- so a place still in flight at goal time lands.
        last_time = anchors[-1].time if anchors else float("-inf")
        for time, loc, coords in positions:
            if time <= last_time:
                continue
            xy = coords or env_coords.get(loc)
            if xy is not None:
                anchors.append(Anchor(float(time), "rest", xy=xy, loc=loc))

        if not anchors:
            continue
        anchors.sort(key=lambda a: a.time)
        if found_time is None:
            found_time = first_at_time if first_at_time is not None else anchors[0].time
        timelines[obj] = ObjectTimeline(obj, found_time, _collapse(anchors))
    return timelines


def _smoothstep(u: np.ndarray) -> np.ndarray:
    return u * u * (3.0 - 2.0 * u)


def sample(
    timeline: ObjectTimeline,
    times: Any,
    robot_xy: dict[str, Any],
    *,
    fade: float,
    offset_for: Callable[[Anchor], Coord] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Position and opacity of one object's sprite at each of *times*.

    A pure function of *times*, which is what lets the video's poster frame jump
    to the end and rewind without any special handling.

    Args:
        robot_xy: interpolated robot positions, sampled at the same *times*.
        fade: seconds over which the sprite fades in once found.
        offset_for: per-anchor fan-out offset, in grid cells.
    """
    query = np.asarray(times, dtype=float).reshape(-1)
    anchors = timeline.anchors
    count = len(query)

    def offset(anchor: Anchor) -> Coord:
        return offset_for(anchor) if offset_for is not None else (0.0, 0.0)

    # Fall back to a rest position whenever a carrier has no interpolated track
    # (a robot the trajectory builder dropped for having too few waypoints).
    rest_fallback = next(
        (a.xy for a in anchors if a.kind == "rest" and a.xy is not None), (0.0, 0.0)
    )

    tracks = np.empty((len(anchors), count, 2), dtype=float)
    for index, anchor in enumerate(anchors):
        dx, dy = offset(anchor)
        if anchor.kind == "rest" and anchor.xy is not None:
            tracks[index, :, 0] = anchor.xy[0] + dx
            tracks[index, :, 1] = anchor.xy[1] + dy
        else:
            track = robot_xy.get(anchor.robot or "")
            if track is None:
                tracks[index, :, 0] = rest_fallback[0] + dx
                tracks[index, :, 1] = rest_fallback[1] + dy
            else:
                arr = np.asarray(track, dtype=float)
                tracks[index, :, 0] = arr[:, 0] + dx
                tracks[index, :, 1] = arr[:, 1] + dy

    anchor_times = np.array([a.time for a in anchors], dtype=float)
    lower = np.clip(np.searchsorted(anchor_times, query, side="right") - 1, 0, len(anchors) - 1)
    upper = np.minimum(lower + 1, len(anchors) - 1)

    rows = np.arange(count)
    position = tracks[lower, rows]

    span = anchor_times[upper] - anchor_times[lower]
    with np.errstate(divide="ignore", invalid="ignore"):
        progress = np.where(span > 0, (query - anchor_times[lower]) / span, 0.0)
    progress = np.clip(progress, 0.0, 1.0)

    same_target = np.array(
        [anchors[i].target == anchors[min(i + 1, len(anchors) - 1)].target
         for i in range(len(anchors))]
    )
    # Blend only across a genuine change of anchor: a pick, a place, or a move
    # between shelves. Within a run the position is constant anyway.
    blending = (upper != lower) & ~same_target[lower]
    if blending.any():
        weight = _smoothstep(progress[blending])[:, None]
        start = tracks[lower[blending], rows[blending]]
        end = tracks[upper[blending], rows[blending]]
        position[blending] = start * (1.0 - weight) + end * weight

    if fade > 0:
        alpha = np.clip((query - timeline.found_time) / fade, 0.0, 1.0)
    else:
        alpha = (query >= timeline.found_time).astype(float)
    return position, alpha

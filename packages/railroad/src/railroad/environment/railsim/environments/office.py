"""Office generator, ported from the Unity-era ``office2`` environment.

Random axis-aligned hallway centerlines are drawn on a grid (kept mutually
connected), inflated to corridor width, then lined with rooms reachable
through doors; "special" rooms bridge pairs of hallway intersections, and
rooms are furnished with tables (solid blocks, obstacles in the occupancy
grid). Walls bordering hallways and rooms get class-specific colors,
matching the Unity wall-hallway / wall-classroom materials.

Differences from the original ``office2.py``: local ``np.random.default_rng``
instead of global seeding (deterministic per seed, but not bit-identical to
the old maps), hallway intersections are computed directly from the
centerline segments (the original skeletonized the corridor mask and used
``sknw``), and results are returned as `MapData`.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import scipy.ndimage
import skimage.graph

from .. import grid as grid_utils
from .base import MapData
from ..palette import Color, resolve_palette
from ..pose import Pose

SEMANTIC_LABELS = {
    'background': 0,
    'clutter': 1,
    'door': 2,
    'hallway': 3,
    'room': 4,
}

# Transient markers used only during line generation.
_L_TMP = 100
_L_UNK = 5

# Cells (resolution 0.5 m) for which adjacent walls take a class color.
WALL_CLASS_LABELS = ('hallway', 'room')

LineSegment = tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class OfficeConfig:
    """Defaults match the original ``office2.py`` (units are grid cells
    unless suffixed ``_m``; one cell is ``resolution`` meters)."""

    resolution: float = 0.5
    grid_size: tuple[int, int] = (500, 300)
    num_hallways: int = 5
    boundary_threshold: int = 30
    min_spacing_hallways: int = 30
    hallway_width: int = 5
    room_width: int = 20
    room_length_range: tuple[int, int] = (25, 35)
    room_door_space: int = 1
    hallway_room_space: int = 1
    door_size: int = 8
    max_tables_per_room: int = 2
    table_size_range: tuple[int, int] = (4, 8)
    table_wall_buffer: int = 3
    inflation_radius_m: float = 0.75
    min_start_goal_separation_cells: int = 150
    # Sparser than the old sim's 3.0 m: distinct light pools, ~1/3 the
    # lights (and per-fragment lighting cost) on a default-size office.
    min_interlight_distance_m: float = 6.0


def generate_random_lines(rng: np.random.Generator,
                          num_of_lines: int = OfficeConfig.num_hallways,
                          grid_size: tuple[int, int] = OfficeConfig.grid_size,
                          spacing_between_lines: int = OfficeConfig.min_spacing_hallways,
                          boundary_threshold: int = OfficeConfig.boundary_threshold,
                          max_iter: int = 10000) -> tuple[np.ndarray, list[LineSegment]]:
    """Random horizontal/vertical hallway centerlines, kept connected.

    Returns the grid marked with `_L_UNK` along the lines, and the list of
    segments as ((start_i, start_j), (end_i, end_j)).
    """

    def _check_if_connected(semantic_grid: np.ndarray, grid: np.ndarray) -> bool:
        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        grid[semantic_grid == _L_UNK] = 0
        new_grid = 1 - grid.copy()
        _, num_features = scipy.ndimage.label(new_grid, structure=s)
        return num_features <= 1

    # Track rows/cols too close to an existing parallel line.
    row: set[int] = set()
    col: set[int] = set()
    space_between_parallel = spacing_between_lines
    xy_lower_bound = 0 + boundary_threshold + 1
    x_upper_bound = grid_size[0] - boundary_threshold - 1
    y_upper_bound = grid_size[1] - boundary_threshold - 1

    grid = np.ones(grid_size, dtype=int)
    final_semantic_grid = grid.copy() * _L_TMP
    intermediate_semantic_grid = grid.copy() * _L_TMP
    line_segments: list[LineSegment] = []

    for _ in range(max_iter):
        random_point = rng.integers(
            xy_lower_bound, [x_upper_bound, y_upper_bound])

        distance_to_bounds = [x_upper_bound - random_point[0],
                              random_point[0] - xy_lower_bound + 1,
                              random_point[1] - xy_lower_bound + 1,
                              y_upper_bound - random_point[1]]
        sorted_direction = np.argsort(distance_to_bounds)[::-1]

        for direction in sorted_direction:
            if direction == 0:  # Draws toward increasing i
                if random_point[1] in col:
                    continue
                intermediate_semantic_grid[random_point[0]:x_upper_bound + 1,
                                           random_point[1]] = _L_UNK

                if _check_if_connected(intermediate_semantic_grid, grid.copy()):
                    final_semantic_grid = intermediate_semantic_grid.copy()
                    grid[final_semantic_grid == _L_UNK] = 0
                    lb = max(random_point[1] - space_between_parallel, xy_lower_bound)
                    ub = min(random_point[1] + space_between_parallel + 1, y_upper_bound)
                    lb_buffer = max(random_point[0] - space_between_parallel, xy_lower_bound)

                    line_segments.append(((int(random_point[0]), int(random_point[1])),
                                          (int(x_upper_bound), int(random_point[1]))))
                    col.update(range(lb, ub))
                    row.update(range(lb_buffer, random_point[0]))
                    break
                else:
                    intermediate_semantic_grid = final_semantic_grid.copy()

            elif direction == 1:  # Draws toward decreasing i
                if random_point[1] in col:
                    continue
                intermediate_semantic_grid[xy_lower_bound:random_point[0] + 1,
                                           random_point[1]] = _L_UNK

                if _check_if_connected(intermediate_semantic_grid, grid.copy()):
                    final_semantic_grid = intermediate_semantic_grid.copy()
                    grid[final_semantic_grid == _L_UNK] = 0
                    lb = max(random_point[1] - space_between_parallel, xy_lower_bound)
                    ub = min(random_point[1] + space_between_parallel + 1, y_upper_bound)
                    ub_buffer = min(random_point[0] + space_between_parallel, x_upper_bound)

                    line_segments.append(((int(random_point[0]), int(random_point[1])),
                                          (int(xy_lower_bound), int(random_point[1]))))
                    col.update(range(lb, ub))
                    row.update(range(random_point[0], ub_buffer))
                    break
                else:
                    intermediate_semantic_grid = final_semantic_grid.copy()

            elif direction == 2:  # Draws toward decreasing j
                if random_point[0] in row:
                    continue
                intermediate_semantic_grid[random_point[0],
                                           xy_lower_bound:random_point[1] + 1] = _L_UNK
                if _check_if_connected(intermediate_semantic_grid, grid.copy()):
                    final_semantic_grid = intermediate_semantic_grid.copy()
                    grid[final_semantic_grid == _L_UNK] = 0
                    lb = max(random_point[0] - space_between_parallel, xy_lower_bound)
                    ub = min(random_point[0] + space_between_parallel + 1, x_upper_bound)
                    ub_buffer = min(random_point[1] + space_between_parallel, y_upper_bound)

                    line_segments.append(((int(random_point[0]), int(random_point[1])),
                                          (int(random_point[0]), int(xy_lower_bound))))
                    row.update(range(lb, ub))
                    col.update(range(random_point[1], ub_buffer))
                    break
                else:
                    intermediate_semantic_grid = final_semantic_grid.copy()

            elif direction == 3:  # Draws toward increasing j
                if random_point[0] in row:
                    continue
                intermediate_semantic_grid[random_point[0],
                                           random_point[1]:y_upper_bound + 1] = _L_UNK
                if _check_if_connected(intermediate_semantic_grid, grid.copy()):
                    final_semantic_grid = intermediate_semantic_grid.copy()
                    grid[final_semantic_grid == _L_UNK] = 0
                    lb = max(random_point[0] - space_between_parallel, xy_lower_bound)
                    ub = min(random_point[0] + space_between_parallel + 1, x_upper_bound)
                    lb_buffer = max(random_point[1] - space_between_parallel, xy_lower_bound)

                    line_segments.append(((int(random_point[0]), int(random_point[1])),
                                          (int(random_point[0]), int(y_upper_bound))))
                    row.update(range(lb, ub))
                    col.update(range(lb_buffer, random_point[1]))
                    break
                else:
                    intermediate_semantic_grid = final_semantic_grid.copy()

        if len(line_segments) >= num_of_lines:
            break
    else:
        warnings.warn(f"Needed to generate {num_of_lines} lines but only "
                      f"generated {len(line_segments)} lines.")

    return final_semantic_grid, line_segments


def inflate_lines_to_create_hallways(
        grid: np.ndarray,
        hallway_inflation_scale: int = OfficeConfig.hallway_width) -> np.ndarray:
    """Inflate centerlines by a square kernel into hallway corridors."""
    original_grid = np.zeros_like(grid)
    original_grid[grid == _L_UNK] = 1
    kernel_dim = 2 * hallway_inflation_scale + 1
    hallway_inflation_kernel = np.ones((kernel_dim, kernel_dim), dtype=int)

    grid_with_hallway = scipy.ndimage.convolve(
        original_grid, hallway_inflation_kernel)
    grid_with_hallway[grid_with_hallway > 0] = SEMANTIC_LABELS['hallway']
    grid_with_hallway[grid_with_hallway == 0] = SEMANTIC_LABELS['background']

    return grid_with_hallway


def determine_intersections(line_segments: list[LineSegment]) -> dict[str, np.ndarray]:
    """Hallway intersection and deadend points.

    The centerlines are axis-aligned, so intersections are computed
    directly from the segments (the original skeletonized the corridor
    mask and built a graph with ``sknw``): a junction is any point where a
    constant-i segment crosses or touches a constant-j segment.
    """
    normalized = []
    for (a, b) in line_segments:
        horizontal = a[0] == b[0]  # constant i, varying j
        const = a[0] if horizontal else a[1]
        axis = 1 if horizontal else 0
        lo, hi = sorted((a[axis], b[axis]))
        normalized.append((horizontal, const, lo, hi))

    intersections = []
    for idx_a, (horiz_a, const_a, lo_a, hi_a) in enumerate(normalized):
        for (horiz_b, const_b, lo_b, hi_b) in normalized[idx_a + 1:]:
            if horiz_a == horiz_b:
                continue  # parallel lines never share a row/col
            (h_const, h_lo, h_hi) = (const_a, lo_a, hi_a) if horiz_a else (const_b, lo_b, hi_b)
            (v_const, v_lo, v_hi) = (const_b, lo_b, hi_b) if horiz_a else (const_a, lo_a, hi_a)
            if h_lo <= v_const <= h_hi and v_lo <= h_const <= v_hi:
                intersections.append((h_const, v_const))

    junction_set = set(intersections)
    deadends = []
    for (a, b) in line_segments:
        for point in (tuple(a), tuple(b)):
            if point not in junction_set:
                deadends.append(point)

    return {
        'intersections': np.array(intersections, dtype=float).reshape(-1, 2),
        'deadends': np.array(deadends, dtype=float).reshape(-1, 2),
    }


def add_special_rooms(grid_with_hallway: np.ndarray,
                      intersections: np.ndarray,
                      hallway_inflation_scale: int = OfficeConfig.hallway_width,
                      room_length_range: tuple[int, int] = OfficeConfig.room_length_range,
                      room_door_space: int = OfficeConfig.room_door_space,
                      hallway_room_space: int = OfficeConfig.hallway_room_space,
                      door_size: int = OfficeConfig.door_size,
                      ) -> tuple[np.ndarray, list]:
    """Add rooms connecting two hallway intersections wherever possible."""
    labels = SEMANTIC_LABELS

    def _check_intersection_or_hallway_end(side_point, extended_point):
        check_point_start, check_point_end = side_point
        hallway_end_check_start, hallway_end_check_end = extended_point

        another_intersection_met, hallway_end = False, False
        if grid_with_sp_room[check_point_start[0], check_point_start[1]] == labels['hallway']:
            another_intersection_met = True
        if grid_with_sp_room[check_point_end[0], check_point_end[1]] == labels['hallway']:
            another_intersection_met = True
        if grid_with_sp_room[hallway_end_check_start[0], hallway_end_check_start[1]] == labels['background']:
            hallway_end = True
        if grid_with_sp_room[hallway_end_check_end[0], hallway_end_check_end[1]] == labels['background']:
            hallway_end = True
        return another_intersection_met, hallway_end

    grid_with_sp_room = grid_with_hallway.copy()
    intersections = np.round(intersections).astype(int)
    intersection_pairs: list[tuple[np.ndarray, np.ndarray, float]] = []
    for i, inter in enumerate(intersections):
        x, y = inter[0], inter[1]
        for next_point in intersections[i + 1:]:
            if next_point[0] == x or next_point[1] == y:
                intersection_pairs.append(
                    (inter, next_point, float(np.linalg.norm(inter - next_point))))
    intersection_pairs.sort(key=lambda entry: entry[2])

    min_room_length, max_room_length = room_length_range
    min_intersection_distance = min_room_length + 3 * hallway_inflation_scale
    max_intersection_distance = max_room_length + 5 * hallway_inflation_scale

    intersection_pairs = [
        entry for entry in intersection_pairs
        if min_intersection_distance <= entry[2] < max_intersection_distance]

    rooms_coords = []
    for start, end, _ in intersection_pairs:
        is_horizontal = start[0] == end[0]
        axis = int(is_horizontal)
        if start[axis] > end[axis]:
            start, end = end, start
        distance = {}
        if is_horizontal:
            # Min distance along the hallway the room can expand, ascending.
            another_intersection_met = False
            hallway_end = False
            distance_ascending = hallway_inflation_scale
            while not (another_intersection_met or hallway_end):
                distance_ascending += 1
                poi_ascending = start[0] + distance_ascending

                check_point_start = [poi_ascending, start[1] + hallway_inflation_scale + 1]
                check_point_end = [poi_ascending, end[1] - hallway_inflation_scale - 1]
                hallway_end_check_start = [poi_ascending + 1, start[1]]
                hallway_end_check_end = [poi_ascending + 1, end[1]]

                another_intersection_met, hallway_end = _check_intersection_or_hallway_end(
                    [check_point_start, check_point_end],
                    [hallway_end_check_start, hallway_end_check_end])
            distance['ascending'] = distance_ascending

            # ... and descending.
            another_intersection_met = False
            hallway_end = False
            distance_descending = hallway_inflation_scale
            while not (another_intersection_met or hallway_end):
                distance_descending += 1
                poi_descending = start[0] - distance_descending

                check_point_start = [poi_descending, start[1] + hallway_inflation_scale + 1]
                check_point_end = [poi_descending, end[1] - hallway_inflation_scale - 1]
                hallway_end_check_start = [poi_descending - 1, start[1]]
                hallway_end_check_end = [poi_descending - 1, end[1]]

                another_intersection_met, hallway_end = _check_intersection_or_hallway_end(
                    [check_point_start, check_point_end],
                    [hallway_end_check_start, hallway_end_check_end])
            distance['descending'] = distance_descending

            if distance['ascending'] > max_room_length:
                room_p1 = [start[0] + distance['ascending'] - max_room_length,
                           start[1] + hallway_inflation_scale + hallway_room_space + 1]
                room_p2 = [end[0] + distance['ascending'],
                           end[1] - hallway_inflation_scale - hallway_room_space]
                room_slice = grid_with_sp_room[room_p1[0] - 1:room_p2[0] + 1,
                                               room_p1[1]:room_p2[1]]
                if not (np.any(room_slice == labels['room'])
                        or np.any(room_slice == labels['hallway'])):
                    grid_with_sp_room[room_p1[0]:room_p2[0],
                                      room_p1[1]:room_p2[1]] = labels['room']
                    rooms_coords.append((room_p1, room_p2))
                    grid_with_sp_room[room_p1[0] + room_door_space:room_p1[0] + room_door_space + door_size,
                                      room_p1[1] - hallway_room_space:room_p1[1]] = labels['door']
                    grid_with_sp_room[room_p2[0] - room_door_space - door_size:room_p2[0] - room_door_space,
                                      room_p2[1]:room_p2[1] + hallway_room_space] = labels['door']

            if distance['descending'] > max_room_length:
                room_q1 = [start[0] - distance['descending'] + 1,
                           start[1] + hallway_inflation_scale + hallway_room_space + 1]
                room_q2 = [end[0] - distance['descending'] + max_room_length,
                           end[1] - hallway_inflation_scale - hallway_room_space]
                room_slice = grid_with_sp_room[room_q1[0] - 1:room_q2[0] + 1,
                                               room_q1[1]:room_q2[1]]
                if not (np.any(room_slice == labels['room'])
                        or np.any(room_slice == labels['hallway'])):
                    grid_with_sp_room[room_q1[0]:room_q2[0],
                                      room_q1[1]:room_q2[1]] = labels['room']
                    rooms_coords.append((room_q1, room_q2))
                    grid_with_sp_room[room_q1[0] + room_door_space:room_q1[0] + room_door_space + door_size,
                                      room_q1[1] - hallway_room_space:room_q1[1]] = labels['door']
                    grid_with_sp_room[room_q2[0] - room_door_space - door_size:room_q2[0] - room_door_space,
                                      room_q2[1]:room_q2[1] + hallway_room_space] = labels['door']
        else:
            another_intersection_met = False
            hallway_end = False
            distance_ascending = hallway_inflation_scale
            while not (another_intersection_met or hallway_end):
                distance_ascending += 1
                poi_ascending = start[1] + distance_ascending
                check_point_start = [start[0] + hallway_inflation_scale + 1, poi_ascending]
                check_point_end = [end[0] - hallway_inflation_scale - 1, poi_ascending]
                hallway_end_check_start = [start[0], poi_ascending + 1]
                hallway_end_check_end = [end[0], poi_ascending + 1]
                another_intersection_met, hallway_end = _check_intersection_or_hallway_end(
                    [check_point_start, check_point_end],
                    [hallway_end_check_start, hallway_end_check_end])
            distance['ascending'] = distance_ascending

            another_intersection_met = False
            hallway_end = False
            distance_descending = hallway_inflation_scale
            while not (another_intersection_met or hallway_end):
                distance_descending += 1
                poi_descending = start[1] - distance_descending
                check_point_start = [start[0] + hallway_inflation_scale + 1, poi_descending]
                check_point_end = [end[0] - hallway_inflation_scale - 1, poi_descending]
                hallway_end_check_start = [start[0], poi_descending - 1]
                hallway_end_check_end = [end[0], poi_descending - 1]
                another_intersection_met, hallway_end = _check_intersection_or_hallway_end(
                    [check_point_start, check_point_end],
                    [hallway_end_check_start, hallway_end_check_end])
            distance['descending'] = distance_descending

            if distance['ascending'] > max_room_length:
                room_p1 = [start[0] + hallway_inflation_scale + hallway_room_space + 1,
                           start[1] + distance['ascending'] - max_room_length]
                room_p2 = [end[0] - hallway_inflation_scale - hallway_room_space,
                           end[1] + distance['ascending'] - 1]
                room_slice = grid_with_sp_room[room_p1[0]:room_p2[0],
                                               room_p1[1] - 1:room_p2[1] + 1]
                if not (np.any(room_slice == labels['room'])
                        or np.any(room_slice == labels['hallway'])):
                    grid_with_sp_room[room_p1[0]:room_p2[0],
                                      room_p1[1]:room_p2[1]] = labels['room']
                    rooms_coords.append((room_p1, room_p2))
                    grid_with_sp_room[room_p1[0] - hallway_room_space:room_p1[0],
                                      room_p1[1] + room_door_space:room_p1[1] + room_door_space + door_size] = (
                        labels['door'])
                    grid_with_sp_room[room_p2[0]:room_p2[0] + hallway_room_space,
                                      room_p2[1] - room_door_space - door_size:room_p2[1] - room_door_space] = (
                        labels['door'])

            if distance['descending'] > max_room_length:
                room_q1 = [start[0] + hallway_inflation_scale + hallway_room_space + 1,
                           start[1] - distance['descending']]
                room_q2 = [end[0] - hallway_inflation_scale - hallway_room_space,
                           end[1] - distance['descending'] + max_room_length]
                room_slice = grid_with_sp_room[room_q1[0] + 1:room_q2[0] - 1,
                                               room_q1[1]:room_q2[1]]
                if not (np.any(room_slice == labels['room'])
                        or np.any(room_slice == labels['hallway'])):
                    grid_with_sp_room[room_q1[0]:room_q2[0],
                                      room_q1[1]:room_q2[1]] = labels['room']
                    rooms_coords.append((room_q1, room_q2))
                    grid_with_sp_room[room_q1[0] - hallway_room_space:room_q1[0],
                                      room_q1[1] + room_door_space:room_q1[1] + room_door_space + door_size] = (
                        labels['door'])
                    grid_with_sp_room[room_q2[0]:room_q2[0] + hallway_room_space,
                                      room_q2[1] - room_door_space - door_size:room_q2[1] - room_door_space] = (
                        labels['door'])

    return grid_with_sp_room, rooms_coords


def add_rooms(rng: np.random.Generator,
              grid_with_hallway: np.ndarray,
              line_segments: list[LineSegment],
              hallway_inflation_scale: int = OfficeConfig.hallway_width,
              room_b: int = OfficeConfig.room_width,
              room_l_range: tuple[int, int] = OfficeConfig.room_length_range,
              room_door_space: int = OfficeConfig.room_door_space,
              hallway_room_space: int = OfficeConfig.hallway_room_space,
              door_size: int = OfficeConfig.door_size,
              ) -> tuple[np.ndarray, list]:
    """Add rooms (with doors) along and at the ends of each hallway."""
    labels = SEMANTIC_LABELS
    grid_with_room = grid_with_hallway.copy()
    rooms_coords = []
    for line in line_segments:
        start, end = line
        is_horizontal = start[0] == end[0]
        axis = int(is_horizontal)
        if start[axis] > end[axis]:
            start, end = end, start

        if is_horizontal:
            # Rooms at the hallway end points.
            room_l = int(rng.integers(*room_l_range))
            room_p1 = (start[0] - int(room_l / 2),
                       start[1] - hallway_inflation_scale - room_b - hallway_room_space)
            room_p2 = (room_p1[0] + room_l, room_p1[1] + room_b)
            room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1,
                                        room_p1[1] - 1:room_p2[1] + 1]
            if not (np.any(room_slice == labels['room'])
                    or np.any(room_slice == labels['hallway'])):
                grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = labels['room']
                rooms_coords.append((room_p1, room_p2))
                door_p1 = (start[0] - int(door_size / 2),
                           start[1] - hallway_inflation_scale - hallway_room_space)
                door_p2 = (door_p1[0] + door_size, door_p1[1] + hallway_room_space)
                grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = labels['door']

            room_l = int(rng.integers(*room_l_range))
            room_q1 = (end[0] - int(room_l / 2),
                       end[1] + hallway_inflation_scale + hallway_room_space + 1)
            room_q2 = (room_q1[0] + room_l, room_q1[1] + room_b)
            room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1,
                                        room_q1[1] - 1:room_q2[1] + 1]
            if not (np.any(room_slice == labels['room'])
                    or np.any(room_slice == labels['hallway'])):
                grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = labels['room']
                rooms_coords.append((room_q1, room_q2))
                door_q1 = (end[0] - int(door_size / 2),
                           end[1] + hallway_inflation_scale + 1)
                door_q2 = (door_q1[0] + door_size, door_q1[1] + hallway_room_space)
                grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = labels['door']

            # Rooms along the hallway.
            for y in range(start[1], end[1] - hallway_inflation_scale, 1):
                room_l = int(rng.integers(*room_l_range))
                room_p1 = (start[0] - hallway_inflation_scale - room_b - hallway_room_space, y)
                room_p2 = (room_p1[0] + room_b, room_p1[1] + room_l)
                room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1,
                                            room_p1[1] - 1:room_p2[1] + 1]
                if not (np.any(room_slice == labels['room'])
                        or np.any(room_slice == labels['hallway'])):
                    grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = labels['room']
                    rooms_coords.append((room_p1, room_p2))
                    door_p1 = (room_p2[0], room_p2[1] - room_door_space - door_size)
                    door_p2 = (room_p2[0] + hallway_room_space, room_p2[1] - room_door_space)
                    # Correction for a door extending beyond the hallway end.
                    door_check_slice = grid_with_room[door_p2[0]:door_p2[0] + 1,
                                                      door_p1[1]:door_p2[1]]
                    overflow_len = len(np.where(door_check_slice == labels['background'])[1])
                    if overflow_len > 0:
                        door_p1 = (room_p1[0] + room_b, room_p1[1] + room_door_space)
                        door_p2 = (door_p1[0] + hallway_room_space, door_p1[1] + door_size)
                    grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = labels['door']

                room_l = int(rng.integers(*room_l_range))
                room_q1 = (start[0] + hallway_inflation_scale + 1 + hallway_room_space, y)
                room_q2 = (room_q1[0] + room_b, room_q1[1] + room_l)
                room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1,
                                            room_q1[1] - 1:room_q2[1] + 1]
                if not (np.any(room_slice == labels['room'])
                        or np.any(room_slice == labels['hallway'])):
                    grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = labels['room']
                    rooms_coords.append((room_q1, room_q2))
                    door_q1 = (room_q1[0] - hallway_room_space, room_q1[1] + room_door_space)
                    door_q2 = (room_q1[0], room_q1[1] + room_door_space + door_size)
                    grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = labels['door']

        else:
            # Rooms at the hallway end points.
            room_l = int(rng.integers(*room_l_range))
            room_p1 = (start[0] - hallway_inflation_scale - room_b - hallway_room_space,
                       start[1] - int(room_l / 2))
            room_p2 = (room_p1[0] + room_b, room_p1[1] + room_l)
            room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1,
                                        room_p1[1] - 1:room_p2[1] + 1]
            if not (np.any(room_slice == labels['room'])
                    or np.any(room_slice == labels['hallway'])):
                grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = labels['room']
                rooms_coords.append((room_p1, room_p2))
                door_p1 = (start[0] - hallway_inflation_scale - hallway_room_space,
                           start[1] - int(door_size / 2))
                door_p2 = (door_p1[0] + hallway_room_space, door_p1[1] + door_size)
                grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = labels['door']

            room_l = int(rng.integers(*room_l_range))
            room_q1 = (end[0] + hallway_inflation_scale + hallway_room_space + 1,
                       end[1] - int(room_l / 2))
            room_q2 = (room_q1[0] + room_b, room_q1[1] + room_l)
            room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1,
                                        room_q1[1] - 1:room_q2[1] + 1]
            if not (np.any(room_slice == labels['room'])
                    or np.any(room_slice == labels['hallway'])):
                grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = labels['room']
                rooms_coords.append((room_q1, room_q2))
                door_q1 = (end[0] + hallway_inflation_scale + 1,
                           end[1] - int(door_size / 2))
                door_q2 = (door_q1[0] + hallway_room_space, door_q1[1] + door_size)
                grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = labels['door']

            # Rooms along the hallway.
            for x in range(start[0], end[0] - hallway_inflation_scale, 1):
                room_l = int(rng.integers(*room_l_range))
                room_p1 = (x, start[1] - hallway_inflation_scale - room_b - hallway_room_space)
                room_p2 = (room_p1[0] + room_l, room_p1[1] + room_b)
                room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1,
                                            room_p1[1] - 1:room_p2[1] + 1]
                if not (np.any(room_slice == labels['room'])
                        or np.any(room_slice == labels['hallway'])):
                    grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = labels['room']
                    rooms_coords.append((room_p1, room_p2))
                    door_p1 = (room_p2[0] - room_door_space - door_size, room_p2[1])
                    door_p2 = (room_p2[0] - room_door_space, room_p2[1] + hallway_room_space)
                    # Correction for a door extending beyond the hallway end.
                    door_check_slice = grid_with_room[door_p1[0]:door_p2[0],
                                                      door_p2[1]:door_p2[1] + 1]
                    overflow_len = len(np.where(door_check_slice == labels['background'])[0])
                    if overflow_len > 0:
                        door_p1 = (room_p1[0] + room_door_space, room_p1[1] + room_b)
                        door_p2 = (door_p1[0] + door_size, door_p1[1] + hallway_room_space)
                    grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = labels['door']

                room_l = int(rng.integers(*room_l_range))
                room_q1 = (x, start[1] + hallway_inflation_scale + 1 + hallway_room_space)
                room_q2 = (room_q1[0] + room_l, room_q1[1] + room_b)
                room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1,
                                            room_q1[1] - 1:room_q2[1] + 1]
                if not (np.any(room_slice == labels['room'])
                        or np.any(room_slice == labels['hallway'])):
                    grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = labels['room']
                    rooms_coords.append((room_q1, room_q2))
                    door_q1 = (room_q1[0] + room_door_space, room_q1[1] - hallway_room_space)
                    door_q2 = (room_q1[0] + room_door_space + door_size, room_q1[1])
                    grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = labels['door']

    return grid_with_room, rooms_coords


def add_tables(rng: np.random.Generator,
               grid_with_rooms: np.ndarray,
               rooms_coords: list,
               max_tables_per_room: int = OfficeConfig.max_tables_per_room,
               table_size_range: tuple[int, int] = OfficeConfig.table_size_range,
               table_wall_buffer: int = OfficeConfig.table_wall_buffer,
               ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Add tables (clutter) to rooms.

    Returns the grid with tables and a list of (center_x, center_y,
    length_x, length_y) tuples in grid cells.
    """
    labels = SEMANTIC_LABELS
    grid_with_tables = grid_with_rooms.copy()
    table_poses_sizes = []
    for room_p1, room_p2 in rooms_coords:
        for _ in range(max_tables_per_room):
            size_x, size_y = rng.choice(
                np.arange(table_size_range[0], table_size_range[1] + 1, 2), size=2)
            size_x, size_y = int(size_x), int(size_y)
            table_x = int(rng.integers(room_p1[0] + int(size_x / 2) + table_wall_buffer,
                                       room_p2[0] - int(size_x / 2) - table_wall_buffer))
            table_y = int(rng.integers(room_p1[1] + int(size_y / 2) + table_wall_buffer,
                                       room_p2[1] - int(size_y / 2) - table_wall_buffer))
            table_p1 = (table_x - int(size_x / 2), table_y - int(size_y / 2))
            table_p2 = (table_x + int(size_x / 2), table_y + int(size_y / 2))
            table_slice = grid_with_tables[table_p1[0]:table_p2[0],
                                           table_p1[1]:table_p2[1]]
            if not np.any(table_slice == labels['clutter']):
                grid_with_tables[table_p1[0]:table_p2[0],
                                 table_p1[1]:table_p2[1]] = labels['clutter']
                table_poses_sizes.append((table_x, table_y, size_x, size_y))

    return grid_with_tables, table_poses_sizes


def _sample_start_end_cells(rng: np.random.Generator,
                            occ_grid: np.ndarray,
                            config: OfficeConfig,
                            num_attempts: int = 1000) -> tuple[tuple[int, int], tuple[int, int]]:
    """Two random free cells on the inflated grid, connected by a path of
    cost >= ``min_start_goal_separation_cells`` (the old sim's
    ``get_start_goal_poses``)."""
    inflation_radius_cells = config.inflation_radius_m / config.resolution
    inflated = grid_utils.inflate_grid(occ_grid, inflation_radius_cells)
    free_cells = np.argwhere(inflated < 0.5)
    if len(free_cells) < 2:
        raise RuntimeError("No free space to sample start/goal poses from.")

    costs = np.ones_like(inflated)
    costs[inflated >= 0.5] = -1  # barrier cells for skimage's MCP

    for _ in range(num_attempts):
        idx = rng.choice(len(free_cells), size=2, replace=False)
        start, goal = free_cells[idx[0]], free_cells[idx[1]]
        try:
            _, cost = skimage.graph.route_through_array(
                costs, tuple(start), tuple(goal),
                fully_connected=True, geometric=True)
        except ValueError:
            continue  # disconnected components
        if cost >= config.min_start_goal_separation_cells:
            return (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
    raise RuntimeError("Could not find a pair of poses that connect "
                       "during start/goal pose generation.")


def gen_map_office(config: OfficeConfig = OfficeConfig(),
                   seed: int | None = None,
                   palette: Mapping[str, Color] | None = None) -> MapData:
    rng = np.random.default_rng(seed)

    grid_with_lines, line_segments = generate_random_lines(
        rng,
        num_of_lines=config.num_hallways,
        grid_size=config.grid_size,
        spacing_between_lines=config.min_spacing_hallways,
        boundary_threshold=config.boundary_threshold)
    grid_with_hallway = inflate_lines_to_create_hallways(
        grid_with_lines, hallway_inflation_scale=config.hallway_width)
    features = determine_intersections(line_segments)
    grid_with_special_rooms, special_rooms_coords = add_special_rooms(
        grid_with_hallway,
        intersections=features['intersections'],
        hallway_inflation_scale=config.hallway_width,
        room_length_range=config.room_length_range,
        room_door_space=config.room_door_space,
        hallway_room_space=config.hallway_room_space,
        door_size=config.door_size)
    grid_with_rooms, rooms_coords = add_rooms(
        rng, grid_with_special_rooms, line_segments,
        hallway_inflation_scale=config.hallway_width,
        room_b=config.room_width,
        room_l_range=config.room_length_range,
        room_door_space=config.room_door_space,
        hallway_room_space=config.hallway_room_space,
        door_size=config.door_size)
    rooms_coords += special_rooms_coords

    grid, table_poses_sizes = add_tables(
        rng, grid_with_rooms, rooms_coords,
        max_tables_per_room=config.max_tables_per_room,
        table_size_range=config.table_size_range,
        table_wall_buffer=config.table_wall_buffer)
    occ_grid = (grid <= SEMANTIC_LABELS['clutter']).astype(float)

    start_cell, end_cell = _sample_start_end_cells(rng, occ_grid, config)

    res = config.resolution
    tables_m = tuple((x * res, y * res, sx * res, sy * res)
                     for x, y, sx, sy in table_poses_sizes)

    return MapData(occ_grid=occ_grid,
                   semantic_grid=grid.astype(float),
                   semantic_labels=dict(SEMANTIC_LABELS),
                   start_cell=start_cell,
                   end_cell=end_cell,
                   resolution=res,
                   tables=tables_m,
                   wall_class_labels=WALL_CLASS_LABELS,
                   palette=resolve_palette(palette),
                   min_interlight_distance=config.min_interlight_distance_m)


class OfficeGenerator:
    """`MapGenerator` implementation for the office environment."""

    def __init__(self,
                 config: OfficeConfig = OfficeConfig(),
                 palette: Mapping[str, Color] | None = None):
        self.config = config
        self.palette = palette

    def generate(self, seed: int | None = None) -> MapData:
        return gen_map_office(self.config, seed=seed, palette=self.palette)

    def sample_start_goal(self, map_data: MapData, seed: int | None = None) -> tuple[Pose, Pose]:
        rng = random.Random(seed)
        res = map_data.resolution
        start = Pose(x=map_data.start_cell[0] * res,
                     y=map_data.start_cell[1] * res,
                     yaw=2 * math.pi * rng.random())
        goal = Pose(x=map_data.end_cell[0] * res,
                    y=map_data.end_cell[1] * res,
                    yaw=2 * math.pi * rng.random())
        return start, goal


def make_office(seed: int | None = None,
                config: OfficeConfig = OfficeConfig(),
                palette: Mapping[str, Color] | None = None) -> tuple[MapData, Pose, Pose]:
    """Generate an office map plus start/goal poses (meters).

    Office maps have no ``goal_path`` label (matching the original sim), so
    build their worlds with ``num_breadcrumb_elements=0``. ``palette``
    entries override `DEFAULT_PALETTE` colors for domain randomization.
    """
    generator = OfficeGenerator(config, palette=palette)
    map_data = generator.generate(seed=seed)
    start, goal = generator.sample_start_goal(map_data, seed=seed)
    return map_data, start, goal

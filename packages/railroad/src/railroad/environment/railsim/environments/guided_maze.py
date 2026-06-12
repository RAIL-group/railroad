"""Guided maze generator, ported from the Unity-era `environments` package.

A maze is carved on a coarse cell lattice by randomized wall removal; a
"goal path" between two random lattice cells is then routed and marked in
the semantic grid (and rendered wider than regular hallways). Breadcrumbs
sampled along that path provide the visual guidance signal.

Differences from the original `guided_maze.py`: local RNGs instead of
seeding global `random`/`np.random` state (deterministic per seed, but not
bit-identical to the old maps), and results are returned as `MapData`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import skimage.graph

from railroad.navigation.pathing import cells_connected

from .base import MapData
from ..palette import Color, resolve_palette
from ..pose import Pose

SEMANTIC_LABELS = {
    'wall': 1,
    'hallway': 2,
    'goal_path': 3,
}


@dataclass(frozen=True)
class GuidedMazeConfig:
    """Defaults match what `generate.map_and_poses` used for the old sim."""

    width_cells: int = 6
    height_cells: int = 8
    path_width: int = 10
    wide_path_width: int = 14
    all_wide: bool = True
    resolution: float = 0.3
    inflation_radius_m: float = 0.75


def _axis_positions(n: int, pw: int, ww: int) -> tuple[np.ndarray, np.ndarray]:
    """Block widths and their cumulative offsets for inflating an ``n``-cell
    lattice axis: even indices (cells) are ``ww`` wide, odd indices
    (passages) ``pw`` wide."""
    widths = np.ones(2 * n, dtype=int) * pw
    widths[::2] = ww
    return widths, np.cumsum(widths).astype(int)


def gen_map_maze_base(config: GuidedMazeConfig = GuidedMazeConfig(),
                      seed: int | None = None,
                      palette: Mapping[str, Color] | None = None) -> MapData:
    pw = config.path_width
    ww = config.path_width + 2
    PW = config.wide_path_width
    dPW = (PW - pw) // 2

    assert PW < ww + pw

    rng = random.Random(seed)
    w, h = config.width_cells, config.height_cells

    # Wall and cell bookkeeping for randomized wall removal (Kruskal-style:
    # remove a wall whenever it separates two disconnected cell sets).
    v_walls = [(x, y, x + 1, y) for x in range(w - 1) for y in range(h)]
    h_walls = [(x, y, x, y + 1) for x in range(w) for y in range(h - 1)]
    walls = v_walls + h_walls
    cells = [{(x, y)} for x in range(w) for y in range(h)]

    # Lattice grid: cells at even indices, walls between them at odd indices.
    grid = np.zeros([w * 2 - 1, h * 2 - 1])
    grid[::2, ::2] = 1

    def get_wall_cell(wall: tuple[int, int, int, int]) -> tuple[int, int]:
        return (wall[0] + wall[2], wall[1] + wall[3])

    rng.shuffle(walls)

    for wall in walls:
        set_a = None
        set_b = None
        for s in cells:
            if (wall[0], wall[1]) in s:
                set_a = s
            if (wall[2], wall[3]) in s:
                set_b = s
        assert set_a is not None and set_b is not None

        if set_a is not set_b:
            cells.remove(set_a)
            cells.remove(set_b)
            cells.append(set_a | set_b)
            grid[get_wall_cell(wall)] = 1

    # Route the goal path between two random lattice cells. Cells valued -1
    # are barriers for skimage's MCP.
    path_grid = grid
    path_grid[path_grid == 0] = -1
    start = (2 * rng.randint(0, w - 1), 2 * rng.randint(0, h - 1))
    end = start
    while end == start:
        end = (2 * rng.randint(0, w - 1), 2 * rng.randint(0, h - 1))

    path, _ = skimage.graph.route_through_array(path_grid, start, end, fully_connected=False)
    path_arr = np.array(path).T
    grid[path_arr[0], path_arr[1]] = 2

    # Inflate the lattice to a high-resolution grid: lattice cells become
    # ww-wide blocks separated by pw-wide passages.
    xd, xp = _axis_positions(w, pw, ww)
    yd, yp = _axis_positions(h, pw, ww)

    semantic_grid = np.zeros([xp[-1] + ww, yp[-1] + ww])

    def paint(xx: int, yy: int, label: int, pad_lo: int, pad_hi: int) -> None:
        semantic_grid[xp[xx] - pad_lo:xp[xx] + xd[xx + 1] + pad_hi,
                      yp[yy] - pad_lo:yp[yy] + yd[yy + 1] + pad_hi] = label

    # Paint hallway cells, then goal-path cells second so they win where the
    # padded blocks overlap. ``all_wide`` widens hallways on the high side.
    hallway_pad_hi = dPW if config.all_wide else 0
    for xx in range(2 * w - 1):
        for yy in range(2 * h - 1):
            if grid[xx, yy] == 1:
                paint(xx, yy, SEMANTIC_LABELS['hallway'], dPW, hallway_pad_hi)
    for xx in range(2 * w - 1):
        for yy in range(2 * h - 1):
            if grid[xx, yy] == 2:
                paint(xx, yy, SEMANTIC_LABELS['goal_path'], dPW, dPW)

    out_grid = np.ones(semantic_grid.shape)
    out_grid[semantic_grid == SEMANTIC_LABELS['goal_path']] = 0
    out_grid[semantic_grid == SEMANTIC_LABELS['hallway']] = 0
    semantic_grid[out_grid == 1] = SEMANTIC_LABELS['wall']

    start_cell = (int(xp[start[0]] + pw // 2), int(yp[start[1]] + pw // 2))
    end_cell = (int(xp[end[0]] + pw // 2), int(yp[end[1]] + pw // 2))

    return MapData(occ_grid=out_grid,
                   semantic_grid=semantic_grid,
                   semantic_labels=dict(SEMANTIC_LABELS),
                   start_cell=start_cell,
                   end_cell=end_cell,
                   resolution=config.resolution,
                   palette=resolve_palette(palette))


class GuidedMazeGenerator:
    """`MapGenerator` implementation for the guided maze."""

    def __init__(self,
                 config: GuidedMazeConfig = GuidedMazeConfig(),
                 palette: Mapping[str, Color] | None = None):
        self.config = config
        self.palette = palette

    def generate(self, seed: int | None = None) -> MapData:
        return gen_map_maze_base(self.config, seed=seed, palette=self.palette)

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


def make_guided_maze(seed: int | None = None,
                     config: GuidedMazeConfig = GuidedMazeConfig(),
                     palette: Mapping[str, Color] | None = None) -> tuple[MapData, Pose, Pose]:
    """Generate a guided maze plus start/goal poses (meters), with a
    connectivity sanity check matching the old `map_and_poses` behavior.
    ``palette`` entries override `DEFAULT_PALETTE` colors."""
    generator = GuidedMazeGenerator(config, palette=palette)
    map_data = generator.generate(seed=seed)
    start, goal = generator.sample_start_goal(map_data, seed=seed)

    inflation_radius_cells = config.inflation_radius_m / config.resolution
    if not cells_connected(map_data.occ_grid,
                           map_data.start_cell,
                           map_data.end_cell,
                           inflation_radius_cells):
        raise RuntimeError("Generated maze start/goal are not connected after inflation.")

    return map_data, start, goal

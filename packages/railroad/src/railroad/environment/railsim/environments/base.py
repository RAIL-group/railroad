"""Shared types for procedural environment generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..palette import Palette, resolve_palette
from ..pose import Pose


@dataclass(frozen=True)
class MapData:
    """Output of a procedural map generator.

    ``occ_grid`` follows the railsim convention: ``grid[i, j]`` maps to the
    world point ``(x = i * resolution, y = j * resolution)``; 1.0 is
    occupied, 0.0 is free.
    """

    occ_grid: np.ndarray
    semantic_grid: np.ndarray
    semantic_labels: dict[str, int]
    start_cell: tuple[int, int]
    end_cell: tuple[int, int]
    resolution: float
    # Tables as (x, y, size_x, size_y) in meters; occupied in occ_grid but
    # rendered as table boxes rather than walls.
    tables: tuple[tuple[float, float, float, float], ...] = ()
    # Semantic label names whose adjacent walls get a class-specific color
    # (palette key 'wall_<name>'); empty = all walls use palette['wall'].
    wall_class_labels: tuple[str, ...] = ()
    palette: Palette = field(default_factory=resolve_palette)
    # Default minimum distance between ceiling lights for worlds built from
    # this map (overridable via OccupancyGridWorld's argument).
    min_interlight_distance: float = 3.0


class MapGenerator(Protocol):
    """Interface for map generators (guided maze now; office et al. later)."""

    def generate(self, seed: int | None = None) -> MapData: ...

    def sample_start_goal(self, map_data: MapData, seed: int | None = None) -> tuple[Pose, Pose]:
        """Return (start, goal) poses in meters with randomized yaw."""
        ...

"""Frontier extraction and filtering for unknown-space exploration."""

from __future__ import annotations

import numpy as np
import scipy.ndimage

from railroad.navigation.constants import FREE_VAL, OBSTACLE_THRESHOLD, UNOBSERVED_VAL
from railroad.navigation.pathing import build_traversal_costs
from .types import Frontier


def extract_frontiers(observed_grid: np.ndarray) -> list[Frontier]:
    """Extract frontier regions from an observed occupancy grid.

    A frontier cell is a free observed cell adjacent to at least one
    unknown cell (8-connected neighbourhood).

    Returns:
        List of Frontier named-tuples sorted by deterministic ID.
    """
    free_mask = (observed_grid >= FREE_VAL) & (observed_grid < OBSTACLE_THRESHOLD)
    unknown_mask = observed_grid == UNOBSERVED_VAL

    structure = np.ones((3, 3))
    dilated_unknown = scipy.ndimage.binary_dilation(unknown_mask, structure=structure)

    frontier_mask = free_mask & dilated_unknown
    if not frontier_mask.any():
        return []

    labels, num_components = scipy.ndimage.label(frontier_mask, structure=structure)

    frontiers: list[Frontier] = []
    for component_id in range(1, num_components + 1):
        component_mask = labels == component_id
        cell_coords = np.argwhere(component_mask)  # Nx2 (row, col)

        # Compute centroid and snap to nearest member cell
        centroid = cell_coords.mean(axis=0)
        distances = np.sum((cell_coords - centroid) ** 2, axis=1)
        nearest_idx = int(np.argmin(distances))
        snap_row = int(cell_coords[nearest_idx, 0])
        snap_col = int(cell_coords[nearest_idx, 1])

        frontier_id = f"frontier_{snap_row}_{snap_col}"
        cells = cell_coords.T  # 2xN

        frontiers.append(Frontier(
            id=frontier_id,
            centroid_row=snap_row,
            centroid_col=snap_col,
            cells=cells,
        ))

    frontiers.sort(key=lambda f: f.id)
    return frontiers


def filter_reachable_frontiers(
    frontiers: list[Frontier],
    observed_grid: np.ndarray,
    robot_positions: list[tuple[int, int]],
) -> list[Frontier]:
    """Keep only frontiers reachable from at least one robot.

    Uses the traversal cost grid with unknown cells treated as obstacles.
    A frontier is reachable if a finite-cost path exists from any robot
    to the frontier centroid.
    """
    if not frontiers or not robot_positions:
        return []

    costs = build_traversal_costs(
        observed_grid,
        use_soft_cost=False,
        unknown_as_obstacle=True,
    )

    # Force robot cells traversable so they are valid sources
    for r, c in robot_positions:
        if 0 <= r < costs.shape[0] and 0 <= c < costs.shape[1]:
            costs[r, c] = 1.0

    reachable: list[Frontier] = []
    for frontier in frontiers:
        fr, fc = frontier.centroid_row, frontier.centroid_col
        if not (0 <= fr < costs.shape[0] and 0 <= fc < costs.shape[1]):
            continue
        if np.isinf(costs[fr, fc]):
            continue

        # Quick check: can any robot reach this frontier?
        # Use the cost grid directly — if frontier cell has finite cost
        # and a robot cell has finite cost, they may be connected.
        # For a proper check we'd need per-robot flood fill, but since
        # both robot and frontier are in the finite-cost region, check
        # label connectivity in the finite-cost mask.
        for rr, rc in robot_positions:
            if not (0 <= rr < costs.shape[0] and 0 <= rc < costs.shape[1]):
                continue
            if np.isinf(costs[rr, rc]):
                continue
            # Both cells have finite cost — check if they share a
            # connected component in the traversable region.
            reachable.append(frontier)
            break

    # Verify actual connectivity via connected-component labeling
    if reachable:
        traversable = ~np.isinf(costs)
        labels, _ = scipy.ndimage.label(traversable, structure=np.ones((3, 3)))
        robot_labels = set()
        for rr, rc in robot_positions:
            if 0 <= rr < labels.shape[0] and 0 <= rc < labels.shape[1]:
                lbl = labels[rr, rc]
                if lbl > 0:
                    robot_labels.add(lbl)

        reachable = [
            f for f in reachable
            if labels[f.centroid_row, f.centroid_col] in robot_labels
        ]

    return reachable

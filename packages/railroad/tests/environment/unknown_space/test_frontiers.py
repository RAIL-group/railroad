"""Tests 2-3: Frontier extraction and lifecycle."""

import numpy as np

from railroad.navigation.constants import (
    COLLISION_VAL,
    FREE_VAL,
    UNOBSERVED_VAL,
)
from railroad.environment.unknown_space.frontiers import (
    extract_frontiers,
    filter_reachable_frontiers,
)


def _partially_observed_grid() -> np.ndarray:
    """Create a 20x20 grid with known left half and unknown right half."""
    grid = UNOBSERVED_VAL * np.ones((20, 20))
    # Left half observed (free interior, walls on boundary)
    grid[:, :10] = FREE_VAL
    grid[0, :10] = COLLISION_VAL
    grid[-1, :10] = COLLISION_VAL
    grid[:, 0] = COLLISION_VAL
    return grid


def test_frontier_extraction_deterministic_ids():
    """Frontier IDs are deterministic for a fixed grid."""
    grid = _partially_observed_grid()
    frontiers = extract_frontiers(grid)

    assert len(frontiers) > 0, "Should find at least one frontier"

    # IDs should be deterministic
    ids_first = [f.id for f in frontiers]
    frontiers_again = extract_frontiers(grid)
    ids_second = [f.id for f in frontiers_again]
    assert ids_first == ids_second, "Frontier IDs should be deterministic"

    # IDs should follow the frontier_<row>_<col> pattern
    for f in frontiers:
        assert f.id.startswith("frontier_")
        parts = f.id.split("_")
        assert len(parts) == 3
        assert parts[1].isdigit() and parts[2].isdigit()


def test_frontier_cells_are_at_boundary():
    """Frontier cells should be free cells adjacent to unknown space."""
    grid = _partially_observed_grid()
    frontiers = extract_frontiers(grid)

    for f in frontiers:
        for i in range(f.cells.shape[1]):
            r, c = f.cells[0, i], f.cells[1, i]
            # Cell must be free
            assert grid[r, c] >= FREE_VAL and grid[r, c] < 0.5, (
                f"Frontier cell ({r},{c}) has value {grid[r, c]}, not free"
            )
            # Must have at least one unknown 8-neighbor
            has_unknown_neighbor = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        if grid[nr, nc] == UNOBSERVED_VAL:
                            has_unknown_neighbor = True
            assert has_unknown_neighbor, (
                f"Frontier cell ({r},{c}) has no unknown neighbor"
            )


def test_frontiers_empty_when_fully_observed():
    """No frontiers when the grid is fully observed."""
    grid = FREE_VAL * np.ones((10, 10))
    grid[0, :] = COLLISION_VAL
    grid[-1, :] = COLLISION_VAL
    grid[:, 0] = COLLISION_VAL
    grid[:, -1] = COLLISION_VAL

    frontiers = extract_frontiers(grid)
    assert len(frontiers) == 0, "Fully observed grid should have no frontiers"


def test_reachability_filter_removes_disconnected():
    """Frontiers unreachable from any robot are filtered out."""
    grid = UNOBSERVED_VAL * np.ones((20, 20))
    # Two disconnected observed regions:
    # Region A (top-left): rows 1-5, cols 1-5 (free), with unknown to the right
    grid[1:6, 1:6] = FREE_VAL
    # Region B (bottom-right): rows 14-18, cols 14-18 (free), with unknown to the left
    grid[14:19, 14:19] = FREE_VAL
    # Add walls around both to separate them
    grid[0, :] = COLLISION_VAL
    grid[-1, :] = COLLISION_VAL
    grid[:, 0] = COLLISION_VAL
    grid[:, -1] = COLLISION_VAL

    frontiers = extract_frontiers(grid)
    assert len(frontiers) >= 2, f"Expected at least 2 frontiers, got {len(frontiers)}"

    # Robot is only in region A
    robot_positions = [(3, 3)]
    reachable = filter_reachable_frontiers(frontiers, grid, robot_positions)

    # Only frontiers in region A should remain
    for f in reachable:
        # Centroid should be in or near region A (rows 1-6, cols 1-6)
        assert f.centroid_row <= 8 and f.centroid_col <= 8, (
            f"Frontier {f.id} at ({f.centroid_row},{f.centroid_col}) "
            "should not be reachable from robot at (3,3)"
        )

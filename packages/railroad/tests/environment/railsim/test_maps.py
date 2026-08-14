"""Tests for RailsimScene wrappers (no OpenGL required)."""

from __future__ import annotations

import numpy as np
import pytest

from railroad.environment.railsim import (
    GuidedMazeConfig,
    OfficeConfig,
    RailsimScene,
)
from railroad.environment.types import Pose

SMALL_MAZE = GuidedMazeConfig(width_cells=3, height_cells=3)
SMALL_OFFICE = OfficeConfig(
    grid_size=(150, 100), num_hallways=2, min_start_goal_separation_cells=30
)


@pytest.fixture(scope="module")
def maze_scene() -> RailsimScene:
    return RailsimScene.maze(seed=13, config=SMALL_MAZE)


@pytest.fixture(scope="module")
def office_scene() -> RailsimScene:
    return RailsimScene.office(seed=7, config=SMALL_OFFICE)


@pytest.mark.parametrize("scene_name", ["maze_scene", "office_scene"])
def test_scene_provides_navigation_grid(scene_name: str, request) -> None:  # noqa: ANN001
    scene = request.getfixturevalue(scene_name)
    assert scene.grid.shape == scene.raw_grid.shape
    assert set(np.unique(scene.grid)) <= {0.0, 1.0}
    # Inflation only adds obstacles.
    assert (scene.grid >= 0.5).sum() > (scene.raw_grid >= 0.5).sum()
    assert ((scene.raw_grid >= 0.5) & (scene.grid < 0.5)).sum() == 0


@pytest.mark.parametrize("scene_name", ["maze_scene", "office_scene"])
def test_scene_locations_and_objects(scene_name: str, request) -> None:  # noqa: ANN001
    scene = request.getfixturevalue(scene_name)
    assert set(scene.locations) == {"start_loc", "goal_loc"}
    # Start and goal stay free in the inflated navigation grid.
    for cell in scene.locations.values():
        assert scene.grid[cell] < 0.5
    assert scene.object_locations == {}


def test_cell_pose_to_meters(maze_scene: RailsimScene) -> None:
    pose = Pose(10.0, 20.0, 1.25)
    sim_pose = maze_scene.cell_pose_to_meters(pose)
    res = maze_scene.resolution
    assert sim_pose.x == pytest.approx(10.0 * res)
    assert sim_pose.y == pytest.approx(20.0 * res)
    assert sim_pose.yaw == pytest.approx(1.25)


def test_zero_inflation_preserves_raw_grid() -> None:
    scene = RailsimScene.maze(seed=13, config=SMALL_MAZE, inflation_radius_m=0.0)
    np.testing.assert_array_equal(scene.grid, scene.raw_grid)


def test_top_down_image(maze_scene: RailsimScene) -> None:
    image = maze_scene.get_top_down_image(orthographic=True)
    assert image.shape == (*maze_scene.raw_grid.shape, 3)
    assert image.dtype == np.uint8


def test_top_down_view_is_transposed_to_match_the_plot(maze_scene: RailsimScene) -> None:
    """``get_top_down_image`` is indexed [x, y]; the plot draws ``grid.T``.

    That mismatch was invisible while the image had its own subplot. Drawn
    underneath the trajectory it would render a mirrored map, so the view
    transposes and the test pins which way round it ends up.
    """
    view = maze_scene.get_top_down_view()
    n_x, n_y = maze_scene.raw_grid.shape
    assert view.image.shape == (n_y, n_x, 3)

    raw = maze_scene.get_top_down_image()
    occupied = np.argwhere(maze_scene.raw_grid >= 0.5)
    free = np.argwhere(maze_scene.raw_grid < 0.5)
    for i, j in (occupied[len(occupied) // 2], free[len(free) // 2]):
        np.testing.assert_array_equal(view.image[j, i], raw[i, j])


def test_top_down_view_covers_exactly_the_grid(maze_scene: RailsimScene) -> None:
    """One pixel per cell, so it lands on the extent imshow infers for the grid."""
    view = maze_scene.get_top_down_view()
    n_x, n_y = maze_scene.grid.shape
    assert view.extent == pytest.approx((-0.5, n_x - 0.5, n_y - 0.5, -0.5))


def test_seeded_generation_is_deterministic() -> None:
    a = RailsimScene.maze(seed=99, config=SMALL_MAZE)
    b = RailsimScene.maze(seed=99, config=SMALL_MAZE)
    np.testing.assert_array_equal(a.raw_grid, b.raw_grid)
    assert a.locations == b.locations

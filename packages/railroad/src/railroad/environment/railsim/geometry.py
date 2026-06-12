"""Occupancy grid -> shapely polygon conversion (shapely 2.x)."""

from __future__ import annotations

import itertools

import numpy as np
import shapely
import shapely.geometry
import shapely.ops


def full_simplify_shapely_polygon(
        poly: shapely.geometry.base.BaseGeometry) -> shapely.geometry.base.BaseGeometry:
    """Simplify a polygon, removing any colinear points. Shapely's built-in
    simplify won't remove the ring's start point even if it's colinear."""

    if isinstance(poly, (shapely.geometry.MultiPolygon, shapely.geometry.GeometryCollection)):
        return shapely.geometry.MultiPolygon(
            [full_simplify_shapely_polygon(p) for p in poly.geoms])

    poly = poly.simplify(0.001, preserve_topology=True)
    # The final point is removed, since shapely will auto-close the polygon.
    points = np.array(poly.exterior.coords)
    if (points[-1] == points[0]).all():
        points = points[:-1]

    def is_colinear(p1, p2, p3, tol=1e-6):
        return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] *
                   (p1[1] - p2[1])) < tol

    if is_colinear(points[0], points[1], points[-1]):
        poly = shapely.geometry.Polygon(points[1:])

    return poly


def convert_grid_to_poly(grid: np.ndarray,
                         resolution: float,
                         do_full_simplify: bool = True) -> shapely.geometry.base.BaseGeometry:
    """Union the cells of a boolean/float grid (values >= 0.5) into a polygon.

    Follows the railsim convention: ``grid[i, j]`` covers the square centered
    on the world point ``(i * resolution, j * resolution)``.
    """
    r = resolution
    indices = np.argwhere(grid >= 0.5)
    xs = indices[:, 0] * r - 0.5 * r
    ys = indices[:, 1] * r - 0.5 * r
    polys = [
        shapely.geometry.box(x, y, x + r, y + r).buffer(0.001 * r, quad_segs=0)
        for x, y in zip(xs, ys)
    ]
    joined_poly = shapely.union_all(polys)

    if do_full_simplify:
        return full_simplify_shapely_polygon(joined_poly)
    else:
        return joined_poly


def obstacles_and_boundary_from_occupancy_grid(
        grid: np.ndarray,
        resolution: float) -> tuple[list[shapely.geometry.Polygon], shapely.geometry.Polygon]:
    """Convert a *free-space* mask (1 = free) into (obstacles, boundary).

    The boundary is the polygon enclosing all free space; obstacles are the
    holes inside it (plus, for multi-part free space, smaller components'
    boundaries — matching the old behavior).
    """
    known_space_poly = convert_grid_to_poly(grid, resolution, do_full_simplify=False)

    def get_obstacles(poly):
        if isinstance(poly, shapely.geometry.MultiPolygon):
            return list(
                itertools.chain.from_iterable(
                    [get_obstacles(p) for p in poly.geoms]))

        obstacles = [
            full_simplify_shapely_polygon(shapely.geometry.Polygon(interior))
            for interior in list(poly.interiors)
        ]
        obstacles.append(full_simplify_shapely_polygon(poly))
        return obstacles

    obs = get_obstacles(known_space_poly)
    obs.sort(key=lambda x: x.area, reverse=True)
    return obs[1:], obs[0]

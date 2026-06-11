"""Triangle-mesh construction from a `World` (no OpenGL dependencies)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .palette import DEFAULT_PALETTE, Color, Palette, resolve_palette
from .world import World

# Unity-material defaults (sRGB; the shader decodes to linear). Kept as
# module constants for convenience; the values live in `railsim.palette`.
WALL_COLOR = DEFAULT_PALETTE['wall']
FLOOR_COLOR = DEFAULT_PALETTE['floor']
CEILING_COLOR = DEFAULT_PALETTE['ceiling']
BREADCRUMB_COLOR = DEFAULT_PALETTE['breadcrumb']
LIGHT_FIXTURE_COLOR = DEFAULT_PALETTE['light_fixture']
TABLE_COLOR = DEFAULT_PALETTE['table']

# The Unity breadcrumb prefab is the built-in Plane (10x10 units) at scale
# 0.1: a flat 1 m square lying on the floor. Many of them overlap along the
# goal path, painting a green trail.
BREADCRUMB_SIZE = 1.0
# Unity light prefab fixture: built-in Cylinder (radius 0.5, height 2)
# scaled (0.25, 0.1, 0.25) and centered on the ceiling, so a 0.125 m-radius
# can whose lower 0.1 m hangs below the ceiling.
LIGHT_FIXTURE_RADIUS = 0.125
LIGHT_FIXTURE_DROP = 0.1
# The Unity table prefab's "top" is the built-in Cube at y 0.8 scaled 1.6:
# a solid block over the full footprint (its corner legs are hidden inside).
TABLE_HEIGHT = 1.6


@dataclass(frozen=True)
class SceneConfig:
    wall_height: float = 2.8
    floor_margin: float = 1.0  # floor/ceiling extend past map bounds by this


@dataclass
class Mesh:
    """Non-indexed triangle soup; (N, 3) float32 arrays, N divisible by 3."""

    positions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    normals: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    colors: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    # Vertices [0:shadow_caster_vertices] cast shadows; None = all of them.
    # (Unity disabled shadow casting on breadcrumbs and light fixtures.)
    shadow_caster_vertices: int | None = None

    @staticmethod
    def concatenate(meshes: list["Mesh"]) -> "Mesh":
        return Mesh(
            positions=np.concatenate([m.positions for m in meshes]).astype(np.float32),
            normals=np.concatenate([m.normals for m in meshes]).astype(np.float32),
            colors=np.concatenate([m.colors for m in meshes]).astype(np.float32),
        )

    def interleaved(self) -> np.ndarray:
        """(N, 9) position|normal|color array for VBO upload."""
        return np.hstack([self.positions, self.normals, self.colors]).astype(np.float32)


def _quad(p0, p1, p2, p3, normal, color) -> Mesh:
    """Two triangles spanning the (planar) quad p0-p1-p2-p3."""
    positions = np.array([p0, p1, p2, p0, p2, p3], dtype=np.float32)
    normals = np.tile(np.asarray(normal, dtype=np.float32), (6, 1))
    colors = np.tile(np.asarray(color, dtype=np.float32), (6, 1))
    return Mesh(positions, normals, colors)


class _WallColorClassifier:
    """Assigns wall colors by the semantic label of the adjacent free cell.

    The old sim split the free-space boundary by buffered semantic-class
    polygons and assigned a Unity material per class (poly_hallway /
    poly_room / poly_base); sampling the semantic grid just inside the free
    side of each wall reproduces that classification.
    """

    def __init__(self, world: World, palette: Palette):
        self.base_color = palette['wall']
        self.map_data = getattr(world, 'map_data', None)
        self.color_by_label: dict[int, Color] = {}
        if self.map_data is not None:
            for name in self.map_data.wall_class_labels:
                value = self.map_data.semantic_labels[name]
                self.color_by_label[value] = palette.get(f'wall_{name}',
                                                         self.base_color)

    @property
    def active(self) -> bool:
        return bool(self.color_by_label)

    def color_at(self, mid_x: float, mid_y: float,
                 normal_x: float, normal_y: float) -> Color:
        """Color for a wall chunk centered at (mid_x, mid_y); the free cell
        is sampled just off the wall on whichever side is free space."""
        assert self.map_data is not None
        res = self.map_data.resolution
        occ = self.map_data.occ_grid
        sem = self.map_data.semantic_grid
        for sign in (1.0, -1.0):
            i = int(round((mid_x + sign * 0.45 * res * normal_x) / res))
            j = int(round((mid_y + sign * 0.45 * res * normal_y) / res))
            i = min(max(i, 0), occ.shape[0] - 1)
            j = min(max(j, 0), occ.shape[1] - 1)
            if occ[i, j] < 0.5:
                return self.color_by_label.get(int(sem[i, j]), self.base_color)
        return self.base_color


def _wall_quads_from_ring(coords, wall_height: float, color,
                          classifier: _WallColorClassifier | None = None,
                          max_chunk: float = float('inf')) -> list[Mesh]:
    """Vertical quads (z=0 to wall_height) along a closed coordinate ring.

    With a ``classifier``, each edge is walked in chunks of at most
    ``max_chunk`` and consecutive same-color chunks are merged into one
    quad (polygon simplification can merge wall runs that border different
    semantic classes, so color can change mid-edge).
    """
    meshes = []
    pts = list(coords)
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        length = float(np.hypot(xb - xa, yb - ya))
        if length < 1e-9:
            continue
        # Horizontal normal perpendicular to the edge; orientation is
        # irrelevant because the shader shades two-sided.
        normal = ((yb - ya) / length, -(xb - xa) / length, 0.0)

        if classifier is None or not classifier.active:
            meshes.append(_quad((xa, ya, 0.0), (xb, yb, 0.0),
                                (xb, yb, wall_height), (xa, ya, wall_height),
                                normal, color))
            continue

        n_chunks = max(1, int(np.ceil(length / max_chunk)))
        ts = np.linspace(0.0, 1.0, n_chunks + 1)
        chunk_colors = []
        for k in range(n_chunks):
            tm = (ts[k] + ts[k + 1]) / 2
            chunk_colors.append(classifier.color_at(
                xa + tm * (xb - xa), ya + tm * (yb - ya),
                normal[0], normal[1]))
        # Merge consecutive same-color chunks back into single quads.
        run_start = 0
        for k in range(1, n_chunks + 1):
            if k == n_chunks or chunk_colors[k] != chunk_colors[run_start]:
                t0, t1 = ts[run_start], ts[k]
                x0, y0 = xa + t0 * (xb - xa), ya + t0 * (yb - ya)
                x1, y1 = xa + t1 * (xb - xa), ya + t1 * (yb - ya)
                meshes.append(_quad((x0, y0, 0.0), (x1, y1, 0.0),
                                    (x1, y1, wall_height), (x0, y0, wall_height),
                                    normal, chunk_colors[run_start]))
                run_start = k
    return meshes


def build_walls(world: World, config: SceneConfig,
                palette: Palette | None = None) -> Mesh:
    palette = resolve_palette(palette)
    classifier = _WallColorClassifier(world, palette)
    max_chunk = (classifier.map_data.resolution
                 if classifier.active and classifier.map_data is not None
                 else float('inf'))

    # Worlds with tables expose table-free wall polygons (tables render as
    # boxes, not full-height walls).
    wall_obstacles = getattr(world, 'wall_obstacles', None)
    if wall_obstacles is not None:
        polygons = list(wall_obstacles) + [getattr(world, 'wall_boundary',
                                                   world.boundary)]
    else:
        polygons = world.obstacles

    meshes: list[Mesh] = []
    for polygon in polygons:
        meshes.extend(
            _wall_quads_from_ring(polygon.exterior.coords, config.wall_height,
                                  palette['wall'], classifier, max_chunk))
        for interior in polygon.interiors:
            meshes.extend(
                _wall_quads_from_ring(interior.coords, config.wall_height,
                                      palette['wall'], classifier, max_chunk))
    return Mesh.concatenate(meshes)


def build_floor_ceiling(world: World, config: SceneConfig,
                        palette: Palette | None = None) -> Mesh:
    palette = resolve_palette(palette)
    (xmin, xmax), (ymin, ymax) = world.map_bounds
    m = config.floor_margin
    x0, x1 = xmin - m, xmax + m
    y0, y1 = ymin - m, ymax + m
    floor = _quad((x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0),
                  (0.0, 0.0, 1.0), palette['floor'])
    h = config.wall_height
    ceiling = _quad((x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h),
                    (0.0, 0.0, -1.0), palette['ceiling'])
    return Mesh.concatenate([floor, ceiling])


def build_tables(world: World, config: SceneConfig,
                 palette: Palette | None = None) -> Mesh:
    """Solid boxes for tables, matching the Unity table prefab: the full
    footprint extruded to 1.6 m (4 sides + top; no bottom face)."""
    tables = getattr(world, 'table_poses_sizes', [])
    if not tables:
        return Mesh()
    palette = resolve_palette(palette)

    meshes = []
    h = TABLE_HEIGHT
    color = palette['table']
    for x, y, size_x, size_y in tables:
        hx, hy = size_x / 2, size_y / 2
        x0, x1, y0, y1 = x - hx, x + hx, y - hy, y + hy
        meshes.append(_quad((x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h),
                            (0.0, 0.0, 1.0), color))
        meshes.append(_quad((x0, y0, 0), (x1, y0, 0), (x1, y0, h), (x0, y0, h),
                            (0.0, -1.0, 0.0), color))
        meshes.append(_quad((x1, y1, 0), (x0, y1, 0), (x0, y1, h), (x1, y1, h),
                            (0.0, 1.0, 0.0), color))
        meshes.append(_quad((x0, y1, 0), (x0, y0, 0), (x0, y0, h), (x0, y1, h),
                            (-1.0, 0.0, 0.0), color))
        meshes.append(_quad((x1, y0, 0), (x1, y1, 0), (x1, y1, h), (x1, y0, h),
                            (1.0, 0.0, 0.0), color))
    return Mesh.concatenate(meshes)


def build_breadcrumbs(world: World, config: SceneConfig,
                      palette: Palette | None = None) -> Mesh:
    """Flat axis-aligned squares on the floor, one per breadcrumb pose.

    The Unity sim lifted each plane by ``random() * 1 mm`` to avoid
    z-fighting between overlapping crumbs; we do the same with a
    deterministic golden-ratio sequence.
    """
    poses = getattr(world, 'breadcrumb_element_poses', [])
    if not poses:
        return Mesh()
    palette = resolve_palette(palette)

    h = BREADCRUMB_SIZE / 2
    template = np.array([(-h, -h, 0), (h, -h, 0), (h, h, 0),
                         (-h, -h, 0), (h, h, 0), (-h, h, 0)])
    heights = (np.arange(1, len(poses) + 1) * 0.6180339887) % 1.0 * 1e-3
    centers = np.array([[p.x, p.y, z] for p, z in zip(poses, heights)])
    positions = (template[None, :, :] + centers[:, None, :]).reshape(-1, 3)
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32),
                      (len(positions), 1))
    colors = np.tile(np.asarray(palette['breadcrumb'], dtype=np.float32),
                     (len(positions), 1))
    return Mesh(positions.astype(np.float32), normals, colors)


def build_light_fixtures(world: World, config: SceneConfig,
                         palette: Palette | None = None) -> Mesh:
    """Dark cylindrical cans hanging below the ceiling at each light pose
    (the illumination itself comes from the renderer's lights). The
    above-ceiling half of the Unity cylinder is clipped away: it is never
    visible from inside."""
    poses = getattr(world, 'light_poses', [])
    if not poses:
        return Mesh()
    palette = resolve_palette(palette)

    n_sides = 16
    angles = np.linspace(0.0, 2 * np.pi, n_sides + 1)
    rim = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    z1 = config.wall_height
    z0 = z1 - LIGHT_FIXTURE_DROP
    r = LIGHT_FIXTURE_RADIUS

    positions = []
    normals = []
    for x, y in poses:
        for k in range(n_sides):
            (ca, sa), (cb, sb) = rim[k], rim[k + 1]
            a0 = (x + r * ca, y + r * sa, z0)
            a1 = (x + r * ca, y + r * sa, z1)
            b0 = (x + r * cb, y + r * sb, z0)
            b1 = (x + r * cb, y + r * sb, z1)
            positions.extend([a0, b0, b1, a0, b1, a1])
            normals.extend([(ca, sa, 0.0), (cb, sb, 0.0), (cb, sb, 0.0),
                            (ca, sa, 0.0), (cb, sb, 0.0), (ca, sa, 0.0)])
            positions.extend([(x, y, z0), b0, a0])
            normals.extend([(0.0, 0.0, -1.0)] * 3)
    pos = np.array(positions, dtype=np.float32)
    colors = np.tile(np.asarray(palette['light_fixture'], dtype=np.float32),
                     (len(pos), 1))
    return Mesh(pos, np.array(normals, dtype=np.float32), colors)


def build_scene(world: World, config: SceneConfig | None = None) -> Mesh:
    """Build the full static scene mesh for a world, using the world's
    palette (`world.palette`, set from MapData) when present."""
    config = config or SceneConfig()
    palette = resolve_palette(getattr(world, 'palette', None))
    # Only walls and tables go into the shadow maps: the floor and ceiling
    # cannot occlude lights that hang between them, and rendering those
    # large planes at grazing angles into the point-light faces causes
    # self-shadowing acne. (Unity also disabled casting on breadcrumbs and
    # light fixtures.)
    casters = Mesh.concatenate([
        build_walls(world, config, palette),
        build_tables(world, config, palette),
    ])
    mesh = Mesh.concatenate([
        casters,
        build_floor_ceiling(world, config, palette),
        build_breadcrumbs(world, config, palette),
        build_light_fixtures(world, config, palette),
    ])
    mesh.shadow_caster_vertices = casters.positions.shape[0]
    return mesh

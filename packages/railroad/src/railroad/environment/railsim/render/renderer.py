"""Offscreen renderer: one draw call, two outputs (color + distance)."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

import moderngl

from .camera import look_at, perspective
from .context import make_current
from .shaders import MAX_LIGHTS, VERTEX_SHADER, fragment_shader
from ..scene import Mesh

# Unity light prefab: warm white, sRGB.
DEFAULT_LIGHT_COLOR = (1.0, 0.9569, 0.8392)


@dataclass(frozen=True)
class LightRig:
    """Per-fixture light parameters, defaulting to the Unity ``light-urp``
    prefab: a shadowless downward spot 5 cm below the ceiling anchor plus a
    bare point light 0.8 m below it, sharing one warm-white color (sRGB).
    Angles are full cone angles in degrees."""

    color: tuple[float, float, float] = DEFAULT_LIGHT_COLOR
    spot_drop: float = 0.05
    spot_intensity: float = 8.0
    spot_range: float = 4.0
    spot_angle_deg: float = 86.505486
    spot_inner_angle_deg: float = 43.863358
    point_drop: float = 0.8
    point_intensity: float = 3.0
    point_range: float = 5.0


class Renderer:
    """Renders a static scene mesh from arbitrary viewpoints.

    ``render_view`` returns ``(rgb, dist)`` where ``rgb`` is uint8 (H, W, 3)
    and ``dist`` is float32 (H, W) Euclidean view distance in meters.
    Pixels where nothing was drawn read as ``far``.

    If ``lights`` (an (N, 3) array of fixture anchors, typically on the
    ceiling) is given, the scene is lit by a spot + point pair per anchor
    (see :class:`LightRig`) plus ambient; otherwise a camera-attached
    headlight is used.

    With ``shadows`` (the default), shadow maps are baked once at
    construction (the scene and lights are static), so lights no longer
    shine through walls: one downward tile per spot, and -- unless
    ``point_shadows`` is disabled for strict Unity parity (Unity's point
    lights cast no shadows) -- a 5-face mini-cubemap per point light.
    ``tonemap`` applies URP's Neutral tonemapping instead of hard-clipping
    bright values.
    """

    def __init__(self, ctx: moderngl.Context, mesh: Mesh,
                 near: float = 0.3, far: float = 100.0, ambient: float = 0.35,
                 lights: np.ndarray | None = None,
                 light_rig: LightRig | None = None,
                 shadows: bool = True,
                 shadow_map_size: int = 256,
                 shadow_bias: float = 0.05,
                 point_shadows: bool = True,
                 point_shadow_map_size: int = 128,
                 tonemap: bool = True):
        self.ctx = ctx
        make_current(ctx)
        self.near = near
        self.far = far
        self.ambient = ambient
        self._shadow_atlas: moderngl.Texture | None = None
        self._point_atlas: moderngl.Texture | None = None

        if lights is None or len(lights) == 0:
            num_lights = 0
            lights = None
        else:
            lights = np.asarray(lights, dtype=np.float32).reshape(-1, 3)
            if len(lights) > MAX_LIGHTS:
                warnings.warn(f"{len(lights)} lights exceeds MAX_LIGHTS="
                              f"{MAX_LIGHTS}; extra lights are dropped.")
                lights = lights[:MAX_LIGHTS]
            num_lights = len(lights)

        # The light array is compiled at the scene's actual light count:
        # uniform components are scarce (typically 4096, with vec3 array
        # elements padded to vec4) and office-scale maps need most of them.
        self.program = ctx.program(vertex_shader=VERTEX_SHADER,
                                   fragment_shader=fragment_shader(num_lights))
        self.vbo = ctx.buffer(mesh.interleaved().tobytes())
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])
        self._fbos: dict[tuple[int, int], moderngl.Framebuffer] = {}

        if lights is not None:
            self._uniform('u_light_positions').write(
                lights.astype(np.float32).tobytes())

        rig = light_rig or LightRig()
        self._uniform('u_num_lights').value = num_lights
        self._uniform('u_headlight').value = 0.0 if num_lights else 1.0
        # Light color is given in sRGB; the shader works in linear.
        self._uniform('u_light_color').value = tuple(
            float(c)**2.2 for c in rig.color)
        self._uniform('u_spot_drop').value = rig.spot_drop
        self._uniform('u_spot_intensity').value = rig.spot_intensity
        self._uniform('u_spot_range').value = rig.spot_range
        self._uniform('u_spot_cos').value = (
            math.cos(math.radians(rig.spot_inner_angle_deg) / 2),
            math.cos(math.radians(rig.spot_angle_deg) / 2))
        self._uniform('u_point_drop').value = rig.point_drop
        self._uniform('u_point_intensity').value = rig.point_intensity
        self._uniform('u_point_range').value = rig.point_range
        self._uniform('u_ambient').value = self.ambient
        self._uniform('u_tonemap').value = 1.0 if tonemap else 0.0

        self._uniform('u_shadows').value = 0.0
        self._uniform('u_shadow_atlas').value = 2
        self._uniform('u_point_shadows').value = 0.0
        self._uniform('u_point_atlas').value = 3
        # Keep the samplers' texture units occupied even with shadows off;
        # an unbound sampler makes macOS GL warn (and is undefined).
        self._shadow_atlas = ctx.texture((1, 1), components=1, dtype='f4')
        self._shadow_atlas.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._point_atlas = ctx.texture((1, 1), components=1, dtype='f4')
        self._point_atlas.filter = (moderngl.NEAREST, moderngl.NEAREST)
        if shadows and num_lights:
            assert lights is not None
            self._bake_shadow_atlas(lights, rig, shadow_map_size,
                                    shadow_bias, mesh.shadow_caster_vertices)
            if point_shadows and rig.point_intensity > 0.0:
                self._bake_point_atlas(lights, rig, point_shadow_map_size,
                                       mesh.shadow_caster_vertices)

    # Faces of the point-light mini-cubemap, in the order the shader
    # expects: +x, -x, +y, -y, down. The up face is skipped: lights keep
    # >= 2 m of wall clearance and hang only 0.8 m below the ceiling, so
    # nothing ever occludes the up cone.
    _POINT_FACES = (((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
                    ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))

    def _render_distance_atlas(self, views, tile_size: int,
                               caster_vertices: int | None
                               ) -> tuple[moderngl.Texture, int, int]:
        """Render distance-from-eye tiles for every (eye, forward, up) view
        into one square-ish atlas. Reuses the main program: its second
        output is already the Euclidean distance from the eye."""
        n = len(views)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        size = (cols * tile_size, rows * tile_size)
        color = self.ctx.texture(size, components=4)
        atlas = self.ctx.texture(size, components=1, dtype='f4')
        atlas.filter = (moderngl.NEAREST, moderngl.NEAREST)
        depth = self.ctx.depth_renderbuffer(size)
        fbo = self.ctx.framebuffer(color_attachments=[color, atlas],
                                   depth_attachment=depth)
        fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        vertices = -1 if caster_vertices is None else caster_vertices
        for i, (eye, forward, up) in enumerate(views):
            viewport = ((i % cols) * tile_size, (i // cols) * tile_size,
                        tile_size, tile_size)
            self.render_into(fbo, np.asarray(eye), np.asarray(forward),
                             np.asarray(up), 90.0, tile_size, tile_size,
                             viewport=viewport, vertices=vertices)

        # Only the distance attachment is needed for sampling.
        fbo.release()
        color.release()
        depth.release()
        return atlas, cols, rows

    def _bake_shadow_atlas(self, anchors: np.ndarray, rig: LightRig,
                           tile_size: int, bias: float,
                           caster_vertices: int | None) -> None:
        """One downward tile per spot (a 90-degree camera looking straight
        down covers the 86.5-degree cone)."""
        down = (0.0, 0.0, -1.0)
        up = (0.0, 1.0, 0.0)
        views = [((x, y, z - rig.spot_drop), down, up)
                 for x, y, z in anchors]
        atlas, cols, rows = self._render_distance_atlas(
            views, tile_size, caster_vertices)

        if self._shadow_atlas is not None:
            self._shadow_atlas.release()
        self._shadow_atlas = atlas
        self._uniform('u_shadow_grid').value = (cols, rows)
        self._uniform('u_shadow_tan_half').value = 1.0  # tan(90 deg / 2)
        self._uniform('u_shadow_bias').value = bias
        self._uniform('u_shadows').value = 1.0

    def _bake_point_atlas(self, anchors: np.ndarray, rig: LightRig,
                          tile_size: int,
                          caster_vertices: int | None) -> None:
        """Five 90-degree faces per point light (see `_POINT_FACES`)."""
        views = [((x, y, z - rig.point_drop), forward, up)
                 for x, y, z in anchors
                 for forward, up in self._POINT_FACES]
        atlas, cols, rows = self._render_distance_atlas(
            views, tile_size, caster_vertices)

        if self._point_atlas is not None:
            self._point_atlas.release()
        self._point_atlas = atlas
        self._uniform('u_point_grid').value = (cols, rows)
        self._uniform('u_point_inset').value = 0.5 / tile_size
        self._uniform('u_point_shadows').value = 1.0

    def _uniform(self, name: str) -> moderngl.Uniform:
        member = self.program[name]
        assert isinstance(member, moderngl.Uniform)
        return member

    def _get_fbo(self, width: int, height: int) -> moderngl.Framebuffer:
        key = (width, height)
        if key not in self._fbos:
            color = self.ctx.texture((width, height), components=4)
            dist = self.ctx.texture((width, height), components=1, dtype='f4')
            depth = self.ctx.depth_renderbuffer((width, height))
            self._fbos[key] = self.ctx.framebuffer(
                color_attachments=[color, dist], depth_attachment=depth)
        return self._fbos[key]

    def render_into(self,
                    fbo: moderngl.Framebuffer,
                    eye: np.ndarray,
                    forward: np.ndarray,
                    up: np.ndarray,
                    fov_y_deg: float,
                    width: int,
                    height: int,
                    viewport: tuple[int, int, int, int] | None = None,
                    vertices: int = -1) -> None:
        """Draw the scene into (a viewport region of) an existing
        framebuffer without reading back."""
        make_current(self.ctx)
        eye = np.asarray(eye, dtype=np.float64)
        proj = perspective(fov_y_deg, width / height, self.near, self.far)
        view = look_at(eye, eye + np.asarray(forward, dtype=np.float64), up)
        view_proj = proj @ view

        fbo.use()
        fbo.viewport = viewport or (0, 0, width, height)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        if self._shadow_atlas is not None:
            self._shadow_atlas.use(location=2)
        if self._point_atlas is not None:
            self._point_atlas.use(location=3)

        # GLSL expects column-major; our matrices are row-major math convention.
        self._uniform('u_view_proj').write(view_proj.T.astype('f4').tobytes())
        self._uniform('u_camera_pos').value = tuple(eye)
        self.vao.render(moderngl.TRIANGLES, vertices=vertices)

    def render_view(self,
                    eye: np.ndarray,
                    forward: np.ndarray,
                    up: np.ndarray,
                    fov_y_deg: float,
                    width: int,
                    height: int) -> tuple[np.ndarray, np.ndarray]:
        make_current(self.ctx)
        fbo = self._get_fbo(width, height)
        fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.render_into(fbo, eye, forward, up, fov_y_deg, width, height)

        rgb_raw = fbo.read(components=3, attachment=0)
        rgb = np.frombuffer(rgb_raw, dtype=np.uint8).reshape(height, width, 3)
        dist_raw = fbo.read(components=1, attachment=1, dtype='f4')
        dist = np.frombuffer(dist_raw, dtype=np.float32).reshape(height, width)

        # fbo.read() is bottom-up; flip to image convention (row 0 = top).
        rgb = np.flipud(rgb).copy()
        dist = np.flipud(dist).copy()
        # The distance attachment clears to 0; report background as far.
        dist[dist == 0.0] = self.far
        return rgb, dist

    def release(self) -> None:
        make_current(self.ctx)
        for fbo in self._fbos.values():
            for attachment in fbo.color_attachments or []:
                attachment.release()
            if fbo.depth_attachment is not None:
                fbo.depth_attachment.release()
            fbo.release()
        self._fbos.clear()
        if self._shadow_atlas is not None:
            self._shadow_atlas.release()
            self._shadow_atlas = None
        if self._point_atlas is not None:
            self._point_atlas.release()
            self._point_atlas = None
        self.vao.release()
        self.vbo.release()
        self.program.release()

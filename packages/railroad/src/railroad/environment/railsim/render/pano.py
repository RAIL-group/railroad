"""Equirectangular panorama rendering, resampled entirely on the GPU.

Six square perspective views (90 degrees + one pixel of overscan so bilinear
taps never cross a face boundary) are rendered in the robot frame into a
3x2 atlas framebuffer -- no per-face readback. A fullscreen pass then maps
each pano pixel to a direction, selects the cube face, and samples the
atlas; only the final pano (color + distance) is read back.

Pano convention: the center column looks along the robot heading, and
moving right in the image turns clockwise (rightward), matching the
left/right sense of a perspective image. Rendering at yaw theta equals
``np.roll(pano_at_yaw0, +round(W * theta / 2pi), axis=1)``. Row 0 is
straight up.
"""

from __future__ import annotations

import math

import numpy as np

import moderngl

from .context import make_current
from .renderer import Renderer

# (forward, up) per face, in the robot frame (x forward, y left, z up).
# Face i occupies atlas cell (column i % 3, row i // 3), rows bottom-up.
_FACES = (
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),    # forward
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),    # left
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),   # back
    ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),   # right
    ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),    # up
    ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),   # down
)


def face_fov_deg(face_size: int) -> float:
    """Face FOV with one pixel of overscan beyond the nominal 90 degrees."""
    return 2.0 * math.degrees(math.atan((face_size + 2) / face_size))


_EQUIRECT_VERTEX = """
#version 330

out vec2 v_uv;

void main() {
    // Fullscreen triangle from gl_VertexID; no vertex buffer needed.
    vec2 pos = vec2[3](vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.))[gl_VertexID];
    v_uv = (pos + 1.0) * 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

_EQUIRECT_FRAGMENT = """
#version 330

uniform sampler2D u_atlas_color;
uniform sampler2D u_atlas_dist;
uniform float u_overscan_tan;   // tan(face_fov / 2)

in vec2 v_uv;

layout(location = 0) out vec4 out_color;
layout(location = 1) out float out_dist;

const float PI = 3.14159265358979;

void main() {
    // Longitude decreases with column: image-right = clockwise = the
    // robot's right. v_uv.y = 1 is the top row (straight up).
    float lon = PI - 2.0 * PI * v_uv.x;
    float lat = (v_uv.y - 0.5) * PI;
    vec3 d = vec3(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat));

    // Dominant axis selects the cube face; basis must match _FACES.
    float ax = abs(d.x), ay = abs(d.y), az = abs(d.z);
    int face;
    vec3 fwd, up;
    if (az >= ax && az >= ay) {
        if (d.z > 0.0) { face = 4; fwd = vec3(0., 0., 1.); }
        else           { face = 5; fwd = vec3(0., 0., -1.); }
        up = vec3(0., 1., 0.);
    } else if (ax >= ay) {
        if (d.x > 0.0) { face = 0; fwd = vec3(1., 0., 0.); }
        else           { face = 2; fwd = vec3(-1., 0., 0.); }
        up = vec3(0., 0., 1.);
    } else {
        if (d.y > 0.0) { face = 1; fwd = vec3(0., 1., 0.); }
        else           { face = 3; fwd = vec3(0., -1., 0.); }
        up = vec3(0., 0., 1.);
    }
    vec3 side = cross(fwd, up);

    float w = dot(d, fwd);
    vec2 s = vec2(dot(d, side), dot(d, up)) / (w * u_overscan_tan);

    // Position inside the face cell (GL bottom-up), then atlas UV.
    vec2 cell = vec2(float(face % 3), float(face / 3));
    vec2 uv = (cell + (s + 1.0) * 0.5) / vec2(3.0, 2.0);

    out_color = texture(u_atlas_color, uv);
    out_dist = texture(u_atlas_dist, uv).r;
}
"""


class PanoRenderer:
    """Renders equirectangular (rgb, dist) panoramas using a scene
    `Renderer` for the cube faces and a GPU resampling pass."""

    def __init__(self, ctx: moderngl.Context, renderer: Renderer,
                 pano_width: int, pano_height: int, face_size: int):
        self.ctx = ctx
        make_current(ctx)
        self.renderer = renderer
        self.pano_width = pano_width
        self.pano_height = pano_height
        self.face_size = face_size

        n = face_size
        self._atlas_color = ctx.texture((3 * n, 2 * n), components=4)
        self._atlas_dist = ctx.texture((3 * n, 2 * n), components=1, dtype='f4')
        self._atlas_color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._atlas_dist.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._atlas_depth = ctx.depth_renderbuffer((3 * n, 2 * n))
        self._atlas_fbo = ctx.framebuffer(
            color_attachments=[self._atlas_color, self._atlas_dist],
            depth_attachment=self._atlas_depth)

        self._pano_color = ctx.texture((pano_width, pano_height), components=4)
        self._pano_dist = ctx.texture((pano_width, pano_height), components=1,
                                      dtype='f4')
        self._pano_fbo = ctx.framebuffer(
            color_attachments=[self._pano_color, self._pano_dist])

        self._program = ctx.program(vertex_shader=_EQUIRECT_VERTEX,
                                    fragment_shader=_EQUIRECT_FRAGMENT)
        self._vao = ctx.vertex_array(self._program, [])
        member = self._program['u_overscan_tan']
        assert isinstance(member, moderngl.Uniform)
        member.value = (n + 2) / n
        for name, unit in (('u_atlas_color', 0), ('u_atlas_dist', 1)):
            member = self._program[name]
            assert isinstance(member, moderngl.Uniform)
            member.value = unit

    def render(self, eye: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        make_current(self.ctx)
        n = self.face_size
        fov = face_fov_deg(n)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        rot = np.array([[cos_y, -sin_y, 0.0],
                        [sin_y, cos_y, 0.0],
                        [0.0, 0.0, 1.0]])

        self._atlas_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        for i, (fwd, up) in enumerate(_FACES):
            viewport = ((i % 3) * n, (i // 3) * n, n, n)
            self.renderer.render_into(self._atlas_fbo, eye,
                                      rot @ np.asarray(fwd),
                                      rot @ np.asarray(up),
                                      fov, n, n, viewport=viewport)

        self._pano_fbo.use()
        self._pano_fbo.viewport = (0, 0, self.pano_width, self.pano_height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._atlas_color.use(location=0)
        self._atlas_dist.use(location=1)
        self._vao.render(moderngl.TRIANGLES, vertices=3)

        w, h = self.pano_width, self.pano_height
        rgb_raw = self._pano_fbo.read(components=3, attachment=0)
        rgb = np.frombuffer(rgb_raw, dtype=np.uint8).reshape(h, w, 3)
        dist_raw = self._pano_fbo.read(components=1, attachment=1, dtype='f4')
        dist = np.frombuffer(dist_raw, dtype=np.float32).reshape(h, w)

        rgb = np.flipud(rgb).copy()
        dist = np.flipud(dist).copy()
        dist[dist == 0.0] = self.renderer.far
        return rgb, dist

    def release(self) -> None:
        make_current(self.ctx)
        self._vao.release()
        self._program.release()
        self._pano_fbo.release()
        self._pano_color.release()
        self._pano_dist.release()
        self._atlas_fbo.release()
        self._atlas_color.release()
        self._atlas_dist.release()
        self._atlas_depth.release()

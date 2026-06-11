"""Public simulator API: render RGB/depth images (perspective and
panoramic) of a world from arbitrary poses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import TracebackType

import numpy as np

from .pose import Pose
from .render.context import create_gl_context, release_context
from .render.pano import PanoRenderer
from .render.renderer import LightRig, Renderer
from .scene import SceneConfig, build_scene
from .world import World


@dataclass(frozen=True)
class SimulatorConfig:
    width: int = 320
    height: int = 240
    fov_deg: float = 90.0  # vertical FOV, matching Unity's convention
    near: float = 0.3
    far: float = 100.0
    camera_height: float = 1.5
    pano_width: int = 512
    pano_height: int = 256
    pano_face_size: int = 256
    wall_height: float = 2.8
    ambient: float = 0.35
    light_rig: LightRig = LightRig()
    shadows: bool = True
    shadow_map_size: int = 256
    # Point-light shadows are a realism extension; Unity's point lights
    # cast none, so set point_shadows=False for strict parity.
    point_shadows: bool = True
    point_shadow_map_size: int = 128
    tonemap: bool = True
    gl_backend: str | None = None


class Simulator:
    """Offscreen renderer for a static `World`.

    The OpenGL context is created lazily on the first ``get_*`` call, so a
    Simulator can be constructed on machines without GL. Contexts are
    thread-affine: use a Simulator from a single thread. Supports use as a
    context manager; call :meth:`release` when done otherwise.
    """

    def __init__(self, world: World, config: SimulatorConfig | None = None):
        self.world = world
        self.config = config or SimulatorConfig()
        self._ctx = None
        self._renderer: Renderer | None = None
        self._pano_renderer: PanoRenderer | None = None

    def _ensure_renderer(self) -> Renderer:
        if self._renderer is None:
            cfg = self.config
            self._ctx = create_gl_context(cfg.gl_backend)
            mesh = build_scene(self.world,
                               SceneConfig(wall_height=cfg.wall_height))
            light_xy = getattr(self.world, 'light_poses', [])
            lights = None
            if light_xy:
                # Fixture anchors sit on the ceiling; the LightRig hangs the
                # spot and point lights below each anchor.
                lights = np.array([[x, y, cfg.wall_height]
                                   for x, y in light_xy])
            self._renderer = Renderer(self._ctx, mesh, near=cfg.near,
                                      far=cfg.far, ambient=cfg.ambient,
                                      lights=lights,
                                      light_rig=cfg.light_rig,
                                      shadows=cfg.shadows,
                                      shadow_map_size=cfg.shadow_map_size,
                                      point_shadows=cfg.point_shadows,
                                      point_shadow_map_size=cfg.point_shadow_map_size,
                                      tonemap=cfg.tonemap)
        return self._renderer

    def _eye_forward(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        eye = np.array([pose.x, pose.y, self.config.camera_height])
        forward = np.array([math.cos(pose.yaw), math.sin(pose.yaw), 0.0])
        return eye, forward

    def _render(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        renderer = self._ensure_renderer()
        eye, forward = self._eye_forward(pose)
        cfg = self.config
        return renderer.render_view(eye, forward, np.array([0.0, 0.0, 1.0]),
                                    cfg.fov_deg, cfg.width, cfg.height)

    def get_image(self, pose: Pose) -> np.ndarray:
        """Perspective RGB image, uint8 (height, width, 3)."""
        return self._render(pose)[0]

    def get_depth_image(self, pose: Pose) -> np.ndarray:
        """Perspective depth image, float32 (height, width), Euclidean view
        distance in meters (background reads as ``config.far``)."""
        return self._render(pose)[1]

    def get_image_and_depth(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        """Both perspective outputs from a single render pass."""
        return self._render(pose)

    def _render_pano(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        renderer = self._ensure_renderer()
        cfg = self.config
        if self._pano_renderer is None:
            assert self._ctx is not None
            self._pano_renderer = PanoRenderer(self._ctx, renderer,
                                               cfg.pano_width,
                                               cfg.pano_height,
                                               cfg.pano_face_size)
        eye = np.array([pose.x, pose.y, cfg.camera_height])
        return self._pano_renderer.render(eye, pose.yaw)

    def get_pano_image(self, pose: Pose) -> np.ndarray:
        """Equirectangular RGB panorama, uint8 (pano_height, pano_width, 3).
        The center column looks along the robot heading."""
        return self._render_pano(pose)[0]

    def get_pano_depth_image(self, pose: Pose) -> np.ndarray:
        """Equirectangular depth panorama, float32 (pano_height, pano_width),
        meters."""
        return self._render_pano(pose)[1]

    def get_pano_image_and_depth(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        """Both panoramic outputs from a single set of face renders."""
        return self._render_pano(pose)

    def release(self) -> None:
        if self._pano_renderer is not None:
            self._pano_renderer.release()
            self._pano_renderer = None
        if self._renderer is not None:
            self._renderer.release()
            self._renderer = None
        if self._ctx is not None:
            release_context(self._ctx)
            self._ctx = None

    def __enter__(self) -> "Simulator":
        return self

    def __exit__(self,
                 exc_type: type[BaseException] | None,
                 exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.release()


def image_aligned_to_world(image: np.ndarray, pose: Pose) -> np.ndarray:
    """Roll a robot-aligned panorama so its center column faces world +x
    (the old Unity-sim convention)."""
    cols = image.shape[1]
    roll_amount = -int(round(cols * pose.yaw / (2 * math.pi)))
    return np.roll(image, shift=roll_amount, axis=1)

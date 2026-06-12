"""Unknown-space exploration environment with panoramic image capture.

``VisualUnknownSpaceEnvironment`` couples the frontier-exploration stack
(:class:`UnknownSpaceEnvironment`) with a railsim scene: every laser sensor
step — the initial observation at t=0 and each ``sensor_dt``-capped step
while a robot is moving — also renders a panoramic image at the robot's
current pose and appends it to ``pano_records``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from railroad.environment.types import Pose
from railroad.experimental.unknown_search import UnknownSpaceEnvironment

from .maps import RailsimScene


@dataclass
class PanoRecord:
    """A panoramic image captured at a sensing step.

    Attributes:
        robot: Name of the robot that captured the image.
        time: Environment time of the capture.
        pose_cells: Robot pose in grid-cell coordinates.
        pose_meters: The same pose in meters (x, y, yaw) as rendered.
        image: Robot-aligned equirectangular RGB panorama,
            ``(pano_height, pano_width, 3)`` uint8.
        visibility_polygon: Closed vertex loop ``(2, N+2)`` in grid-cell
            coordinates of the laser scan polygon visible from this pose,
            or None when no scan accompanied the capture.
    """

    robot: str
    time: float
    pose_cells: Pose
    pose_meters: Tuple[float, float, float]
    image: np.ndarray
    visibility_polygon: np.ndarray | None = None


class VisualUnknownSpaceEnvironment(UnknownSpaceEnvironment):
    """Frontier-exploration environment that renders panos as robots sense.

    Accepts all :class:`UnknownSpaceEnvironment` keyword arguments;
    ``true_grid`` defaults to ``scene.grid`` (the inflated navigation grid).
    """

    def __init__(
        self,
        *,
        scene: RailsimScene,
        capture_panos: bool = True,
        **kwargs: Any,
    ) -> None:
        # Capture state must exist before super().__init__: the base
        # constructor performs the initial t=0 observation via
        # observe_all_robots, which calls our observe_from_pose override.
        self._scene = scene
        self._capture_panos = capture_panos
        self.pano_records: list[PanoRecord] = []
        self._last_capture: Dict[str, Tuple[float, float, float, float]] = {}
        self._last_scan_polygon: Dict[str, np.ndarray] = {}

        kwargs.setdefault("true_grid", scene.grid)
        super().__init__(**kwargs)

        # Expose the scene so the dashboard can draw overhead maps.
        self.scene = scene

    def _on_laser_scan(
        self,
        robot: str,
        pose: Pose,
        time: float,
        laser_ranges: np.ndarray,
    ) -> None:
        from railroad.experimental.unknown_search.mapping import (
            compute_scan_polygon_vertices,
        )

        self._last_scan_polygon[robot] = compute_scan_polygon_vertices(
            self._laser_directions,
            laser_ranges,
            self._config.sensor_range,
            pose,
        )

    def observe_from_pose(
        self,
        robot: str,
        pose: Pose,
        time: float,
        allow_interrupt: bool = True,
    ) -> int:
        new_cells = super().observe_from_pose(robot, pose, time, allow_interrupt)
        if self._capture_panos:
            key = (time, pose.x, pose.y, pose.yaw)
            if self._last_capture.get(robot) != key:
                sim_pose = self._scene.cell_pose_to_meters(pose)
                self.pano_records.append(
                    PanoRecord(
                        robot=robot,
                        time=time,
                        pose_cells=Pose(pose.x, pose.y, pose.yaw),
                        pose_meters=(sim_pose.x, sim_pose.y, sim_pose.yaw),
                        image=self._scene.get_pano_image(pose),
                        visibility_polygon=self._last_scan_polygon.get(robot),
                    )
                )
                self._last_capture[robot] = key
        return new_cells

"""railsim: portable OpenGL visual simulator for robotics research.

Requires optional dependencies: pip install railroad[railsim]

Coordinate convention (used everywhere, no exceptions):

- The world is right-handed with z up. The ground plane is x/y, walls rise
  from z=0 to the configured wall height.
- Occupancy grids map to the world as ``grid[i, j] <-> (x = i * resolution,
  y = j * resolution)`` at the cell center.
- ``Pose(x, y, yaw)`` is in meters/radians; ``yaw = 0`` faces +x and
  positive yaw rotates +x toward +y. The camera sits at ``z =
  config.camera_height`` with zero pitch and roll.
- Panoramic images are robot-aligned: the center column looks along the
  robot's heading. Use :func:`image_aligned_to_world` to recover the old
  Unity-sim convention (center column along world +x).

Note: railroad environments (e.g. ``VisualUnknownSpaceEnvironment``) work in
grid-cell units; ``RailsimScene.cell_pose_to_meters`` converts cell-space
poses to the meter-space ``SimPose`` the renderer expects.
"""

__all__ = [
    # Integration API
    "RailsimScene",
    "VisualUnknownSpaceEnvironment",
    "PanoRecord",
    "is_available",
    # Core railsim API
    "Color",
    "DEFAULT_PALETTE",
    "GuidedMazeConfig",
    "GuidedMazeGenerator",
    "LightRig",
    "MapData",
    "MapGenerator",
    "OccupancyGridWorld",
    "OfficeConfig",
    "OfficeGenerator",
    "Palette",
    "Pose",
    "SimPose",
    "Simulator",
    "SimulatorConfig",
    "World",
    "image_aligned_to_world",
    "make_guided_maze",
    "make_office",
    "resolve_palette",
    "world_from_occupancy_grid",
]

_INSTALL_MSG = (
    "railsim dependencies not installed. "
    "Install with: pip install railroad[railsim]"
)

# Required packages for railsim functionality
_REQUIRED_PACKAGES = [
    "moderngl",
    "shapely",
]


def is_available() -> bool:
    """Check if all railsim dependencies are installed.

    This is a lightweight check that doesn't load heavy modules.
    """
    import importlib.util

    return all(importlib.util.find_spec(pkg) is not None for pkg in _REQUIRED_PACKAGES)


# Lazy attribute -> (submodule, attribute) mapping. SimPose is an alias for
# railsim's meter-space Pose, disambiguating it from railroad's cell-space Pose.
_LAZY_EXPORTS = {
    "RailsimScene": (".maps", "RailsimScene"),
    "VisualUnknownSpaceEnvironment": (".visual_environment", "VisualUnknownSpaceEnvironment"),
    "PanoRecord": (".visual_environment", "PanoRecord"),
    "Color": (".palette", "Color"),
    "DEFAULT_PALETTE": (".palette", "DEFAULT_PALETTE"),
    "GuidedMazeConfig": (".environments", "GuidedMazeConfig"),
    "GuidedMazeGenerator": (".environments", "GuidedMazeGenerator"),
    "LightRig": (".render.renderer", "LightRig"),
    "MapData": (".environments", "MapData"),
    "MapGenerator": (".environments", "MapGenerator"),
    "OccupancyGridWorld": (".world", "OccupancyGridWorld"),
    "OfficeConfig": (".environments", "OfficeConfig"),
    "OfficeGenerator": (".environments", "OfficeGenerator"),
    "Palette": (".palette", "Palette"),
    "Pose": (".pose", "Pose"),
    "SimPose": (".pose", "Pose"),
    "Simulator": (".simulator", "Simulator"),
    "SimulatorConfig": (".simulator", "SimulatorConfig"),
    "World": (".world", "World"),
    "image_aligned_to_world": (".simulator", "image_aligned_to_world"),
    "make_guided_maze": (".environments", "make_guided_maze"),
    "make_office": (".environments", "make_office"),
    "resolve_palette": (".palette", "resolve_palette"),
    "world_from_occupancy_grid": (".world", "world_from_occupancy_grid"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr = _LAZY_EXPORTS[name]
        try:
            module = importlib.import_module(module_name, __name__)
            return getattr(module, attr)
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

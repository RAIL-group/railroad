# railsim

A portable, offscreen OpenGL visual simulator for robotics research: it renders
perspective and panoramic RGB/depth images of procedurally generated indoor
worlds (mazes and offices) from arbitrary poses. It is a faithful port of a
Unity-era lab simulator — same map generators, materials, and lighting — with
no Unity dependency: rendering is a few hundred lines of `moderngl` (OpenGL
3.3 core), and worlds are plain occupancy grids plus shapely polygons.

railsim is an optional component of `railroad`:

```
pip install railroad[railsim]   # adds moderngl + shapely (and friends)
```

Use `railroad.environment.railsim.is_available()` for a lightweight dependency
check; all heavy imports are lazy, so importing the package never requires GL.

## Quick start

Standalone rendering:

```python
from railroad.environment.railsim import (
    OccupancyGridWorld, Simulator, make_guided_maze,
)

map_data, start, goal = make_guided_maze(seed=13)
with Simulator(OccupancyGridWorld(map_data, seed=13)) as sim:
    rgb = sim.get_image(start)                  # (240, 320, 3) uint8
    pano, depth = sim.get_pano_image_and_depth(start)   # (256, 512, ...)
```

Integrated with the frontier-exploration planning stack:

```python
from railroad.environment.railsim import RailsimScene, VisualUnknownSpaceEnvironment

scene = RailsimScene.office(seed=2005)   # or RailsimScene.maze(seed=13)
env = VisualUnknownSpaceEnvironment(scene=scene, ...)
# ... plan/act as with any UnknownSpaceEnvironment; afterwards:
env.pano_records   # list[PanoRecord]: a panorama per laser sensing step
```

Or run the end-to-end example:

```
uv run railroad example visual-frontier-search --env maze|office
```

## Module layout

The package is layered so that everything below `render/` works without a GPU:

| Module | Role |
| --- | --- |
| `environments/` | Procedural map generators (`guided_maze`, `office`) producing `MapData`: occupancy grid + semantic grid + start/goal. Pure numpy/scipy — no geometry or GL. Occupancy-grid inflation and connectivity checks reuse `railroad.navigation.pathing`. |
| `world.py`, `geometry.py` | `World`/`OccupancyGridWorld`: shapely-polygon world models built from occupancy grids, plus breadcrumb and ceiling-light placement. |
| `scene.py` | Triangle-mesh construction from a `World` (walls, floor/ceiling, tables, breadcrumbs, light fixtures). Numpy only — no GL. |
| `render/` | The only OpenGL code: context creation (`context.py`), the forward renderer with shadow maps (`renderer.py`, `shaders.py`, `camera.py`), and GPU equirectangular panorama resampling (`pano.py`). |
| `simulator.py` | `Simulator`/`SimulatorConfig`: the public rendering API (`get_image`, `get_depth_image`, `get_pano_image`, ...). |
| `palette.py`, `pose.py` | sRGB color palette (Unity material albedos) and the meter-space `Pose`. |
| `maps.py` | `RailsimScene`: the bridge that exposes railsim worlds to railroad environments. |
| `visual_environment.py` | `VisualUnknownSpaceEnvironment` + `PanoRecord`: panorama capture during exploration. |

## Local environments

Both generators return `MapData` (`occ_grid`, `semantic_grid`,
`semantic_labels`, `start_cell`/`end_cell`, `resolution`, optional `tables`
and `palette`) and guarantee start and goal remain connected after inflating
obstacles by `config.inflation_radius_m` (0.75 m by default).

### Guided maze (`make_guided_maze`, `RailsimScene.maze(seed=...)`)

A maze carved on a coarse lattice by randomized wall removal; a "goal path"
between two random lattice cells is routed, marked in the semantic grid, and
rendered wider than regular hallways. Green **breadcrumbs** — flat squares
dart-thrown along the goal path — provide a visual guidance signal, so an
agent that learns to read the floor can follow them to the goal.

### Office (`make_office`, `RailsimScene.office(seed=...)`)

Random axis-aligned hallway centerlines (kept mutually connected) inflated to
corridor width, lined with rooms reachable through doors; "special" rooms
bridge pairs of hallway intersections. Rooms are furnished with **tables**:
occupied in the occupancy grid but rendered as 1.6 m solid boxes rather than
full-height walls. Walls take class-specific colors — light blue beside
hallways, warm gold beside rooms — so wall color alone reveals what kind of
space is around the corner. Offices have no goal path and no breadcrumbs.

## Integration with railroad

`RailsimScene` mirrors the `ProcTHORScene` data-provider surface (`.grid`,
`.locations`, `.object_locations`, `.get_top_down_image()`,
`.get_top_down_view()`), so railsim worlds plug into the unknown-space
exploration stack unchanged. `object_locations` is empty: railsim scenes are
exploration-only.

`get_top_down_view()` returns a `TopDownView` — the same image positioned in
occupancy-grid cells, which is what lets the dashboard draw it underneath a
trajectory. Note it transposes: `get_top_down_image()` is indexed `[x, y]` like
the grid, while the plot draws `grid.T`.

`VisualUnknownSpaceEnvironment` subclasses
`railroad.experimental.unknown_search.UnknownSpaceEnvironment` and hooks
`observe_from_pose`: every laser sensing step (the initial t=0 observation and
each `sensor_dt`-capped step while a robot moves) also renders a panorama at
the robot's pose and appends a `PanoRecord` (robot, time, cell/meter poses,
image) to `env.pano_records`. The dashboard uses these records to show onboard
camera imagery under the top-down view in its videos.

## Design decisions

**One coordinate convention, stated once.** The world is right-handed with z
up; the ground plane is x/y. Occupancy grids map to the world as
`grid[i, j] <-> (x = i * resolution, y = j * resolution)` at the cell center.
`Pose(x, y, yaw)` is meters/radians with `yaw = 0` facing +x and positive yaw
rotating +x toward +y. Every module follows this — there are no per-module
axis flips.

**Two unit spaces, one conversion point.** railroad environments work in
grid-cell units; railsim renders in meters. The conversion is a pure scaling
by `resolution` with yaw unchanged, and it lives in exactly one place:
`RailsimScene.cell_pose_to_meters`. `SimPose` is exported as an alias for
railsim's meter-space `Pose` to disambiguate it from railroad's cell-space
pose type.

**Navigation grid ≠ rendering geometry.** `RailsimScene.grid` is the raw
occupancy grid inflated by `inflation_radius_m` (default 0.75 m, matching the
generators' connectivity guarantee), so the laser senses and plans against
inflated walls while panoramas render the raw geometry (`.raw_grid`). This
keeps paths conservatively collision-free, at the cost that observed free
space ends ~`inflation_radius_m` short of the true walls; pass
`inflation_radius_m=0` to sense the exact geometry.

**Robot-aligned panoramas.** The center column of a panorama looks along the
robot's heading, and moving right in the image turns clockwise — matching the
left/right sense of a perspective image.

**Lazy GL, portable backends.** Constructing a `Simulator` (or a
`RailsimScene`) never touches OpenGL; the context is created on the first
render call, so scenes can be built and grids consumed on machines without a
GPU or display. Context creation tries the platform default (CGL on macOS,
GLX on Linux with a display) and falls back to EGL for headless Linux; set
`RAILSIM_GL_BACKEND` (`cgl`/`egl`/`glx`/`cpu`) to pin one — `cpu` forces
Mesa's llvmpipe software rasterizer on Linux. GL contexts are thread-affine:
use a `Simulator` from a single thread, and call `release()` (or use it as a
context manager) to free GPU resources.

**Static scenes make rendering cheap.** Worlds, lights, and meshes never
change after construction, so shadow maps are baked exactly once and each
frame is a single draw call producing both color and Euclidean view distance.
Panoramas are resampled entirely on the GPU: six overscanned cube faces are
rendered into a 3x2 atlas framebuffer and a fullscreen pass maps each pano
pixel to a face sample — only the final equirectangular image is read back.

**Unity parity by default, extensions opt-out.** Material albedos
(`palette.py`), the light rig (a shadowless downward spot plus a bare point
light per ceiling fixture, warm white), breadcrumb/table/fixture prefab
dimensions, and the dart-throwing light placement all replicate the Unity sim
— including quirks like the bounded sample count, which is part of the look.
Deliberate deviations are documented where they live: point-light shadows are
a realism extension (set `point_shadows=False` for strict parity), and the
generators use local RNGs instead of seeding global state (deterministic per
seed, but not bit-identical to the old maps). A `Palette` is a plain dict, so
experiments can recolor any subset of the scene (or add colors for new object
types) without touching scene code.

**No hard dependency on the lab stack.** The occupancy-grid-to-polygon
conversion (`geometry.py`) is self-contained and built on shapely 2.x, and
grid inflation/connectivity reuse `railroad.navigation.pathing`, so railsim
needs nothing outside PyPI.

## Testing

Map generation and scene/data-provider behavior are covered by
`packages/railroad/tests/environment/railsim/` and run headless. Rendering
paths require a working GL context; on macOS this means running outside
sandboxed shells (CGL needs WindowServer access).

#!/usr/bin/env python3
"""Minimal ProcTHOR scene demo.

Loads a stock ProcTHOR-10k scene exactly as molmospace_scene.py does and
spawns two MobileFranka robots -- but unlike build_scene() there, it does
not strip out any furniture or add box-station tables/cubes/scissors. It's
just the raw ProcTHOR home with two robots parked in it, rendered to a
single saved image.

Robot spawn spots are not hardcoded: molmospace_scene.py's own
WORKSTATION_LOCATIONS were hand-tuned for one specific room layout (scene
index 0) and collide with furniture in any other scene, since every
SCENE_INDEX has a different floor plan. Instead we search outward from the
room's center for two nearby spots where each robot's real MuJoCo collision
geometry doesn't overlap anything already in the scene (or the other
robot), using actual contact detection so it's correct regardless of
whether nearby obstacles are primitive boxes or meshes.
"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import imageio
import mujoco
import numpy as np
from mujoco import MjData, MjSpec

from molmo_spaces.molmo_spaces_constants import get_scenes_root
from molmo_spaces.utils.lazy_loading_utils import install_scene_from_source_index
from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata

from molmospace_scene import (  # pyrefly: ignore [missing-import]
    ROBOT_PREFIXES,
    SCENE_SOURCE,
    SETTLE_SECONDS,
    SIM_DT,
    add_robot_near,
)

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "procthor_demo.png")
IMG_WIDTH = 1280
IMG_HEIGHT = 720
SCENE_INDEX = 90

ROBOTS = ("robot1", "robot2")
ROBOT_THETA_DEG = 90.0
# Spawn search: ring-by-ring outward from the room center, checking real
# MuJoCo collision geometry at each candidate.
SEARCH_STEP_M = 0.4
SEARCH_MAX_RADIUS_M = 8.0
# Parks a not-yet-searched robot far outside the house so it can't produce
# spurious "collisions" while another robot's spot is being searched for.
PARKING_SPOT = (1000.0, 1000.0)


def robot_geom_ids(model: mujoco.MjModel, prefix: str) -> frozenset:
    """All geom ids belonging to any body under `prefix`'s kinematic tree
    (base + arm + gripper) -- includes unnamed collision geoms like the
    base's own box, so name-based filtering would miss some."""
    body_ids = {i for i in range(model.nbody) if model.body(i).name.startswith(prefix)}
    return frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g] in body_ids)


def has_collision(data: MjData, self_geom_ids: frozenset) -> bool:
    """True if any active contact has exactly one side among
    `self_geom_ids` -- i.e. this robot touching something that isn't
    itself (furniture, walls, or another robot). Contacts where both
    sides are `self_geom_ids` are the robot's own self-collision and
    don't depend on where it's placed, so they're ignored."""
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 in self_geom_ids) != (c.geom2 in self_geom_ids):
            return True
    return False


def room_center(model: mujoco.MjModel, data: MjData) -> np.ndarray:
    """(x, y) center of the largest room in the house (by floor mesh
    size), used as the free-spot search seed. A ProcTHOR home usually has
    several disjoint rooms, so averaging across all of them can land
    outside any single one -- anchoring on the biggest room instead keeps
    both robots' searches within one contiguous floor. Room bodies sit at
    the world origin (their floor mesh is offset via the geom's own pos),
    so this reads geom_xpos, not body xpos."""
    best_pos, best_area = None, -1.0
    for g in range(model.ngeom):
        if not model.body(model.geom_bodyid[g]).name.startswith("room_"):
            continue
        _, sy, sz = model.geom_size[g]
        area = sy * sz
        if area > best_area:
            best_area, best_pos = area, data.geom_xpos[g][:2].copy()
    return best_pos if best_pos is not None else np.array([0.0, 0.0])


def _spiral_offsets(step: float, max_radius: float):
    """(dx, dy) offsets spiraling outward from the origin, ring by ring."""
    yield (0.0, 0.0)
    r = step
    while r <= max_radius:
        n = max(8, int(2 * np.pi * r / step))
        for k in range(n):
            angle = 2 * np.pi * k / n
            yield (r * np.cos(angle), r * np.sin(angle))
        r += step


def find_free_spot(model, data, view, self_geom_ids: frozenset, center: np.ndarray) -> tuple:
    """Spiral outward from `center` until a collision-free (x, y) is
    found for the robot behind `view`/`self_geom_ids`."""
    for dx, dy in _spiral_offsets(SEARCH_STEP_M, SEARCH_MAX_RADIUS_M):
        x, y = center[0] + dx, center[1] + dy
        view.set_qpos_dict({"base": [x, y, np.radians(ROBOT_THETA_DEG)]})
        mujoco.mj_forward(model, data)
        if not has_collision(data, self_geom_ids):
            return x, y, ROBOT_THETA_DEG
    raise RuntimeError(f"No collision-free spot found within {SEARCH_MAX_RADIUS_M}m of {tuple(center)}")


def place_robots_free_of_collision(model, data, robot_configs: dict) -> dict:
    """Find and commit a collision-free base pose for each robot in
    `robot_configs`, searching outward from the room center on roughly
    opposite sides so the two robots don't converge on the same spot.
    Updates each config's init_qpos["base"] in place and returns the
    chosen {robot: (x, y, theta_deg)} poses."""
    center = room_center(model, data)
    views = {r: cfg.robot_view_factory(data, ROBOT_PREFIXES[r]) for r, cfg in robot_configs.items()}
    geom_ids = {r: robot_geom_ids(model, ROBOT_PREFIXES[r]) for r in robot_configs}

    for view in views.values():
        view.set_qpos_dict({"base": [*PARKING_SPOT, 0.0]})

    robot_bases = {}
    for i, robot in enumerate(robot_configs):
        seed_angle = i * np.pi  # opposite sides of center for i=0,1
        seed = (center[0] + 0.5 * np.cos(seed_angle), center[1] + 0.5 * np.sin(seed_angle))
        x, y, theta_deg = find_free_spot(model, data, views[robot], geom_ids[robot], np.array(seed))
        robot_configs[robot].init_qpos["base"] = [x, y, np.radians(theta_deg)]
        robot_bases[robot] = (x, y, theta_deg)
        mujoco.mj_forward(model, data)

    return robot_bases


def build_scene():
    """Load the raw ProcTHOR scene (no furniture removed/added) with
    robot1/robot2 spawned at automatically found collision-free spots."""
    print("Installing standard home environment scene assets...")
    install_scene_from_source_index(SCENE_SOURCE, SCENE_INDEX)
    scene_path = get_scenes_root() / SCENE_SOURCE / f"val_{SCENE_INDEX}.xml"
    spec = MjSpec.from_file(str(scene_path))
    scene_meta = get_scene_metadata(scene_path)

    print("Spawning robot1 and robot2 (placeholder positions)...")
    robot_configs = {}
    for robot in ROBOTS:
        robot_configs[robot] = add_robot_near(spec, ROBOT_PREFIXES[robot], (0.0, 0.0, ROBOT_THETA_DEG))

    print("Compiling model...")
    model = spec.compile()
    data = MjData(model)
    mujoco.mj_forward(model, data)

    print("Searching for collision-free spawn spots...")
    robot_bases = place_robots_free_of_collision(model, data, robot_configs)

    for robot, cfg in robot_configs.items():
        view = cfg.robot_view_factory(data, ROBOT_PREFIXES[robot])
        view.set_qpos_dict(cfg.init_qpos)
    mujoco.mj_forward(model, data)
    for robot, cfg in robot_configs.items():
        view = cfg.robot_view_factory(data, ROBOT_PREFIXES[robot])
        for mg_id in view.move_group_ids():
            mg = view.get_move_group(mg_id)
            mg.ctrl = mg.noop_ctrl
    mujoco.mj_forward(model, data)

    print(f"Settling physics for {SETTLE_SECONDS}s...")
    for _ in range(int(SETTLE_SECONDS / SIM_DT)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    return model, data, robot_bases, scene_meta


def _has_free_joint(model: mujoco.MjModel, body_id: int) -> bool:
    """True if `body_id` itself (not a descendant) has a free joint --
    i.e. it's a rigid body a robot could pick up, matching molmo_spaces'
    ObjectManager.has_free_joint()."""
    return any(
        model.jnt_bodyid[j] == body_id and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        for j in range(model.njnt)
    )


def print_scene_contents(model: mujoco.MjModel, scene_meta: dict) -> None:
    """Print the distinct locations (receptacle furniture/surfaces the
    robot can navigate to and put things on, e.g. tables/counters/sinks)
    and objects (pickupable items with a free joint) in the scene,
    deduplicated by category -- same locations-vs-objects split as
    bridge.py's ObjectManager.get_receptacles()/get_pickup_candidates(),
    just grouped by category since raw instance names are already unique.
    """
    objects_meta = scene_meta["objects"]
    location_categories = set()
    object_categories = set()

    for body_id in range(1, model.nbody):
        if model.body_parentid[body_id] != 0:
            continue  # not a top-level (root) scene object
        meta = objects_meta.get(model.body(body_id).name)
        if meta is None:
            continue  # structural (room/wall) or robot body, not a ProcTHOR object
        category = meta["category"]
        if meta.get("name_map", {}).get("sites"):
            location_categories.add(category)
        if _has_free_joint(model, body_id):
            object_categories.add(category)

    print(f"Locations ({len(location_categories)}):")
    for loc in sorted(location_categories):
        print(f"  {loc}")

    print(f"Objects ({len(object_categories)}):")
    for obj in sorted(object_categories):
        print(f"  {obj}")


def render_image(model, data, robot_bases, out_path: str = OUT_PATH) -> str:
    """Render a single top-down frame of the scene and save it as PNG."""
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    xs = [x for x, _, _ in robot_bases.values()]
    ys = [y for _, y, _ in robot_bases.values()]
    # Straight-down view centered on the robots: a perspective/angled shot
    # tuned for one room's coordinates points at the wrong spot (or frames
    # nothing) once SCENE_INDEX changes which room the robots land in, but
    # looking straight down at their midpoint always keeps them in frame.
    spread = max(np.hypot(xs[0] - xs[1], ys[0] - ys[1]), 1.0) if len(xs) > 1 else 3.0
    cam.lookat[:] = [sum(xs) / len(xs), sum(ys) / len(ys), 0.0]
    cam.distance = max(5.0, spread * 2.5)
    cam.azimuth = 90.0
    cam.elevation = -89.9

    renderer.update_scene(data, camera=cam)
    image = renderer.render()
    imageio.imwrite(out_path, image)
    print(f"Saved image to {out_path}")
    return out_path


def main() -> None:
    model, data, robot_bases, scene_meta = build_scene()
    print_scene_contents(model, scene_meta)
    render_image(model, data, robot_bases)


if __name__ == "__main__":
    main()

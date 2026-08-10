#!/usr/bin/env python3
"""MolmoSpaces scene construction for the box-station gift-wrap domain.

Builds the same kind of manipulation scene as demo_planners.py (a standard
ProcTHOR home layout with all backdrop furniture stripped out, leaving the
bare room shell plus 4 free-standing tables and 2 MobileFranka arms), but
names and places
everything to match the box-station domain already exercised by
pick_and_place_astar_boxstation.py / common.py / decentralized.py /
centralized.py:

  workstation1, workstation2 — each robot's own table (goods are wrapped
    here; only that robot may enter).
  tool_space, box_station    — shared tables common to both robots: the
    scissors live at tool_space, the gift cubes live at box_station.

This module only builds the scene (geometry + registry + robot configs). It
has no notion of the symbolic planning domain or of how a plan gets executed
— see molmospace_domain.py and molmospace_executor.py for those, so a scene
built here can be replayed against any plan without rebuilding it.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Tuple

warnings.filterwarnings("ignore")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import mujoco
import numpy as np
from mujoco import MjData, MjSpec, mjtGeom
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.robot_configs import MobileFrankaRobotConfig
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.molmo_spaces_constants import get_scenes_root
from molmo_spaces.robots.mobile_franka import MobileFrankaRobot
from molmo_spaces.utils.lazy_loading_utils import install_scene_from_source_index
from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata

# ─────────────────────────────────────────────────────────
#  Scene source (standard home environment layout)
# ─────────────────────────────────────────────────────────
SCENE_SOURCE = "procthor-10k-val"
SCENE_INDEX = 0

# Every top-level (non-child) object in the room is removed — no backdrop
# furniture is kept, just the bare structural shell (walls/floor/ceiling,
# which never appear in scene_meta["objects"] at all) plus doorway/window
# openings, which are metadata "root" objects but are really part of the
# wall shell rather than freestanding furniture.
KEEP_ROOT_CATEGORIES = frozenset({"Doorway", "Window"})

# ─────────────────────────────────────────────────────────
#  Table geometry (shared by all 4 tables)
# ─────────────────────────────────────────────────────────
TABLE_TOP_HALF = np.array([0.45, 0.30, 0.02])
TABLE_LEG_HALF = np.array([0.03, 0.03, 0.365])
TABLE_HEIGHT = 0.75
TABLE_SURFACE_Z = TABLE_HEIGHT + TABLE_TOP_HALF[2]

# The 4 tables, named directly after the box-station domain's own locations
# (see pick_and_place_astar_boxstation.py) rather than after arbitrary Greek
# letters — workstation1/workstation2 are each robot's own table;
# tool_space/box_station are shared. Laid out on a 2x2 grid, well clear of
# room_2's walls/counters (floor spans roughly x:[2.1, 10.7], y:[1.1, 13.9]).
WORKSTATION_LOCATIONS: Dict[str, Tuple[float, float]] = {
    "workstation1": (4.5, 6.0),
    "workstation2": (6.5, 6.0),
    "tool_space":   (4.5, 8.2),
    "box_station":  (6.5, 8.2),
}

# Each robot's home table and MuJoCo body-namespace prefix.
ROBOT_HOME: Dict[str, str] = {"robot1": "workstation1", "robot2": "workstation2"}
ROBOT_PREFIXES: Dict[str, str] = {"robot1": "robot_0/", "robot2": "robot_1/"}

# Lateral (x) offset applied to a robot's stand point at any table, so two
# robots working the same *shared* table (tool_space/box_station) approach
# side-by-side spots instead of both being sent to the exact same physical
# point -- see stand_pose(). Harmless at each robot's own workstation too,
# since nobody else ever stands there.
#
# MobileFrankaRobotConfig's base_size is [0.5, 0.5, 0.58] -- a 0.25m
# half-width box -- so a +/-0.25 offset (0.5m center-to-center separation)
# only gets the two bases exactly edge-to-edge with zero real clearance,
# which reads as a collision in a rendered video.
#
# Pushed higher than that runs straight into a different limit: the arm's
# IK reach to the object is solved holding the *neutral* pose's orientation
# fixed, so a bigger lateral offset makes it a harder reach at a fixed
# wrist angle. Measured empirically (see the offset scan against every
# real (robot, location, object) pick/place target in the plan): IK still
# converges for all of them up to +/-0.32, starts failing at +/-0.35.
# +/-0.30 keeps a safety margin under that boundary while still giving a
# real ~0.1m gap between the two 0.25m-half-width bases (vs. 0 before).
ROBOT_LATERAL_OFFSET: Dict[str, float] = {"robot1": -0.30, "robot2": 0.30}

# Which shared table each object starts at, matching the symbolic domain's
# initial state (see molmospace_domain.py).
OBJECT_HOME: Dict[str, str] = {"cube_1": "box_station", "cube_2": "box_station", "scissors": "tool_space"}
GRABBABLE_OBJECTS: Tuple[str, ...] = ("cube_1", "cube_2", "scissors")

# MobileFranka's gripper TCP site, namespaced per robot -- see
# molmo_spaces.robots.robot_views.mobile_franka_droid_view.MobileFrankaDroidRobotView
# (grasp_site_name="gripper/grasp_site").
GRIPPER_GRASP_SITE = "gripper/grasp_site"

# Small per-object offset from its table's center, so multiple objects on
# the same table don't spawn exactly on top of each other. Reused by
# molmospace_executor.py so pick/place animations interpolate to/from
# exactly where an object actually rests.
OBJECT_TABLE_OFFSET: Dict[str, Tuple[float, float]] = {
    "cube_1": (-0.12, 0.06),
    "cube_2": (0.12, -0.06),
    "scissors": (0.0, 0.0),
}

CUBE_HALF = 0.03
STAND_OFFSET_Y = 0.70   # robots stand this far south of a table, facing north
STAND_THETA_DEG = 90.0
SETTLE_SECONDS = 1.0
SIM_DT = 0.002


def resting_spot(obj: str, loc: str) -> np.ndarray:
    """World xyz where `obj` sits when resting on table `loc`."""
    tx, ty = WORKSTATION_LOCATIONS[loc]
    ox, oy = OBJECT_TABLE_OFFSET.get(obj, (0.0, 0.0))
    return np.array([tx + ox, ty + oy, TABLE_SURFACE_Z + CUBE_HALF + 0.002])


def stand_pose(loc: str, robot: str) -> Tuple[float, float, float]:
    """(x, y, theta_deg) where `robot` stands to work at table `loc`.

    Offset laterally per robot (see ROBOT_LATERAL_OFFSET) so two robots
    sent to the same shared table don't end up standing on top of each
    other -- they previously did, since this returned one fixed point
    regardless of which robot asked for it.
    """
    tx, ty = WORKSTATION_LOCATIONS[loc]
    tx += ROBOT_LATERAL_OFFSET[robot]
    return (tx, ty - STAND_OFFSET_Y, STAND_THETA_DEG)


# ─────────────────────────────────────────────────────────
#  Object registry — clear string IDs -> compiled MuJoCo body names
# ─────────────────────────────────────────────────────────
class ObjectRegistry:
    def __init__(self):
        self._id_to_body_name: Dict[str, str] = {}

    def register(self, object_id: str, body_name: str) -> None:
        if object_id in self._id_to_body_name:
            raise ValueError(f"Object ID '{object_id}' is already registered")
        self._id_to_body_name[object_id] = body_name

    def body_name(self, object_id: str) -> str:
        return self._id_to_body_name[object_id]

    def body_id(self, model: mujoco.MjModel, object_id: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body_name(object_id))

    def world_pos(self, model: mujoco.MjModel, data: MjData, object_id: str) -> np.ndarray:
        return data.xpos[self.body_id(model, object_id)].copy()

    def ids(self):
        return list(self._id_to_body_name.keys())


# ─────────────────────────────────────────────────────────
#  Procedural furniture / object builders
# ─────────────────────────────────────────────────────────
def all_furniture_roots(scene_meta: dict) -> list:
    """Every top-level (no parent) object in the scene except doorway/window
    openings, which are metadata root objects but are really part of the
    wall shell rather than freestanding furniture."""
    objects = scene_meta["objects"]
    return [
        name for name, o in objects.items()
        if not o.get("parent") and o.get("category") not in KEEP_ROOT_CATEGORIES
    ]


def remove_clashing_furniture(spec: MjSpec, scene_meta: dict, root_names: list) -> int:
    """Delete root_names and everything the scene metadata says rests on
    them. Scene-metadata parent/child links are a *logical* relationship,
    not MJCF body nesting, so deleting a table body alone leaves its
    tabletop items behind as orphaned clutter -- this walks the metadata
    graph to catch those too.
    """
    objects = scene_meta["objects"]
    to_delete: set = set()

    def collect(name: str) -> None:
        if name in to_delete:
            return
        to_delete.add(name)
        for child in objects.get(name, {}).get("children") or []:
            collect(child)

    for root in root_names:
        collect(root)

    deleted = 0
    for body in list(spec.worldbody.bodies):
        if body.name in to_delete:
            spec.delete(body)
            deleted += 1
    return deleted


def add_table(spec: MjSpec, table_id: str, pos_xy: Tuple[float, float]) -> str:
    """Spawn a static 4-legged table at (x, y). Returns the compiled body name."""
    x, y = pos_xy
    body = spec.worldbody.add_body(name=table_id, pos=[x, y, 0.0])

    body.add_geom(
        name=f"{table_id}_top",
        type=mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, TABLE_HEIGHT],
        size=TABLE_TOP_HALF.tolist(),
        rgba=[0.55, 0.38, 0.22, 1.0],
    )
    leg_z = TABLE_LEG_HALF[2]
    leg_xy = (TABLE_TOP_HALF[0] - TABLE_LEG_HALF[0] - 0.02, TABLE_TOP_HALF[1] - TABLE_LEG_HALF[1] - 0.02)
    for i, (lx, ly) in enumerate([
        (leg_xy[0], leg_xy[1]), (-leg_xy[0], leg_xy[1]),
        (leg_xy[0], -leg_xy[1]), (-leg_xy[0], -leg_xy[1]),
    ]):
        body.add_geom(
            name=f"{table_id}_leg{i}",
            type=mjtGeom.mjGEOM_BOX,
            pos=[lx, ly, leg_z],
            size=TABLE_LEG_HALF.tolist(),
            rgba=[0.35, 0.24, 0.14, 1.0],
        )
    return table_id


def grasp_anchor_name(obj_id: str) -> str:
    """Name of the site on `obj_id` that a robot's gripper weld attaches to."""
    return f"{obj_id}/grasp"


def weld_name(robot: str, obj_id: str) -> str:
    """Name of the (initially inactive) weld equality between `robot`'s
    gripper and `obj_id`'s grasp anchor site."""
    return f"{robot}_{obj_id}_weld"


def add_cube(spec: MjSpec, cube_id: str, pos: np.ndarray, rgba: list) -> str:
    """Spawn a small physics-enabled (free-jointed) cube. Returns the body name."""
    body = spec.worldbody.add_body(name=cube_id, pos=pos.tolist())
    body.add_freejoint()
    body.add_geom(
        name=f"{cube_id}_geom",
        type=mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
        rgba=rgba,
        mass=0.05,
    )
    body.add_site(name=grasp_anchor_name(cube_id), pos=[0.0, 0.0, 0.0])
    return cube_id


def add_scissors(spec: MjSpec, scissors_id: str, pos: np.ndarray) -> str:
    """Spawn a small free-jointed scissors-like grabbable tool built from
    primitive geoms. Returns the compiled body name."""
    body = spec.worldbody.add_body(name=scissors_id, pos=pos.tolist())
    body.add_freejoint()

    blade_half = [0.045, 0.006, 0.003]
    handle_half = [0.02, 0.008, 0.004]
    metal_rgba = [0.75, 0.76, 0.78, 1.0]
    handle_rgba = [0.15, 0.15, 0.85, 1.0]

    for side, angle_deg in (("a", 9.0), ("b", -9.0)):
        quat = R.from_euler("z", angle_deg, degrees=True).as_quat(scalar_first=True)
        body.add_geom(
            name=f"{scissors_id}_blade_{side}",
            type=mjtGeom.mjGEOM_BOX,
            pos=[blade_half[0] * 0.9, 0.0, 0.0],
            quat=quat.tolist(),
            size=blade_half,
            rgba=metal_rgba,
        )
        body.add_geom(
            name=f"{scissors_id}_handle_{side}",
            type=mjtGeom.mjGEOM_BOX,
            pos=[-handle_half[0] * 1.6, 0.02 if side == "a" else -0.02, 0.0],
            quat=quat.tolist(),
            size=handle_half,
            rgba=handle_rgba,
        )

    body.add_geom(
        name=f"{scissors_id}_pivot",
        type=mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, 0.0],
        size=[0.006, 0.004, 0.0],
        rgba=metal_rgba,
        mass=0.01,
    )
    body.add_site(name=grasp_anchor_name(scissors_id), pos=[0.0, 0.0, 0.0])
    return scissors_id


def add_robot_near(spec: MjSpec, prefix: str, base_xy_theta_deg: Tuple[float, float, float]) -> MobileFrankaRobotConfig:
    x, y, theta_deg = base_xy_theta_deg
    theta_rad = np.radians(theta_deg)

    robot_cfg = MobileFrankaRobotConfig(robot_namespace=prefix)
    robot_cfg.init_qpos["base"] = [x, y, theta_rad]

    MobileFrankaRobot.add_robot_to_scene(
        robot_cfg, spec,
        prefix=prefix,
        pos=[x, y],
        quat=R.from_euler("z", theta_rad).as_quat(scalar_first=True).tolist(),
    )
    MobileFrankaRobot.apply_control_overrides(spec, robot_cfg)
    return robot_cfg


# ─────────────────────────────────────────────────────────
#  Scene bundle
# ─────────────────────────────────────────────────────────
@dataclass
class MolmoSpaceScene:
    model: mujoco.MjModel
    data: MjData
    registry: ObjectRegistry
    robot_configs: Dict[str, MobileFrankaRobotConfig]   # "robot1"/"robot2" -> config
    robot_prefixes: Dict[str, str] = field(default_factory=lambda: dict(ROBOT_PREFIXES))
    location_positions: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(WORKSTATION_LOCATIONS))
    kinematics: MlSpacesKinematics = None


def build_scene(settle: bool = True) -> MolmoSpaceScene:
    """Build the box-station MolmoSpaces scene: standard home backdrop plus
    workstation1/workstation2/tool_space/box_station tables, 2 gift cubes at
    box_station, scissors at tool_space, and 2 MobileFranka robots parked at
    their own workstation.
    """
    registry = ObjectRegistry()

    print("Installing standard home environment scene assets...")
    install_scene_from_source_index(SCENE_SOURCE, SCENE_INDEX)
    scene_path = get_scenes_root() / SCENE_SOURCE / "val_0.xml"
    spec = MjSpec.from_file(str(scene_path))
    scene_meta = get_scene_metadata(scene_path)

    removed = remove_clashing_furniture(spec, scene_meta, all_furniture_roots(scene_meta))
    print(f"Removed {removed} backdrop furniture bodies (all room furniture + tabletop items).")

    print("Spawning workstation1, workstation2, tool_space, box_station...")
    for loc, pos in WORKSTATION_LOCATIONS.items():
        add_table(spec, loc, pos)

    print("Placing gift cubes at box_station and scissors at tool_space...")
    for obj_id, rgba in (("cube_1", [0.85, 0.1, 0.1, 1.0]), ("cube_2", [0.1, 0.6, 0.85, 1.0])):
        registry.register(obj_id, add_cube(spec, obj_id, resting_spot(obj_id, OBJECT_HOME[obj_id]), rgba))
    registry.register("scissors", add_scissors(spec, "scissors", resting_spot("scissors", OBJECT_HOME["scissors"])))

    print("Spawning robot1 (at workstation1) and robot2 (at workstation2)...")
    robot_configs: Dict[str, MobileFrankaRobotConfig] = {}
    for robot, home in ROBOT_HOME.items():
        base = stand_pose(home, robot)
        robot_configs[robot] = add_robot_near(spec, ROBOT_PREFIXES[robot], base)

    print("Wiring up grasp weld constraints (inactive until picked)...")
    for robot, prefix in ROBOT_PREFIXES.items():
        for obj_id in GRABBABLE_OBJECTS:
            spec.add_equality(
                name=weld_name(robot, obj_id),
                type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                name1=f"{prefix}{GRIPPER_GRASP_SITE}",
                name2=grasp_anchor_name(obj_id),
                active=False,
            )

    print("Compiling model...")
    model = spec.compile()
    data = MjData(model)

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

    if settle:
        print(f"Settling physics for {SETTLE_SECONDS}s...")
        for _ in range(int(SETTLE_SECONDS / SIM_DT)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

    kinematics = MlSpacesKinematics(MobileFrankaRobotConfig())
    return MolmoSpaceScene(
        model=model, data=data, registry=registry,
        robot_configs=robot_configs, kinematics=kinematics,
    )


if __name__ == "__main__":
    scene = build_scene()
    print("\nScene built. Registered objects:")
    for obj_id in scene.registry.ids():
        pos = scene.registry.world_pos(scene.model, scene.data, obj_id)
        print(f"  {obj_id:10s} -> {np.round(pos, 3)}")

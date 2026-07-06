#!/usr/bin/env python3
"""
MolmoSpaces two-robot pick-and-place demo.

Two MobileFranka robots work simultaneously in the same room, each
picking up a different object from the dining table and placing it
on a different destination surface:

  Robot 0 (robot_0/):  cup    -> dining table 2 (west of the room)
  Robot 1 (robot_1/):  plate  -> shelf           (south of the room)

Per-robot phases (identical to the single-robot demo, run concurrently
with one shared physics step per rendered tick):
  1  NAVIGATE          Robot drives to the source table
  2  REACH & GRASP     Arm extends, gripper closes on its object
  3  LIFT              Object lifted off the surface
  4  NAVIGATE w/ OBJ   Robot drives to its destination, carrying object
  5  LOWER & PLACE     Arm lowers object onto destination surface
  6  RELEASE & RETREAT Gripper opens, robot backs away

Output: robot_demo_two_robots.mp4  (640x480, 24 fps)
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import imageio
import mujoco
import numpy as np
from mujoco import MjData, MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.robot_configs import MobileFrankaRobotConfig
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.mobile_franka import MobileFrankaRobot
from molmo_spaces.utils.lazy_loading_utils import install_scene_from_source_index

# ─────────────────────────────────────────────────────────
#  Rendering / timing
# ─────────────────────────────────────────────────────────
OUTPUT_PATH  = "robot_demo_two_robots.mp4"
W, H         = 640, 480
FPS          = 24
SIM_DT       = 0.002                              # MuJoCo timestep (s)
RENDER_EVERY = max(1, int(round(1.0 / (FPS * SIM_DT))))

def secs_to_steps(s: float) -> int:
    return max(1, int(s / SIM_DT))

# ─────────────────────────────────────────────────────────
#  Scene objects  (procthor-10k-val scene 0)
# ─────────────────────────────────────────────────────────
CUP_NAME   = "cup_4697b73732349085e30935e0f35566dc_1_0_2"
CUP_JNT    = "cup_4697b73732349085e30935e0f35566dc_1_0_2_jntfree_0"
CUP_POS    = np.array([5.31, 6.87, 0.82])    # nominal cup world position

PLATE_NAME = "plate_ac6a1b74f35572c9a1a82922f273449a_1_0_2"
PLATE_JNT  = "plate_ac6a1b74f35572c9a1a82922f273449a_1_0_2_jntfree_0"
PLATE_POS  = np.array([5.31, 7.12, 0.76])    # nominal plate world position (same table, north of cup)

HOME_JOINTS   = np.array([0, -0.7853, 0, -2.35619, 0, 1.57079, 0.0])
GRIPPER_OPEN  = np.array([0.0])
GRIPPER_CLOSE = np.array([255.0])

# ─────────────────────────────────────────────────────────
#  Phase durations (seconds) — shared by both robots so their
#  timelines stay synchronized
# ─────────────────────────────────────────────────────────
SCENE_SETTLE     = 0.6   # physics settle before robots move
NAV1_DURATION    = 5.0   # navigate to source table
SETTLE_AFTER_NAV = 0.8   # hold position after each navigation
REACH_DURATION   = 4.5   # arm reach out to object
GRIP_CLOSE_DUR   = 0.8   # gripper close
GRIP_SETTLE_DUR  = 0.8   # arm + gripper settle before measuring offset
LIFT_DURATION    = 3.0   # lift object
NAV2_DURATION    = 6.0   # navigate to destination (carrying object)
LOWER_DURATION   = 4.5   # lower object onto destination
RELEASE_DUR      = 0.6   # open gripper
RETRACT_DURATION = 2.0   # arm home
RETREAT_DURATION = 2.5   # drive away


# ─────────────────────────────────────────────────────────
#  Per-robot task definitions  [x, y, theta_degrees] waypoints
#
#  Both robots start at the same dining table (table 1) where the
#  cup and plate sit side by side, then diverge to different
#  destinations so their navigation paths don't overlap:
#    Robot 0: table 1 -> dining table 2 (west)
#    Robot 1: table 1 -> shelf          (south)
# ─────────────────────────────────────────────────────────
class RobotTask:
    def __init__(self, label, prefix, start, grasp, place, retreat,
                 item_name, item_jnt, item_pos, place_pos):
        self.label      = label
        self.prefix     = prefix
        self.start      = start
        self.grasp      = grasp
        self.place      = place
        self.retreat    = retreat
        self.item_name  = item_name
        self.item_jnt   = item_jnt
        self.item_pos   = np.array(item_pos, dtype=float)
        self.lift_pos   = self.item_pos + np.array([0.0, 0.0, 0.33])
        self.place_pos  = np.array(place_pos, dtype=float)


TASKS = [
    RobotTask(
        label="robot0(cup)", prefix="robot_0/",
        start=[5.0, 5.0, 0.0],
        grasp=[5.31, 6.17, 90.0],     # south of cup, faces +Y
        place=[3.8, 5.23, 180.0],     # east of table 2, faces -X
        retreat=[4.5, 5.23, 0.0],
        item_name=CUP_NAME, item_jnt=CUP_JNT, item_pos=CUP_POS,
        place_pos=[3.0, 5.23, 0.88],
    ),
    RobotTask(
        label="robot1(plate)", prefix="robot_1/",
        start=[4.6, 8.5, -90.0],       # due north of grasp, same x as grasp
                                        # (straight-line approach never crosses
                                        # x=5.31 where the cup/plate sit)
        grasp=[4.6, 7.12, 0.0],       # west of plate, faces +X
        place=[5.94, 5.12, -90.0],    # north of shelf, faces -Y
        retreat=[5.94, 5.7, 90.0],
        item_name=PLATE_NAME, item_jnt=PLATE_JNT, item_pos=PLATE_POS,
        place_pos=[6.05, 4.45, 1.05],
    ),
]


# ─────────────────────────────────────────────────────────
#  Math helpers
# ─────────────────────────────────────────────────────────
def base_pose_mat(x, y, theta_rad):
    """4×4 world pose matrix for base at (x, y, theta)."""
    bp = np.eye(4)
    bp[0, 3] = x
    bp[1, 3] = y
    bp[:2, :2] = R.from_euler("z", theta_rad).as_matrix()[:2, :2]
    return bp

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def lerp(a, b, t):
    return a + (b - a) * t


# ─────────────────────────────────────────────────────────
#  IK helper
# ─────────────────────────────────────────────────────────
def solve_ik(kin, x, y, theta_rad, q0, target_pos, tcp_ori, max_iter=6000):
    bp = base_pose_mat(x, y, theta_rad)
    T  = np.eye(4)
    T[:3, 3]  = target_pos
    T[:3, :3] = tcp_ori
    return kin.ik("arm", T, ["arm"], q0, bp, max_iter=max_iter)

def arm_home_q0(x, y, theta_deg):
    return {
        "base":    np.array([x, y, np.radians(theta_deg)]),
        "arm":     HOME_JOINTS.copy(),
        "gripper": np.array([0.00296, 0.00296]),
    }


# ─────────────────────────────────────────────────────────
#  Free-joint object manipulation
# ─────────────────────────────────────────────────────────
def set_item_pose(model, data, jnt_id, pos, quat=None):
    """Teleport a free-jointed item to pos (world). Zero its velocity."""
    adr = model.jnt_qposadr[jnt_id]
    data.qpos[adr:adr+3] = pos
    data.qpos[adr+3:adr+7] = quat if quat is not None else [1, 0, 0, 0]
    dof = model.jnt_dofadr[jnt_id]
    data.qvel[dof:dof+6] = 0.0

PLACED_ITEM_BIT = 1 << 10
# Bits used for per-robot arm/gripper collision groups (see the robot-vs-robot
# exclusion set up in main(): robot i uses bit (i+1)). Only the bits actually
# assigned to TASKS are cleared from a placed item's conaffinity — scene
# furniture also uses low bits (e.g. a table geom seen with contype=8) for
# its own grouping, so a broader mask would zero out floor/table support too.
ROBOT_BITS_MASK = sum(1 << (i + 1) for i in range(len(TASKS)))

def body_and_descendants(model, root_body_id):
    """Body ids in root_body_id's subtree. Scene assets here commonly split
    the free joint (parent, e.g. "..._1_0_2") from the actual collision mesh
    (a child body, e.g. "..._1_1_2"), so geom edits must walk the subtree
    rather than touching the named body alone."""
    ids = {root_body_id}
    changed = True
    while changed:
        changed = False
        for bid in range(model.nbody):
            if bid not in ids and model.body_parentid[bid] in ids:
                ids.add(bid)
                changed = True
    return ids

def detach_item_from_gripper(model, item_body_id):
    """Move a just-released item to its own collision group so the
    retracting gripper can't snag/drag it on the way out, while it keeps
    colliding normally with furniture/floor. Both contype *and* conaffinity
    must be touched: a geom pair collides if (contype1 & conaffinity2) OR
    (contype2 & conaffinity1) is nonzero, and scene items default to a wide
    conaffinity (15) that matches a gripper's contype bit on its own."""
    body_ids = body_and_descendants(model, item_body_id)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] in body_ids:
            model.geom_contype[gid]     = PLACED_ITEM_BIT
            model.geom_conaffinity[gid] = model.geom_conaffinity[gid] & ~ROBOT_BITS_MASK


# ─────────────────────────────────────────────────────────
#  Rendering
# ─────────────────────────────────────────────────────────
def make_camera():
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [4.8, 5.7, 0.9]
    cam.distance  = 12.0
    cam.azimuth   = 215.0
    cam.elevation = -38.0
    return cam

def grab_frame(renderer, data, cam):
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


# ─────────────────────────────────────────────────────────
#  Per-robot phase generator
#
#  Each call advances exactly one simulation step's worth of control
#  (setting base/arm/gripper ctrl) and yields. The outer loop in
#  main() drives every robot's generator once per shared mj_step,
#  so both robots' phase timelines run concurrently in lockstep.
# ─────────────────────────────────────────────────────────
def robot_task_gen(task, model, data):
    base_mg    = task.base_mg
    arm_mg     = task.arm_mg
    gripper_mg = task.gripper_mg
    item_body_id = task.item_body_id
    item_jnt_id  = task.item_jnt_id

    def run(n, ctrl_fn=None):
        for _ in range(n):
            if ctrl_fn:
                ctrl_fn()
            yield

    def run_smooth(n, ctrl_fn):
        for i in range(n):
            t = smoothstep(i / max(n - 1, 1))
            ctrl_fn(t)
            yield

    def hold(n, base_ctrl, arm_ctrl, grip_ctrl):
        def fn():
            base_mg.ctrl    = base_ctrl
            arm_mg.ctrl     = arm_ctrl
            gripper_mg.ctrl = grip_ctrl
        yield from run(n, fn)

    gx, gy, gth = task.grasp
    px, py, pth = task.place
    rx, ry, rth = task.retreat
    tgt_base_g = np.array([gx, gy, np.radians(gth)])
    tgt_base_p = np.array([px, py, np.radians(pth)])
    tgt_base_r = np.array([rx, ry, np.radians(rth)])

    arm_grasp = task.arm_grasp
    arm_lift  = task.arm_lift
    arm_place = task.arm_place

    # ── PHASE 1 — NAVIGATE to source table ──────────────────
    print(f"[{task.label}] Phase 1: NAVIGATE to source table...")
    def nav1_ctrl():
        base_mg.ctrl    = tgt_base_g
        arm_mg.ctrl     = HOME_JOINTS
        gripper_mg.ctrl = GRIPPER_OPEN
    yield from run(secs_to_steps(NAV1_DURATION), nav1_ctrl)
    yield from hold(secs_to_steps(SETTLE_AFTER_NAV), tgt_base_g, HOME_JOINTS, GRIPPER_OPEN)

    # ── PHASE 2 — REACH & GRASP ──────────────────────────────
    print(f"[{task.label}] Phase 2: REACH & GRASP...")
    arm_start = arm_mg.joint_pos.copy()
    def reach_ctrl(t):
        base_mg.ctrl    = tgt_base_g
        arm_mg.ctrl     = lerp(arm_start, arm_grasp, t)
        gripper_mg.ctrl = GRIPPER_OPEN
    yield from run_smooth(secs_to_steps(REACH_DURATION), reach_ctrl)

    print(f"  [{task.label}] Closing gripper...")
    yield from hold(secs_to_steps(GRIP_CLOSE_DUR), tgt_base_g, arm_grasp, GRIPPER_CLOSE)
    yield from hold(secs_to_steps(GRIP_SETTLE_DUR), tgt_base_g, arm_grasp, GRIPPER_CLOSE)

    # Measure actual item <-> TCP offset now (both fully settled)
    mujoco.mj_forward(model, data)
    tcp_pos      = arm_mg.leaf_frame_to_world[:3, 3].copy()
    item_pos_now = data.xpos[item_body_id].copy()
    grab_offset  = item_pos_now - tcp_pos
    dist = np.linalg.norm(grab_offset)
    print(f"  [{task.label}] TCP={tcp_pos.round(3)}  item={item_pos_now.round(3)}  offset‖={dist:.3f}m")
    if dist > 0.30:
        print(f"  [{task.label}] WARNING: offset > 0.30m — arm may have missed the object.")

    # ── PHASE 3 — LIFT ────────────────────────────────────────
    print(f"[{task.label}] Phase 3: LIFT...")
    arm_at_grasp = arm_mg.joint_pos.copy()
    def carry_item():
        tcp = arm_mg.leaf_frame_to_world[:3, 3]
        set_item_pose(model, data, item_jnt_id, tcp + grab_offset)
    def lift_ctrl(t):
        base_mg.ctrl    = tgt_base_g
        arm_mg.ctrl     = lerp(arm_at_grasp, arm_lift, t)
        gripper_mg.ctrl = GRIPPER_CLOSE
        carry_item()
    yield from run_smooth(secs_to_steps(LIFT_DURATION), lift_ctrl)

    # ── PHASE 4 — NAVIGATE WITH OBJECT ───────────────────────
    print(f"[{task.label}] Phase 4: NAVIGATE WITH OBJECT...")
    def nav2_ctrl():
        base_mg.ctrl    = tgt_base_p
        arm_mg.ctrl     = arm_lift
        gripper_mg.ctrl = GRIPPER_CLOSE
        carry_item()
    yield from run(secs_to_steps(NAV2_DURATION), nav2_ctrl)
    yield from run(secs_to_steps(SETTLE_AFTER_NAV), nav2_ctrl)

    # ── PHASE 5 — LOWER & PLACE ───────────────────────────────
    print(f"[{task.label}] Phase 5: LOWER & PLACE...")
    arm_at_lift = arm_mg.joint_pos.copy()
    def lower_ctrl(t):
        base_mg.ctrl    = tgt_base_p
        arm_mg.ctrl     = lerp(arm_at_lift, arm_place, t)
        gripper_mg.ctrl = GRIPPER_CLOSE
        carry_item()
    yield from run_smooth(secs_to_steps(LOWER_DURATION), lower_ctrl)

    print(f"  [{task.label}] Releasing...")
    yield from hold(secs_to_steps(RELEASE_DUR), tgt_base_p, arm_place, GRIPPER_OPEN)

    # Item is down and the gripper has opened — move it to its own collision
    # group so the retracting/retreating gripper can't snag and drag it off
    # the destination surface on the way out.
    detach_item_from_gripper(model, item_body_id)

    # ── PHASE 6 — RELEASE & RETREAT ───────────────────────────
    print(f"[{task.label}] Phase 6: RELEASE & RETREAT...")
    arm_at_place = arm_mg.joint_pos.copy()
    def retract_ctrl(t):
        base_mg.ctrl    = tgt_base_p
        arm_mg.ctrl     = lerp(arm_at_place, HOME_JOINTS, t)
        gripper_mg.ctrl = GRIPPER_OPEN
    yield from run_smooth(secs_to_steps(RETRACT_DURATION), retract_ctrl)

    def retreat_ctrl():
        base_mg.ctrl    = tgt_base_r
        arm_mg.ctrl     = HOME_JOINTS
        gripper_mg.ctrl = GRIPPER_OPEN
    yield from run(secs_to_steps(RETREAT_DURATION), retreat_ctrl)


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────
def main():
    # ── 1. Scene ──────────────────────────────────────────────
    print("Installing scene assets...")
    install_scene_from_source_index("procthor-10k-val", 0)
    scene_path = (
        "/home/ridwan/.cache/molmospaces/assets/"
        "L2hvbWUvcmlkd2FuL21vbG1vc3BhY2Vz/"
        "scenes/procthor-10k-val/val_0.xml"
    )
    spec = MjSpec.from_file(scene_path)

    # ── 2. Robots ─────────────────────────────────────────────
    print("Adding MobileFranka robots to scene...")
    for task in TASKS:
        sx, sy, sth_deg = task.start
        sth_rad = np.radians(sth_deg)
        robot_cfg = MobileFrankaRobotConfig(base_size=[0.5, 0.5, 0.75], robot_namespace=task.prefix)
        robot_cfg.init_qpos["base"] = [sx, sy, sth_rad]

        MobileFrankaRobot.add_robot_to_scene(
            robot_cfg, spec,
            prefix=task.prefix,
            pos=[sx, sy],
            quat=R.from_euler("z", sth_rad).as_quat(scalar_first=True),
        )
        MobileFrankaRobot.apply_control_overrides(spec, robot_cfg)
        task.robot_cfg = robot_cfg

    # ── 3. Compile ────────────────────────────────────────────
    print("Compiling model...")
    model = spec.compile()
    data  = MjData(model)

    # Disable collision on each robot base platform so the holonomic
    # bases can glide to their targets without getting stuck on
    # furniture (or each other). Arm and gripper geoms keep collision.
    for task in TASKS:
        base_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{task.prefix}base"
        )
        if base_body_id >= 0:
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == base_body_id:
                    model.geom_contype[gid]    = 0
                    model.geom_conaffinity[gid] = 0

    # Give each robot's arm/gripper geoms their own collision group so the
    # two robots can never collide with *each other* (both reach into the
    # same table concurrently, and a transient arm-arm contact during the
    # fast holonomic-base snap can wedge a robot in place) while still
    # colliding normally with furniture/items for realistic grasping.
    for i, task in enumerate(TASKS):
        robot_bit = 1 << (i + 1)
        for gid in range(model.ngeom):
            body_id = model.geom_bodyid[gid]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if body_name.startswith(task.prefix) and body_name != f"{task.prefix}base":
                model.geom_contype[gid]    = robot_bit
                model.geom_conaffinity[gid] = 1

    # Disable collision on the dining chairs tucked tight around table 1
    # (their seat/back are separate child bodies, e.g. "..._1_1_2", not just
    # the "..._1_0_2" body) so a robot's arm fr3_link0 can't wedge against a
    # chair while reaching into the table both robots share.
    TABLE1_CHAIR_PREFIX = "chair_0ac9b0b021ef299afd8f5636f317ad41_"
    for gid in range(model.ngeom):
        body_id = model.geom_bodyid[gid]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith(TABLE1_CHAIR_PREFIX):
            model.geom_contype[gid]    = 0
            model.geom_conaffinity[gid] = 0

    # ── 4. Init robot state ───────────────────────────────────
    for task in TASKS:
        view = task.robot_cfg.robot_view_factory(data, task.prefix)
        view.set_qpos_dict(task.robot_cfg.init_qpos)
        task.view = view

    mujoco.mj_forward(model, data)
    for task in TASKS:
        for mg_id in task.view.move_group_ids():
            mg = task.view.get_move_group(mg_id)
            mg.ctrl = mg.noop_ctrl
    mujoco.mj_forward(model, data)

    for task in TASKS:
        task.base_mg    = task.view.get_move_group("base")
        task.arm_mg     = task.view.get_move_group("arm")
        task.gripper_mg = task.view.get_move_group("gripper")

    # ── 5. Item IDs ───────────────────────────────────────────
    for task in TASKS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, task.item_name)
        jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, task.item_jnt)
        if body_id < 0 or jnt_id < 0:
            raise RuntimeError(f"Item body/joint not found in model for {task.label}")
        task.item_body_id = body_id
        task.item_jnt_id  = jnt_id

    # ── 6. Pre-solve IK ───────────────────────────────────────
    print("Solving IK targets...")
    kin_cfg = MobileFrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
    kin     = MlSpacesKinematics(kin_cfg)

    for task in TASKS:
        gx, gy, gth = task.grasp
        q0_g  = arm_home_q0(gx, gy, gth)
        bp_g  = base_pose_mat(gx, gy, np.radians(gth))
        ori_g = kin.fk(q0_g, bp_g)["arm"][:3, :3]
        res_grasp = solve_ik(kin, gx, gy, np.radians(gth), q0_g, task.item_pos, ori_g)
        res_lift  = solve_ik(kin, gx, gy, np.radians(gth), q0_g, task.lift_pos, ori_g)
        if res_grasp is None or res_lift is None:
            raise RuntimeError(f"Grasp/lift IK failed for {task.label}")
        task.arm_grasp = res_grasp["arm"]
        task.arm_lift  = res_lift["arm"]

        px, py, pth = task.place
        q0_p  = arm_home_q0(px, py, pth)
        bp_p  = base_pose_mat(px, py, np.radians(pth))
        ori_p = kin.fk(q0_p, bp_p)["arm"][:3, :3]
        res_place = solve_ik(kin, px, py, np.radians(pth), q0_p, task.place_pos, ori_p)
        if res_place is None:
            raise RuntimeError(f"Place IK failed for {task.label}")
        task.arm_place = res_place["arm"]
        print(f"  [{task.label}] IK solved: grasp, lift, place  ✓")

    # ── 7. Camera & renderer ──────────────────────────────────
    cam      = make_camera()
    renderer = mujoco.Renderer(model, height=H, width=W)

    # ── 8. Simulation loop ─────────────────────────────────────
    frames     = []
    step_count = [0]

    def render_tick():
        mujoco.mj_step(model, data)
        step_count[0] += 1
        if step_count[0] % RENDER_EVERY == 0:
            frames.append(grab_frame(renderer, data, cam))

    # SCENE SETTLE (let physics stabilise before robots move)
    for _ in range(secs_to_steps(SCENE_SETTLE)):
        render_tick()

    # Run both robots' phase generators concurrently — one shared
    # mj_step per tick, each robot's control set just before it.
    gens  = {task.label: robot_task_gen(task, model, data) for task in TASKS}
    alive = {label: True for label in gens}
    while any(alive.values()):
        for label, gen in gens.items():
            if alive[label]:
                try:
                    next(gen)
                except StopIteration:
                    alive[label] = False
        render_tick()

    # ── 9. Write video ───────────────────────────────────────
    print(f"\nRendered {len(frames)} frames. Writing {OUTPUT_PATH}...")
    with imageio.get_writer(OUTPUT_PATH, fps=FPS, codec="libx264",
                            quality=7, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)

    duration = len(frames) / FPS
    print(f"Done!  {OUTPUT_PATH}  {W}×{H}  {FPS}fps  {duration:.1f}s")


if __name__ == "__main__":
    main()

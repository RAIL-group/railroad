#!/usr/bin/env python3
"""
Plan-then-execute demo: railroad plans, MolmoSpaces executes.

A single MobileFranka robot must move a cup from one table to another in a
MuJoCo ProcTHOR scene. Unlike robot_demo_two_robots.py (where the
navigate/pick/place waypoint *sequence* is hand-authored), here the sequence
of high-level actions comes from railroad's MCTS planner running over a small
symbolic PDDL problem (locations + a move/pick/place operator set). Only the
physical world data (the (x, y, theta) pose and grasp/place targets for each
named location) is hardcoded; the planner decides what to do and in what
order.

Pipeline:
  1. PLAN   Build a symbolic SymbolicEnvironment (robot0, cup, 3 locations)
            and run railroad.planner.MCTSPlanner iteratively until the goal
            ("at cup table2") is reached, collecting the chosen actions.
  2. EXECUTE  Replay that action list against the real MuJoCo scene: each
            "move" becomes a NAVIGATE phase, each "pick" becomes REACH &
            GRASP + LIFT, each "place" becomes LOWER & PLACE + RETRACT.

Run with --plan-only to print just the symbolic plan (fast, no MuJoCo).

Output: planned_single_robot_demo.mp4  (640x480, 24 fps)
"""

import argparse
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
#  World data shared between symbolic planning and physical execution
#  (poses reused from the validated robot0(cup) task in
#  robot_demo_two_robots.py — procthor-10k-val scene 0)
# ─────────────────────────────────────────────────────────
POSES = {
    "start":  (5.0, 5.0, 0.0),
    "table1": (5.31, 6.17, 90.0),    # south of cup, faces +Y
    "table2": (3.8, 5.23, 180.0),    # east of table 2, faces -X
}

CUP_NAME = "cup_4697b73732349085e30935e0f35566dc_1_0_2"
CUP_JNT  = "cup_4697b73732349085e30935e0f35566dc_1_0_2_jntfree_0"
CUP_POS  = np.array([5.31, 6.87, 0.82])

ITEMS = {
    "cup": {
        "body_name": CUP_NAME,
        "jnt_name":  CUP_JNT,
        "item_pos":  CUP_POS,
        "place_pos": np.array([3.0, 5.23, 0.88]),
    },
}

ROBOT_PREFIX = "robot_0/"
SCENE_NAME, SCENE_INDEX = "procthor-10k-val", 0
SCENE_PATH = (
    "/home/ridwan/.cache/molmospaces/assets/"
    "L2hvbWUvcmlkd2FuL21vbG1vc3BhY2Vz/"
    f"scenes/{SCENE_NAME}/val_{SCENE_INDEX}.xml"
)

OUTPUT_PATH = "planned_single_robot_demo.mp4"
W, H  = 640, 480
FPS   = 24
SIM_DT = 0.002
RENDER_EVERY = max(1, int(round(1.0 / (FPS * SIM_DT))))

HOME_JOINTS   = np.array([0, -0.7853, 0, -2.35619, 0, 1.57079, 0.0])
GRIPPER_OPEN  = np.array([0.0])
GRIPPER_CLOSE = np.array([255.0])

SCENE_SETTLE     = 0.6
NAV_DURATION     = 5.0
SETTLE_AFTER_NAV = 0.8
REACH_DURATION   = 4.5
GRIP_CLOSE_DUR   = 0.8
GRIP_SETTLE_DUR  = 0.8
LIFT_DURATION    = 3.0
LOWER_DURATION   = 4.5
RELEASE_DUR      = 0.6
RETRACT_DURATION = 2.0

ROBOT_BIT       = 1 << 1
PLACED_ITEM_BIT = 1 << 10

TABLE1_CHAIR_PREFIX = "chair_0ac9b0b021ef299afd8f5636f317ad41_"


def secs_to_steps(s: float) -> int:
    return max(1, int(s / SIM_DT))


# ─────────────────────────────────────────────────────────
#  1. Symbolic planning
# ─────────────────────────────────────────────────────────
def build_planning_env():
    from railroad.core import Fluent as F, State
    from railroad.environment.symbolic import SymbolicEnvironment, LocationRegistry
    from railroad import operators

    registry = LocationRegistry({name: np.array(pose[:2]) for name, pose in POSES.items()})
    move_op  = operators.construct_move_operator_blocking(registry.move_time_fn(velocity=1.0))
    pick_op  = operators.construct_pick_operator_blocking(pick_time=5.0)
    place_op = operators.construct_place_operator_blocking(place_time=5.0)

    initial_fluents = {
        F("at robot0 start"),
        F("free robot0"),
        F("at cup table1"),
    }
    state = State(0.0, initial_fluents, [])
    return SymbolicEnvironment(
        state=state,
        objects_by_type={
            "robot": {"robot0"},
            "location": set(POSES),
            "object": {"cup"},
        },
        operators=[move_op, pick_op, place_op],
        location_registry=registry,
    )


def generate_plan(max_steps: int = 10) -> list[str]:
    from railroad.core import Fluent as F, get_action_by_name
    from railroad.planner import MCTSPlanner

    env = build_planning_env()
    goal = F("at cup table2")

    plan: list[str] = []
    for _ in range(max_steps):
        if goal.evaluate(env.fluents):
            return plan
        all_actions = env.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(env.state, goal, max_iterations=5000, max_depth=15)
        if action_name == "NONE":
            raise RuntimeError("Planner could not find a next action toward the goal.")
        action = get_action_by_name(all_actions, action_name)
        env.act(action)
        plan.append(action_name)

    raise RuntimeError(f"Plan did not reach the goal within {max_steps} steps.")


def parse_action(action_name: str) -> tuple[str, list[str]]:
    parts = action_name.split()
    return parts[0], parts[1:]


# ─────────────────────────────────────────────────────────
#  2. MuJoCo / MolmoSpaces execution helpers
#  (same math/collision helpers as robot_demo_two_robots.py)
# ─────────────────────────────────────────────────────────
def base_pose_mat(x, y, theta_rad):
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

def pose_to_ctrl(pose):
    x, y, theta_deg = pose
    return np.array([x, y, np.radians(theta_deg)])

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

def set_item_pose(model, data, jnt_id, pos, quat=None):
    adr = model.jnt_qposadr[jnt_id]
    data.qpos[adr:adr+3] = pos
    data.qpos[adr+3:adr+7] = quat if quat is not None else [1, 0, 0, 0]
    dof = model.jnt_dofadr[jnt_id]
    data.qvel[dof:dof+6] = 0.0

def body_and_descendants(model, root_body_id):
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
    body_ids = body_and_descendants(model, item_body_id)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] in body_ids:
            model.geom_contype[gid]     = PLACED_ITEM_BIT
            model.geom_conaffinity[gid] = model.geom_conaffinity[gid] & ~ROBOT_BIT

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


def solve_plan_ik(kin, plan):
    """Pre-solve arm IK for every (verb, location, object) the plan needs."""
    ik_cache = {}
    for action_name in plan:
        verb, args = parse_action(action_name)
        if verb not in ("pick", "place"):
            continue
        _, loc, obj = args
        x, y, th = POSES[loc]
        q0  = arm_home_q0(x, y, th)
        bp  = base_pose_mat(x, y, np.radians(th))
        ori = kin.fk(q0, bp)["arm"][:3, :3]
        item = ITEMS[obj]
        target = item["item_pos"] if verb == "pick" else item["place_pos"]
        res = solve_ik(kin, x, y, np.radians(th), q0, target, ori)
        if res is None:
            raise RuntimeError(f"IK failed for {verb} {loc} {obj}")
        ik_cache[(verb, loc, obj)] = res["arm"]
        if verb == "pick":
            lift_target = item["item_pos"] + np.array([0.0, 0.0, 0.33])
            res_lift = solve_ik(kin, x, y, np.radians(th), q0, lift_target, ori)
            if res_lift is None:
                raise RuntimeError(f"Lift IK failed for {loc} {obj}")
            ik_cache[("lift", loc, obj)] = res_lift["arm"]
    return ik_cache


def plan_executor(base_mg, arm_mg, gripper_mg, model, data, item_ids, ik_cache, plan):
    """Generator: replays the symbolic plan as physical control steps.

    Each yield advances exactly one simulation step's worth of control,
    mirroring robot_task_gen() in robot_demo_two_robots.py but driven by
    plan verbs (move/pick/place) instead of a hand-authored phase list.
    """
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

    base_target = pose_to_ctrl(POSES["start"])
    arm_target  = HOME_JOINTS.copy()
    grip_target = GRIPPER_OPEN
    carry_fn    = None

    for action_name in plan:
        verb, args = parse_action(action_name)

        if verb == "move":
            _, loc_from, loc_to = args
            print(f"[plan] NAVIGATE {loc_from} -> {loc_to}")
            new_base = pose_to_ctrl(POSES[loc_to])

            def nav_ctrl():
                base_mg.ctrl    = new_base
                arm_mg.ctrl     = arm_target
                gripper_mg.ctrl = grip_target
                if carry_fn:
                    carry_fn()
            yield from run(secs_to_steps(NAV_DURATION), nav_ctrl)
            base_target = new_base
            yield from hold(secs_to_steps(SETTLE_AFTER_NAV), base_target, arm_target, grip_target)

        elif verb == "pick":
            _, loc, obj = args
            print(f"[plan] REACH & GRASP {obj} at {loc}")
            arm_grasp = ik_cache[("pick", loc, obj)]
            arm_start = arm_mg.joint_pos.copy()

            def reach_ctrl(t):
                base_mg.ctrl    = base_target
                arm_mg.ctrl     = lerp(arm_start, arm_grasp, t)
                gripper_mg.ctrl = GRIPPER_OPEN
            yield from run_smooth(secs_to_steps(REACH_DURATION), reach_ctrl)

            print(f"  [{obj}] Closing gripper...")
            yield from hold(secs_to_steps(GRIP_CLOSE_DUR), base_target, arm_grasp, GRIPPER_CLOSE)
            yield from hold(secs_to_steps(GRIP_SETTLE_DUR), base_target, arm_grasp, GRIPPER_CLOSE)

            mujoco.mj_forward(model, data)
            body_id, jnt_id = item_ids[obj]
            tcp_pos      = arm_mg.leaf_frame_to_world[:3, 3].copy()
            item_pos_now = data.xpos[body_id].copy()
            offset = item_pos_now - tcp_pos
            print(f"  [{obj}] offset‖={np.linalg.norm(offset):.3f}m")
            if np.linalg.norm(offset) > 0.30:
                print(f"  [{obj}] WARNING: offset > 0.30m — arm may have missed the object.")

            arm_lift = ik_cache[("lift", loc, obj)]

            def carry():
                tcp = arm_mg.leaf_frame_to_world[:3, 3]
                set_item_pose(model, data, jnt_id, tcp + offset)

            def lift_ctrl(t):
                base_mg.ctrl    = base_target
                arm_mg.ctrl     = lerp(arm_grasp, arm_lift, t)
                gripper_mg.ctrl = GRIPPER_CLOSE
                carry()
            yield from run_smooth(secs_to_steps(LIFT_DURATION), lift_ctrl)

            carry_fn    = carry
            arm_target  = arm_lift
            grip_target = GRIPPER_CLOSE

        elif verb == "place":
            _, loc, obj = args
            print(f"[plan] LOWER & PLACE {obj} at {loc}")
            arm_place   = ik_cache[("place", loc, obj)]
            arm_at_lift = arm_mg.joint_pos.copy()

            def lower_ctrl(t):
                base_mg.ctrl    = base_target
                arm_mg.ctrl     = lerp(arm_at_lift, arm_place, t)
                gripper_mg.ctrl = GRIPPER_CLOSE
                if carry_fn:
                    carry_fn()
            yield from run_smooth(secs_to_steps(LOWER_DURATION), lower_ctrl)

            print(f"  [{obj}] Releasing...")
            yield from hold(secs_to_steps(RELEASE_DUR), base_target, arm_place, GRIPPER_OPEN)

            body_id, _ = item_ids[obj]
            detach_item_from_gripper(model, body_id)
            carry_fn = None

            arm_at_place = arm_mg.joint_pos.copy()

            def retract_ctrl(t):
                base_mg.ctrl    = base_target
                arm_mg.ctrl     = lerp(arm_at_place, HOME_JOINTS, t)
                gripper_mg.ctrl = GRIPPER_OPEN
            yield from run_smooth(secs_to_steps(RETRACT_DURATION), retract_ctrl)
            arm_target  = HOME_JOINTS.copy()
            grip_target = GRIPPER_OPEN

        else:
            raise ValueError(f"Unsupported action verb in plan: {verb!r}")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Only run symbolic planning and print the plan; skip MuJoCo simulation/rendering.",
    )
    args = parser.parse_args()

    print("Planning (railroad MCTSPlanner)...")
    plan = generate_plan()
    print("Plan found:")
    for i, action_name in enumerate(plan, 1):
        print(f"  {i}. {action_name}")

    if args.plan_only:
        return

    # ── Scene ──────────────────────────────────────────────
    print("\nInstalling scene assets...")
    install_scene_from_source_index(SCENE_NAME, SCENE_INDEX)
    spec = MjSpec.from_file(SCENE_PATH)

    # ── Robot ──────────────────────────────────────────────
    print("Adding MobileFranka robot to scene...")
    sx, sy, sth_deg = POSES["start"]
    sth_rad = np.radians(sth_deg)
    robot_cfg = MobileFrankaRobotConfig(base_size=[0.5, 0.5, 0.75], robot_namespace=ROBOT_PREFIX)
    robot_cfg.init_qpos["base"] = [sx, sy, sth_rad]

    MobileFrankaRobot.add_robot_to_scene(
        robot_cfg, spec,
        prefix=ROBOT_PREFIX,
        pos=[sx, sy],
        quat=R.from_euler("z", sth_rad).as_quat(scalar_first=True),
    )
    MobileFrankaRobot.apply_control_overrides(spec, robot_cfg)

    # ── Compile ────────────────────────────────────────────
    print("Compiling model...")
    model = spec.compile()
    data  = MjData(model)

    # Disable collision on the robot base so it can glide to targets.
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{ROBOT_PREFIX}base")
    if base_body_id >= 0:
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == base_body_id:
                model.geom_contype[gid]    = 0
                model.geom_conaffinity[gid] = 0

    # Give the arm/gripper their own collision bit so a placed item's
    # conaffinity can later exclude just the gripper (see detach_item_from_gripper).
    for gid in range(model.ngeom):
        body_id = model.geom_bodyid[gid]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith(ROBOT_PREFIX) and body_name != f"{ROBOT_PREFIX}base":
            model.geom_contype[gid]    = ROBOT_BIT
            model.geom_conaffinity[gid] = 1

    # Disable collision on the dining chairs tucked tight around table 1 so
    # the arm can't wedge against one while reaching into the table.
    for gid in range(model.ngeom):
        body_id = model.geom_bodyid[gid]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith(TABLE1_CHAIR_PREFIX):
            model.geom_contype[gid]    = 0
            model.geom_conaffinity[gid] = 0

    # ── Init robot state ───────────────────────────────────
    view = robot_cfg.robot_view_factory(data, ROBOT_PREFIX)
    view.set_qpos_dict(robot_cfg.init_qpos)

    mujoco.mj_forward(model, data)
    for mg_id in view.move_group_ids():
        mg = view.get_move_group(mg_id)
        mg.ctrl = mg.noop_ctrl
    mujoco.mj_forward(model, data)

    base_mg    = view.get_move_group("base")
    arm_mg     = view.get_move_group("arm")
    gripper_mg = view.get_move_group("gripper")

    # ── Item IDs ───────────────────────────────────────────
    item_ids = {}
    for obj, info in ITEMS.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, info["body_name"])
        jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, info["jnt_name"])
        if body_id < 0 or jnt_id < 0:
            raise RuntimeError(f"Item body/joint not found in model for {obj!r}")
        item_ids[obj] = (body_id, jnt_id)

    # ── Pre-solve IK for every pick/place the plan needs ───
    print("Solving IK targets for the plan...")
    kin_cfg = MobileFrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
    kin = MlSpacesKinematics(kin_cfg)
    ik_cache = solve_plan_ik(kin, plan)
    print("  IK solved ✓")

    # ── Camera & renderer ──────────────────────────────────
    cam      = make_camera()
    renderer = mujoco.Renderer(model, height=H, width=W)

    # ── Simulation loop ────────────────────────────────────
    frames     = []
    step_count = [0]

    def render_tick():
        mujoco.mj_step(model, data)
        step_count[0] += 1
        if step_count[0] % RENDER_EVERY == 0:
            frames.append(grab_frame(renderer, data, cam))

    for _ in range(secs_to_steps(SCENE_SETTLE)):
        render_tick()

    for _ in plan_executor(base_mg, arm_mg, gripper_mg, model, data, item_ids, ik_cache, plan):
        render_tick()

    # ── Write video ────────────────────────────────────────
    print(f"\nRendered {len(frames)} frames. Writing {OUTPUT_PATH}...")
    with imageio.get_writer(OUTPUT_PATH, fps=FPS, codec="libx264",
                            quality=7, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)

    duration = len(frames) / FPS
    print(f"Done!  {OUTPUT_PATH}  {W}×{H}  {FPS}fps  {duration:.1f}s")


if __name__ == "__main__":
    main()

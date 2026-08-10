"""Replays a railroad `PlanResult` (see common.py) inside a MolmoSpaces
MuJoCo scene built by molmospace_scene.py.

Deliberately split from scene/domain construction: a `PlanExecutor` is built
once against a compiled `MolmoSpaceScene` and can then render any number of
plans (e.g. from different planners, or different goals) against that same
scene, without rebuilding it.

Fidelity:
  - `move` drives the robot's actual base pose along a collision-aware
    Theta* path (molmospace_navigation.OccupancyGrid) between the scene's
    real table positions, instead of a straight line that can clip through
    the table sitting in between on a diagonal move.
  - `pick`/`place` solve real IK (molmo_spaces' MlSpacesKinematics) to drive
    the arm down to the object/table, then activate/deactivate a weld
    equality constraint between the gripper's grasp site and the object's
    grasp anchor site while real physics (`mj_step`, not just
    `mj_forward`) is running -- so the held object has genuine gravity,
    contact, and collision behavior, and a released object settles onto
    the table for real instead of being teleported into place. The
    gripper fingers are deliberately never actuated closed around the
    object -- the weld already does 100% of the holding, and driving the
    fingers to a near-zero-gap ctrl target around a rigid object the weld
    is *also* holding exactly in place pits two stiff constraints against
    each other, which is unstable (confirmed: cube velocities spiking to
    5-7 m/s and flying off mid-carry before this was removed).
  - The robot's own base/arm joints are still kinematically driven (direct
    qpos writes, re-asserted every physics substep) rather than actuated
    through the position controllers -- this is what keeps `move`
    "collision-aware kinematic path" rather than full dynamic locomotion,
    while still letting every free body (the 3 grabbable objects) live
    under real MuJoCo dynamics throughout.

Each action verb is handled by its own `_apply_*` method (see
`ACTION_HANDLERS`) so adding a new domain operator only means adding one
more handler -- nothing else about the executor changes.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from railroad.navigation import pathing

from common import PlanStep  # pyrefly: ignore [missing-import]
from molmospace_navigation import OccupancyGrid  # pyrefly: ignore [missing-import]
from molmospace_scene import (  # pyrefly: ignore [missing-import]
    MolmoSpaceScene,
    ROBOT_HOME,
    ROBOT_PREFIXES,
    STAND_THETA_DEG,
    WORKSTATION_LOCATIONS,
    resting_spot,
    stand_pose,
    weld_name,
)

REACH_PHASE = 0.5   # fraction of a pick/place action spent reaching down; the
                     # rest is spent retracting back to the neutral arm pose
ARM_JOINT_NAMES = tuple(f"fr3_joint{i + 1}" for i in range(7))
BASE_JOINT_NAMES = ("base_x", "base_y", "base_theta")
CHECKPOINT_FRACS = (0.0, 0.25, 0.5, 0.75)  # re-snapshot the other robot's position at each of these points through a move


def _lerp_vec(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def _base_pose_mat(x: float, y: float, theta_rad: float) -> np.ndarray:
    bp = np.eye(4)
    bp[0, 3] = x
    bp[1, 3] = y
    bp[:2, :2] = R.from_euler("z", theta_rad).as_matrix()[:2, :2]
    return bp


class PlanExecutor:
    def __init__(self, scene: MolmoSpaceScene):
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.registry = scene.registry
        self.kin = scene.kinematics
        self.robots = list(ROBOT_HOME.keys())
        self.grid = OccupancyGrid()

        self._robot_pose: Dict[str, Tuple[float, float, float]] = {
            r: stand_pose(ROBOT_HOME[r], r) for r in self.robots
        }
        self._arm_neutral: Dict[str, np.ndarray] = {
            r: np.array(scene.robot_configs[r].init_qpos["arm"], dtype=float) for r in self.robots
        }
        self._arm_qpos: Dict[str, np.ndarray] = {r: self._arm_neutral[r].copy() for r in self.robots}
        self._held: Dict[str, Optional[str]] = {r: None for r in self.robots}

        self._move_paths: Dict[tuple, Dict[float, np.ndarray]] = {}
        self._reach_cache: Dict[tuple, np.ndarray] = {}
        self._weld_ids = {
            (r, obj): mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name(r, obj))
            for r in self.robots
            for obj in ("cube_1", "cube_2", "scissors")
        }

        self.ACTION_HANDLERS = {
            "move": self._apply_move,
            "pick": self._apply_pick,
            "place": self._apply_place,
            "cut_paper": self._apply_static,
            "wrap_gift": self._apply_static,
            "cut_ribbon": self._apply_static,
            "complete_job": self._apply_static,
            "wait_for_resource": self._apply_static,
            "no_op": self._apply_static,
        }

    # -- weld / gripper control ----------------------------------------------

    def _set_weld(self, robot: str, obj: str, active: bool) -> None:
        self.data.eq_active[self._weld_ids[(robot, obj)]] = 1 if active else 0

    # -- IK reach target (cached per robot/location/object) ------------------

    def _reach_qpos(self, robot: str, obj: str, loc: str) -> np.ndarray:
        key = (robot, loc, obj)
        cached = self._reach_cache.get(key)
        if cached is not None:
            return cached

        cfg = self.scene.robot_configs[robot]
        x, y, theta_deg = stand_pose(loc, robot)
        base_pose = _base_pose_mat(x, y, np.radians(theta_deg))
        neutral_arm = self._arm_neutral[robot]
        q0 = {
            "base": np.array([x, y, np.radians(theta_deg)]),
            "arm": neutral_arm,
            "gripper": np.array(cfg.init_qpos["gripper"], dtype=float),
        }
        ori = self.kin.fk(q0, base_pose)["arm"][:3, :3]
        target = np.eye(4)
        target[:3, 3] = resting_spot(obj, loc)
        target[:3, :3] = ori
        result = self.kin.ik("arm", target, ["arm"], q0, base_pose, max_iter=4000)
        reach_arm = np.array(result["arm"], dtype=float) if result is not None else neutral_arm
        self._reach_cache[key] = reach_arm
        return reach_arm

    # -- per-verb action handlers -----------------------------------------------
    # Each takes (step, frac) where frac in [0, 1] is progress through the
    # action's [start, end) window.

    def _apply_move(self, step: PlanStep, frac: float) -> None:
        loc_from, loc_to = step.action.split()[2:4]
        checkpoint, next_checkpoint = self._checkpoint_window(frac)
        path = self._path_for_move(step, step.robot, loc_from, loc_to, checkpoint)
        total = pathing.path_total_length(path)
        span = next_checkpoint - checkpoint
        local_frac = min(max((frac - checkpoint) / span, 0.0), 1.0) if span > 0 else 1.0
        if total > 0:
            x, y = pathing.get_coordinates_at_distance(path, local_frac * total)
        else:
            x, y = path[0, 0], path[1, 0]
        self._robot_pose[step.robot] = (float(x), float(y), STAND_THETA_DEG)

    def _apply_pick(self, step: PlanStep, frac: float) -> None:
        loc, obj = step.action.split()[2:4]
        self._drive_grasp(step.robot, obj, loc, frac, picking=True)

    def _apply_place(self, step: PlanStep, frac: float) -> None:
        loc, obj = step.action.split()[2:4]
        self._drive_grasp(step.robot, obj, loc, frac, picking=False)

    def _apply_static(self, step: PlanStep, frac: float) -> None:
        """cut_paper / wrap_gift / cut_ribbon / complete_job / wait_for_resource
        / no_op: arm/base hold their current pose; nothing to drive here --
        a held object keeps following the gripper via its (already active)
        weld constraint under real physics."""

    def _drive_grasp(self, robot: str, obj: str, loc: str, frac: float, picking: bool) -> None:
        neutral = self._arm_neutral[robot]
        reach = self._reach_qpos(robot, obj, loc)
        if frac <= REACH_PHASE:
            phase_frac = frac / REACH_PHASE if REACH_PHASE > 0 else 1.0
            self._arm_qpos[robot] = _lerp_vec(neutral, reach, phase_frac)
            weld_active = not picking  # place: still carrying it down
        else:
            phase_frac = (frac - REACH_PHASE) / (1.0 - REACH_PHASE)
            self._arm_qpos[robot] = _lerp_vec(reach, neutral, phase_frac)
            weld_active = picking      # pick: now grasped, retracting with it
        self._set_weld(robot, obj, weld_active)

    # -- collision-aware move path (re-checked at a few points per move) ----

    def _checkpoint_window(self, frac: float) -> Tuple[float, float]:
        """Which [checkpoint, next_checkpoint) window `frac` falls in --
        CHECKPOINT_FRACS divides a move into a few segments so the other
        robot's obstacle position gets re-snapshotted partway through a long
        move instead of only once at its very start (see _path_for_move)."""
        checkpoint = max(cp for cp in CHECKPOINT_FRACS if cp <= frac)
        later = [cp for cp in CHECKPOINT_FRACS if cp > checkpoint]
        next_checkpoint = later[0] if later else 1.0
        return checkpoint, next_checkpoint

    def _path_for_move(self, step: PlanStep, robot: str, loc_from: str, loc_to: str, checkpoint: float) -> np.ndarray:
        """Theta* path for the segment starting at `checkpoint` (a fraction
        of the whole move's duration). Re-snapshots the other robot's
        *current* position each time a new checkpoint is reached, instead of
        relying on a single stale snapshot taken back at the move's start --
        that staleness is what let two robots each independently plan a
        clear path at t=0 and then walk straight through each other once
        their paths actually crossed mid-transit.

        Starts from the robot's current interpolated position (not the
        move's original start) for every checkpoint after the first, so the
        rendered path is continuous across a re-route -- only its remaining
        shape changes, not the robot's actual position.
        """
        key = (robot, step.start, step.end)
        segments = self._move_paths.setdefault(key, {})
        cached = segments.get(checkpoint)
        if cached is not None:
            return cached

        start_xy = stand_pose(loc_from, robot)[:2] if checkpoint == 0.0 else self._robot_pose[robot][:2]
        end_xy = stand_pose(loc_to, robot)[:2]
        other = next(r for r in self.robots if r != robot)
        blocked_xy = self._robot_pose[other][:2]
        path = self.grid.path(start_xy, end_xy, blocked_xy=blocked_xy)
        segments[checkpoint] = path
        return path

    # -- plan replay --------------------------------------------------------

    def _active_step(self, steps: List[PlanStep], robot: str, t: float) -> Optional[PlanStep]:
        best = None
        for s in steps:
            if s.robot == robot and s.start <= t <= s.end and (best is None or s.start > best.start):
                best = s
        return best

    def _sync_completed_pick_place(self, steps: List[PlanStep], robot: str, t: float) -> None:
        """Snap weld/held state to whatever the most recently *finished*
        pick/place is, independent of whether any rendered frame happened to
        land inside that step's [REACH_PHASE, 1] window.

        `_drive_grasp` only toggles the weld while its step is still active
        (`_active_step` requires t <= s.end); a short pick/place window
        relative to the frame spacing can easily fall between two rendered
        frames, so relying solely on hitting the right frac inside an active
        step is unreliable. Recomputing from full plan history every frame
        instead makes this self-correcting.
        """
        completed = sorted(
            (s for s in steps if s.robot == robot and s.end <= t and s.action.split()[0] in ("pick", "place")),
            key=lambda s: s.end,
        )
        if not completed:
            return
        verb, _robot, _loc, obj = completed[-1].action.split()[:4]
        held = verb == "pick"
        self._set_weld(robot, obj, held)
        self._held[robot] = obj if held else None

    def _apply_step(self, step: PlanStep, t: float) -> None:
        verb = step.action.split()[0]
        duration = step.end - step.start
        frac = 0.0 if duration <= 0 else (t - step.start) / duration
        frac = min(max(frac, 0.0), 1.0)
        self.ACTION_HANDLERS.get(verb, self._apply_static)(step, frac)

    def _write_kinematics(self) -> None:
        """Re-assert both robots' base/arm joints at their current cached
        target and zero their velocity, right before each physics substep --
        keeps the robots kinematically puppeteered while every free body
        (the 3 grabbable objects) still evolves under real dynamics/contact.
        """
        for robot in self.robots:
            x, y, theta_deg = self._robot_pose[robot]
            prefix = ROBOT_PREFIXES[robot]
            cfg = self.scene.robot_configs[robot]
            view = cfg.robot_view_factory(self.data, prefix)
            view.set_qpos_dict({"base": [x, y, np.radians(theta_deg)], "arm": self._arm_qpos[robot]})
            for jname in (*BASE_JOINT_NAMES, *ARM_JOINT_NAMES):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{jname}")
                self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0

    def render_video(
        self,
        steps: List[PlanStep],
        out_path: str,
        fps: int = 20,
        seconds_per_plan_second: float = 0.15,
        width: int = 960,
        height: int = 540,
        physics_substeps: Optional[int] = None,
    ) -> str:
        """Render `steps` (a PlanResult.steps list -- works for the output of
        any of the 5 planners in decentralized.py/centralized.py) into a
        video at `out_path`, one shared timeline so concurrent robot actions
        show up simultaneously.

        `physics_substeps` (real `mj_step` calls per rendered frame)
        defaults to however many the model's own timestep needs to cover
        one frame's worth of *plan* time -- e.g. at 200 frames over a 14s
        plan, each frame represents ~0.07 plan-seconds, so with the
        model's 0.002s timestep that's ~35 substeps. A fixed small count
        (e.g. 4) covers far less real time than that per frame, so a
        welded/held object has to move at an unrealistically high velocity
        to keep up with the kinematically-teleported gripper -- that
        mismatch is what caused held cubes to destabilize and go flying
        instead of riding along smoothly. Only override this if you know
        what you're doing.
        """
        makespan = max((s.end for s in steps), default=0.0)
        n_frames = max(int(makespan * seconds_per_plan_second * fps), fps // 2)
        if physics_substeps is None:
            plan_seconds_per_frame = makespan / (n_frames - 1) if n_frames > 1 else makespan
            physics_substeps = max(1, round(plan_seconds_per_frame / self.model.opt.timestep))

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        xs = [p[0] for p in WORKSTATION_LOCATIONS.values()]
        ys = [p[1] for p in WORKSTATION_LOCATIONS.values()]
        cam.lookat[:] = [sum(xs) / len(xs), sum(ys) / len(ys) - 0.3, 0.9]
        cam.distance = 8.0
        cam.azimuth = 200.0
        cam.elevation = -40.0

        print(f"Rendering {n_frames} frames ({n_frames / fps:.1f}s @ {fps}fps) to {out_path}...")
        with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7, macro_block_size=1) as writer:
            for i in range(n_frames):
                t = 0.0 if n_frames <= 1 else (i / (n_frames - 1)) * makespan
                for robot in self.robots:
                    self._sync_completed_pick_place(steps, robot, t)
                    step = self._active_step(steps, robot, t)
                    if step is not None:
                        self._apply_step(step, t)

                for _ in range(physics_substeps):
                    self._write_kinematics()
                    mujoco.mj_step(self.model, self.data)

                mujoco.mj_forward(self.model, self.data)
                renderer.update_scene(self.data, camera=cam)
                writer.append_data(renderer.render())

        print(f"Done! Wrote {out_path}")
        return out_path

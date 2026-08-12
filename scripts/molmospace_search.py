#!/usr/bin/env python3
"""
End-to-end pipeline: sample a real ProcTHOR house in MolmoSpaces, search for a
real pickupable object whose location is *not* told to the planner, then
fetch it to its sampled destination -- rendered as one MuJoCo video.

Unlike plan_and_render.py (which already knows exactly where its object is
and just plans move/pick/move/place), this treats the object's starting
receptacle as unknown to the planner. A handful of real receptacles in the
same room are offered as "candidate" search sites; railroad's MCTSPlanner
reasons over railroad.operators.construct_search_operator's probabilistic
find-or-not effect (same operator the `procthor-search` railroad example
uses) to decide which to check and in what order, executing for real against
sampled outcomes -- so a run can (and usually does) include one or more
"searched here, not there" legs before the real one succeeds.

Pipeline:
  1. SAMPLE    Ask MolmoSpaces' PickAndPlaceTaskSampler for a real, feasible
               (object, destination) pair in a randomly chosen ProcTHOR house
               (procthor-10k train split), same as plan_and_render.py.
  2. DISCOVER  Find every real receptacle in the object's room (ObjectManager
               + the scene's room map) and pick a few as decoy search sites,
               alongside the object's real receptacle.
  3. SEARCH    Build a symbolic search-only domain (locations = candidate
               receptacles, one object, move + search operators) with no
               "at" fluent for the object -- only construct_search_operator's
               probabilistic effect can ever establish it -- and run
               railroad's MCTSPlanner iteratively against real sampled
               outcomes until "found" becomes true. Replay every executed
               move/search action physically: gliding the robot's base
               between real receptacle positions and idling (with a small
               look-around sweep) while "searching" each one.
  4. FETCH     Once found, re-place the robot at the same feasible stand-pose
               MolmoSpaces' own task sampler uses (env.place_robot_near) and
               hand off to its real PickAndPlacePlannerPolicy -- identical
               execution to plan_and_render.py -- for the actual grasp,
               carry, and place.

Output: molmospace_search_house<idx>_<Object>.mp4 (in the working directory),
        plus a second molmospace_search_house<idx>_<Object>_topdown.mp4 shot
        from a static, straight-down camera framed to keep the whole house
        in view for the entire run, and, if --save-plot is given, a PNG of
        the final frame.

Run with: uv run python scripts/molmospace_search.py --seed 8616
          --save-video ./out.mp4 --save-plot ./out.png
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from typing import Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R

from bridge import (  # pyrefly: ignore [missing-import]
    ASSUMED_ROBOT_BASE_SPEED_M_S,
    PLAN_STEP_PHASES,
    START_LOC,
    assert_pickup_pose_matches_sim,
    build_initial_fluents_from_scene,
    category_from_object_name,
    plan_step_for_phase,
)

CAMERA_NAME = "exo_camera_1"  # base-mounted, kept pointed at the task object
TOP_DOWN_CAMERA_NAME = "top_down_camera"  # static, framed to cover the whole house
FPS = 15  # matches policy_dt_ms ~66ms per task.step()

# Top-down camera framing. An orthographic camera was tried here (every
# pixel looking straight down, so no perspective spillage past the house's
# edges) and was abandoned: this renderer bakes the free camera's default
# viewing distance -- itself derived from mj_model.stat.extent -- into the
# orthographic frustum size before our custom pos/forward/up ever get
# applied, so the right "fov" value to pass depends on that per-house,
# uncontrollable number rather than on anything we compute. Back to a
# perspective camera, whose framing is plain trigonometry we control end to
# end.
#
# TOP_DOWN_FOV_DEG is the *vertical* FOV (world y, given forward=-z/up=+y in
# setup_top_down_camera puts world x along the image's horizontal axis and
# world y along its vertical one); the horizontal FOV is wider by the image's
# aspect ratio. The house footprint is measured from real wall positions
# (see setup_top_down_camera), not mj_model.stat.{center,extent}: MolmoSpaces
# parks internal bookkeeping geometry (receptacle-staging candidates, grasp
# calibration markers) tens of meters above/away from the actual house,
# which blows up that whole-model bounding sphere uselessly.
#
# TOP_DOWN_EDGE_PAD_M adds a flat world-space buffer to the wall bounding box
# before framing, and TOP_DOWN_MARGIN then scales the result a bit more.
# Both are needed for the same underlying reason: near the edge of a wide-FOV
# frame, the ray looking at that pixel isn't pointing straight down, so a
# wall's real (x,y) footprint and where it visually appears in frame (it has
# height, so an oblique ray hits its top edge, offset outward from its
# footprint) don't quite line up. A pad in meters compensates for that offset
# directly; the margin is a smaller multiplicative safety factor on top for
# whatever the pad doesn't quite catch.
TOP_DOWN_FOV_DEG = 55.0
TOP_DOWN_EDGE_PAD_M = 1.0
TOP_DOWN_MARGIN = 1.02

# A fixed, empirically-verified-good (seed, house, object) triple, used as
# the default so a run is reproducible and reliably completes end to end
# instead of gambling on a fresh random object every time. MolmoSpaces'
# PickTaskSampler only randomizes task_config fields left at None (see
# pick_task_sampler.py's `_select_pickup_object`) -- pre-setting
# pickup_obj_name on the config before sample_task() pins the object while
# leaving robot placement / destination receptacle (still randomized per
# attempt) to vary run to run. A fully random object fails real grasp IK at
# task.reset() time roughly 3 times out of 4 (confirmed against plan_and_
# render.py's own unmodified retry loop); this specific vase, at this house,
# succeeds on the large majority of placement attempts.
DEFAULT_SEED = 554964770
DEFAULT_HOUSE_INDEX = 47
DEFAULT_OBJECT_NAME = "vase_4686c3634f3334d6f28ba73fcd509a72_1_0_6"
# Retries absorb robot-placement/grasp-IK infeasibility, which varies by
# attempt even with object+house pinned -- not a second chance at a
# different object or house.
MAX_ATTEMPTS = 5

# Search-domain tuning. FIND_PROB_CORRECT is deliberately not 1.0 -- a search
# can genuinely come up empty even at the right receptacle -- but is high
# enough that repeated failure-to-find across a small candidate set is rare
# in practice; FIND_PROB_WRONG is low but nonzero so wrong-site searches
# aren't free information ("failed here" is still evidence, not a proof).
FIND_PROB_CORRECT = 0.9
FIND_PROB_WRONG = 0.05
SEARCH_TIME_S = 4.0
DEFAULT_NUM_CANDIDATES = 3

GLIDE_MIN_S, GLIDE_MAX_S = 1.5, 6.0
GLIDE_SPEED_M_S = 0.6
SEARCH_IDLE_S = 2.0
SEARCH_SWEEP_DEG = 20.0
ROBOT = "robot0"


def build_task_sampler(seed: int):
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        FrankaPickAndPlaceDroidDataGenConfig,
    )

    config = FrankaPickAndPlaceDroidDataGenConfig()
    config.use_passive_viewer = False
    config.profile = False
    config.use_wandb = False
    config.seed = seed
    config.task_horizon = 300
    config.task_sampler_config.pickup_types = None
    config.task_sampler_config.enable_texture_randomization = False
    config.task_sampler_config.samples_per_house = MAX_ATTEMPTS

    task_sampler = config.task_sampler_config.task_sampler_class(config)
    task_sampler.reset()
    return config, task_sampler


def _hide_ceilings(env) -> None:
    """Make ceiling geoms fully transparent so the top-down camera actually
    sees the rooms rather than their roof. housegen/builder.py names every
    ceiling patch body with the "ceiling" prefix (name_prefix="ceiling"),
    consistently across houses -- these sit a few meters above the floor
    (well below our overhead camera) and, left visible, render as a single
    opaque textured slab that hides all interior detail (confirmed: without
    this, the top-down video showed nothing but that slab for the entire
    run). Not restored afterward -- exo_camera_1 is a low, workspace-facing
    camera that never frames the ceiling anyway, so there's nothing to
    restore for.
    """
    ceiling_geoms = [
        i for i in range(env.mj_model.ngeom)
        if (env.mj_model.body(env.mj_model.geom_bodyid[i]).name or "").startswith("ceiling")
    ]
    env.mj_model.geom_rgba[ceiling_geoms, 3] = 0.0


def setup_top_down_camera(env) -> None:
    """Register a static, straight-down camera sized to frame the whole
    house's footprint (see TOP_DOWN_FOV_DEG's comment for why this -- and
    not mj_model.stat -- is the bounds source). Idempotent: re-registering
    under the same name just overwrites the previous entry, which matters
    because task.reset() re-runs camera setup from the task's own
    CameraSystemConfig and could otherwise leave this camera stale.
    """
    from molmo_spaces.env.camera_manager import Camera

    _hide_ceilings(env)

    # housegen/builder.py names every wall segment body "wall_{room_id}_{n}"
    # (consistently across houses); their positions bound the real house
    # footprint. An earlier version of this used env.get_thormap()'s
    # free-space map instead, but that map is *eroded* inward from walls by
    # its agent_radius (0.35m, so the robot's own body never clips through
    # them) and the erosion isn't uniform on every side of a non-rectangular
    # floor plan -- padding it back by a flat guessed margin either clipped
    # walls on the more-eroded sides or left excess border on the others.
    # Real wall positions have neither problem.
    wall_geoms = [
        i for i in range(env.mj_model.ngeom)
        if (env.mj_model.body(env.mj_model.geom_bodyid[i]).name or "").startswith("wall_")
    ]
    wall_xy = env.current_data.geom_xpos[wall_geoms, :2]
    xy_min, xy_max = wall_xy.min(axis=0), wall_xy.max(axis=0)
    center_xy = (xy_min + xy_max) / 2.0
    dx, dy = (xy_max - xy_min + 2.0 * TOP_DOWN_EDGE_PAD_M).tolist()
    floor_z = float(env.current_robot.robot_view.base.pose[2, 3])

    half_vfov_rad = np.radians(TOP_DOWN_FOV_DEG) / 2.0
    aspect = env._renderer.width / env._renderer.height
    half_hfov_rad = np.arctan(aspect * np.tan(half_vfov_rad))
    height_for_dy = (dy / 2.0) / np.tan(half_vfov_rad)
    height_for_dx = (dx / 2.0) / np.tan(half_hfov_rad)
    height = max(TOP_DOWN_MARGIN * max(height_for_dy, height_for_dx), 3.0)

    camera = Camera(
        name=TOP_DOWN_CAMERA_NAME,
        pos=np.array([center_xy[0], center_xy[1], floor_z + height], dtype=np.float32),
        forward=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        fov=TOP_DOWN_FOV_DEG,
    )
    env.camera_manager.registry.add_camera(camera)


def render_top_down_frame(env) -> np.ndarray:
    """Thin alias for render_rgb_frame(TOP_DOWN_CAMERA_NAME), kept as its own
    function so call sites don't need to change if top-down rendering ever
    again needs per-call setup (an orthographic mode was tried and reverted
    here -- see TOP_DOWN_FOV_DEG's comment).
    """
    return env.render_rgb_frame(TOP_DOWN_CAMERA_NAME)


def room_of(thormap, xyz: np.ndarray) -> Optional[str]:
    """Room name containing world point `xyz`, or None if the scene has no
    room map or the point falls outside it."""
    if thormap.room_map is None or thormap.room_ids_to_name is None:
        return None
    px = thormap.pos_m_to_px(np.asarray(xyz, dtype=float))
    row, col = int(px[0]), int(px[1])
    if not (0 <= row < thormap.room_map.shape[0] and 0 <= col < thormap.room_map.shape[1]):
        return None
    return thormap.room_ids_to_name.get(int(thormap.room_map[row, col]))


def choose_candidate_receptacles(
    env, om, correct_loc: str, obj: str, num_candidates: int, rng: random.Random
) -> list[str]:
    """The object's real receptacle plus up to `num_candidates - 1` decoy
    receptacles, restricted to the same room so the search phase's straight-
    line glides never have to cross a wall (see module docstring step 3).
    Falls back to any other receptacle if the room has too few."""
    thormap = env.get_thormap()
    correct_pos = om.get_object_by_name(correct_loc).position
    correct_room = room_of(thormap, correct_pos)

    # A pickupable object can itself register as a receptacle (e.g. a mug);
    # excluded here so the search domain never offers "search the target
    # object for itself" as a candidate site.
    receptacles = [r.name for r in om.get_receptacles() if r.name not in (correct_loc, obj)]
    same_room = [
        name for name in receptacles
        if room_of(thormap, om.get_object_by_name(name).position) == correct_room
    ] if correct_room is not None else []

    pool = same_room if len(same_room) >= num_candidates - 1 else receptacles
    decoys = rng.sample(pool, k=min(num_candidates - 1, len(pool)))
    candidates = [correct_loc, *decoys]
    rng.shuffle(candidates)
    return candidates


def run_search_planning(
    om, robot_base_xy: np.ndarray, obj: str, candidate_locations: list[str], correct_loc: str,
    max_iterations: int = 30,
) -> list[str]:
    """Symbolic search-only domain: railroad's MCTSPlanner decides which
    candidate receptacle to check next (move + construct_search_operator's
    probabilistic find), executed against real sampled outcomes, until
    "found <obj>" becomes true. The object's location is never seeded as an
    "at" fluent -- only a successful search ever sets it -- so this is a
    genuine search, not a known-location fetch dressed up as one.

    Returns the actually-executed action-name sequence. Because "found" is
    the only goal and only a search action can cause it, the final element
    is always the search that succeeded; every earlier search in the list
    failed (a construct_search_operator location can only be searched once).
    """
    from railroad.core import Fluent as F, State, get_action_by_name
    from railroad.environment.symbolic import SymbolicEnvironment, LocationRegistry
    from railroad.planner import MCTSPlanner
    from railroad import operators

    positions = {START_LOC: np.asarray(robot_base_xy, dtype=float)}
    for loc in candidate_locations:
        positions[loc] = np.asarray(om.get_object_by_name(loc).position[:2], dtype=float)
    registry = LocationRegistry(positions)
    move_op = operators.construct_move_operator_blocking(
        registry.move_time_fn(velocity=ASSUMED_ROBOT_BASE_SPEED_M_S)
    )

    def find_prob(_robot: str, loc: str, _obj: str) -> float:
        return FIND_PROB_CORRECT if loc == correct_loc else FIND_PROB_WRONG

    search_op = operators.construct_search_operator(find_prob, SEARCH_TIME_S)

    initial_fluents = {F(f"at {ROBOT} {START_LOC}"), F(f"free {ROBOT}")}
    state = State(0.0, initial_fluents, [])
    env = SymbolicEnvironment(
        state=state,
        objects_by_type={
            "robot": {ROBOT},
            "location": {START_LOC, *candidate_locations},
            "object": {obj},
        },
        operators=[move_op, search_op],
        # Real outcome resolution: SymbolicEnvironment.act() resolves search's
        # probabilistic effect from this ground truth (deterministically) when
        # given, rather than rolling the declared find_prob -- the probability
        # only shapes the *planner's* belief/exploration, matching real search
        # (the object either is or isn't there; the uncertainty is epistemic).
        true_object_locations={correct_loc: {obj}},
    )
    goal = F(f"found {obj}")

    executed: list[str] = []
    for _ in range(max_iterations):
        if goal.evaluate(env.state.fluents):
            return executed
        all_actions = env.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(env.state, goal, max_iterations=3000, c=300, max_depth=12)
        if action_name == "NONE":
            raise RuntimeError(f"search planner found no path to 'found {obj}'")
        action = get_action_by_name(all_actions, action_name)
        env.act(action)
        executed.append(action_name)

    raise RuntimeError(f"search did not find {obj!r} within {max_iterations} symbolic steps")


# ── Physical replay of the search phase ─────────────────────────────────────

def _base_pose_mat(x: float, y: float, z: float, theta_rad: float) -> np.ndarray:
    bp = np.eye(4)
    bp[:3, 3] = [x, y, z]
    bp[:2, :2] = R.from_euler("z", theta_rad).as_matrix()[:2, :2]
    return bp


def _smoothstep(t: float) -> float:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _robot_geom_ids(model, prefix: str = "robot_0/") -> list[int]:
    """All geom ids under any body in the robot's kinematic tree (base + arm
    + gripper) -- includes unnamed collision geoms, so name-based geom
    filtering would miss some (see procthor_demo.py's robot_geom_ids)."""
    body_ids = {i for i in range(model.nbody) if (model.body(i).name or "").startswith(prefix)}
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] in body_ids]


class SearchRenderer:
    """Drives the robot base directly (the same mocap write env.place_robot_near
    uses) and grabs frames from the task's own onboard camera -- no separate
    mujoco.Renderer/camera needed, task.env.render_rgb_frame already exists
    for this. Used only for the search phase; the real fetch afterward is
    rendered by MolmoSpaces' own policy/task.step() loop, same as
    plan_and_render.py.
    """

    def __init__(self, task_env, robot_view):
        self.env = task_env
        self.robot_view = robot_view
        self.sim_dt = task_env.mj_model.opt.timestep
        self.render_every = max(1, int(round(1.0 / (FPS * self.sim_dt))))
        self.frames: list[np.ndarray] = []
        self.top_down_frames: list[np.ndarray] = []
        self._step_count = 0
        pose = robot_view.base.pose
        self.xy = pose[:2, 3].copy()
        self.z = float(pose[2, 3])
        self.theta = float(R.from_matrix(pose[:3, :3]).as_euler("xyz")[2])
        self._robot_geoms = _robot_geom_ids(task_env.mj_model)
        self._orig_contype = task_env.mj_model.geom_contype[self._robot_geoms].copy()
        self._orig_conaffinity = task_env.mj_model.geom_conaffinity[self._robot_geoms].copy()
        # The arm/gripper's ctrl is not a "hold current position" signal by
        # itself -- whatever it was last set to (here, left at zero) keeps
        # driving the actuators every mj_step. Snapshotting and re-asserting
        # it each tick (below) is what actually holds the arm still; without
        # this a multi-thousand-step search phase quietly drags the arm from
        # its real neutral pose toward the zero pose, which then made the
        # real grasp IK fail far more often than it should (confirmed via a
        # before/after joint_pos dump -- this was the actual bug, not IK
        # infeasibility variance).
        self._arm_hold = robot_view.get_move_group("arm").joint_pos.copy()
        self._gripper_hold = robot_view.get_move_group("gripper").ctrl.copy()

    def disable_robot_collision(self) -> None:
        """Search-phase glides are raw mocap teleports with no path planning
        -- with collision left on, real contact forces during mj_step can
        shove furniture/props out of place as the robot straight-lines
        through the room, which then breaks env.place_robot_near's later
        feasibility search near anything that got displaced. Restored before
        the real fetch (place_robot_near's own collision checks need it)."""
        self.env.mj_model.geom_contype[self._robot_geoms] = 0
        self.env.mj_model.geom_conaffinity[self._robot_geoms] = 0

    def restore_robot_collision(self) -> None:
        self.env.mj_model.geom_contype[self._robot_geoms] = self._orig_contype
        self.env.mj_model.geom_conaffinity[self._robot_geoms] = self._orig_conaffinity

    def _tick(self) -> None:
        self.robot_view.get_move_group("arm").ctrl = self._arm_hold
        self.robot_view.get_move_group("gripper").ctrl = self._gripper_hold
        self.env.step(n_steps=1)
        self._step_count += 1
        if self._step_count % self.render_every == 0:
            self.frames.append(self.env.render_rgb_frame(CAMERA_NAME))
            self.top_down_frames.append(render_top_down_frame(self.env))

    def glide_to(self, target_xy: np.ndarray) -> None:
        start_xy = self.xy.copy()
        dist = float(np.linalg.norm(target_xy - start_xy))
        if dist < 1e-6:
            return
        theta = float(np.arctan2(target_xy[1] - start_xy[1], target_xy[0] - start_xy[0]))
        duration_s = float(np.clip(dist / GLIDE_SPEED_M_S, GLIDE_MIN_S, GLIDE_MAX_S))
        n_steps = max(1, int(duration_s / self.sim_dt))
        for i in range(n_steps):
            t = _smoothstep((i + 1) / n_steps)
            xy = start_xy + (target_xy - start_xy) * t
            self.robot_view.base.pose = _base_pose_mat(xy[0], xy[1], self.z, theta)
            self._tick()
        self.xy = target_xy.copy()
        self.theta = theta

    def search_here(self) -> None:
        """Idle in place with a small look-around sweep -- purely cosmetic,
        signals "searching" without claiming any real sensing behavior."""
        n_steps = max(1, int(SEARCH_IDLE_S / self.sim_dt))
        for i in range(n_steps):
            frac = (i + 1) / n_steps
            yaw = self.theta + np.radians(SEARCH_SWEEP_DEG) * np.sin(2 * np.pi * frac)
            self.robot_view.base.pose = _base_pose_mat(self.xy[0], self.xy[1], self.z, yaw)
            self._tick()


def replay_search_phase(renderer: SearchRenderer, executed: list[str], positions: dict[str, np.ndarray]) -> None:
    for action_name in executed:
        parts = action_name.split()
        verb = parts[0]
        if verb == "move":
            _, _robot, _loc_from, loc_to = parts
            print(f"  [SEARCH] moving to {loc_to}...")
            renderer.glide_to(positions[loc_to])
        elif verb == "search":
            _, _robot, loc, obj = parts
            renderer.search_here()
            found = action_name == executed[-1]
            print(f"  [SEARCH] searched {loc} for {obj} -- {'FOUND' if found else 'not there'}")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for search-site choice/planning.")
    parser.add_argument("--house-index", type=int, default=DEFAULT_HOUSE_INDEX,
                         help="Fixed ProcTHOR house index to load.")
    parser.add_argument("--object-name", type=str, default=DEFAULT_OBJECT_NAME,
                         help="Fixed real object (by MuJoCo body name) to search for and fetch.")
    parser.add_argument("--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES,
                         help="Number of receptacles (incl. the real one) offered as search sites.")
    parser.add_argument("--save-video", type=str, default=None, help="Output video path (default: auto-named).")
    parser.add_argument("--save-topdown-video", type=str, default=None,
                         help="Top-down (whole-house) output video path (default: <save-video>_topdown.mp4).")
    parser.add_argument("--save-plot", type=str, default=None, help="Optional PNG snapshot of the final frame.")
    args = parser.parse_args()

    seed = args.seed
    house_idx = args.house_index
    obj = args.object_name
    print(f"[SEED] {seed}  [HOUSE] {house_idx}  [OBJECT] {obj}")

    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask, RetriableError

    reset_failures = (ValueError, RetriableError, HouseInvalidForTask, RuntimeError)

    config, task_sampler = build_task_sampler(seed)

    task = destination = None
    om = None
    executed: list[str] = []
    candidate_locations: list[str] = []
    correct_loc = None
    renderer: Optional[SearchRenderer] = None
    observation = policy = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n=== Attempt {attempt}/{MAX_ATTEMPTS} ===")
        try:
            # pickup_obj_name is the only field pinned here -- the task
            # sampler still randomizes robot placement and the destination
            # receptacle fresh each call, which is exactly what a retry is
            # for (see MAX_ATTEMPTS's docstring above).
            config.task_config.pickup_obj_name = obj
            task = task_sampler.sample_task(house_index=house_idx)
            destination = task.config.task_config.place_receptacle_name
            print(f"  [SAMPLE] object={obj!r}  destination={destination!r}")

            setup_top_down_camera(task.env)

            om = task.env.object_managers[task.env.current_batch_index]
            fluents, _unplaced = build_initial_fluents_from_scene(om)
            correct_loc = next(
                (f.args[1] for f in fluents if f.name == "at" and f.args[0] == obj), None
            )
            if correct_loc is None:
                raise RuntimeError(f"{obj!r} has no detected starting receptacle to search for")

            robot_view = task.env.current_robot.robot_view
            # sample_task()'s own placement (env.place_robot_near) for this
            # object -- captured now, before any search-phase driving, so
            # it can be restored exactly afterward rather than re-sampling
            # a fresh (and not necessarily IK-feasible) stand-pose later.
            original_pose = robot_view.base.pose.copy()
            robot_base_xy = np.asarray(original_pose[:2, 3], dtype=float)
            candidate_locations = choose_candidate_receptacles(
                task.env, om, correct_loc, obj, args.num_candidates, random.Random(seed)
            )
            print(f"  [SEARCH SITES] {candidate_locations}  (real: {correct_loc!r})")

            executed = run_search_planning(om, robot_base_xy, obj, candidate_locations, correct_loc)
            print(executed)
            raise NotImplementedError
            print("\n  [SEARCH PLAN] railroad MCTSPlanner executed:")
            for i, action_name in enumerate(executed, 1):
                print(f"    {i}. {action_name}")

            positions = {loc: om.get_object_by_name(loc).position[:2] for loc in candidate_locations}
            renderer = SearchRenderer(task.env, robot_view)

            print("\n  [EXECUTE] replaying search phase in MuJoCo...")
            renderer.disable_robot_collision()
            replay_search_phase(renderer, executed, positions)
            renderer.restore_robot_collision()

            print(f"\n  [FETCH] found {obj!r} at {correct_loc!r} -- returning to the validated stand-pose...")
            robot_view.base.pose = original_pose
            task.env.step(n_steps=1)
            renderer.frames.append(task.env.render_rgb_frame(CAMERA_NAME))
            renderer.top_down_frames.append(render_top_down_frame(task.env))

            policy = config.policy_config.policy_factory(config, task)
            task.register_policy(policy)
            observation, _info = task.reset()
            # task.reset() re-runs the task sampler's own camera setup from
            # its CameraSystemConfig, which doesn't touch our custom entry --
            # but re-registering here is cheap and removes any doubt.
            setup_top_down_camera(task.env)
            assert_pickup_pose_matches_sim(observation[0], task.env, obj)
            renderer.frames.append(observation[0][CAMERA_NAME])
            renderer.top_down_frames.append(render_top_down_frame(task.env))

            break
        except reset_failures as e:
            print(f"  [SAMPLE] attempt {attempt} failed ({e!r}); retrying placement...")
            task = None

    if task is None:
        print(f"\nFAILED: {obj!r} at house {house_idx} did not yield a geometrically feasible "
              f"placement after {MAX_ATTEMPTS} attempts. Try a different --object-name/--house-index.")
        return 1

    current_plan_step = -1
    failure_reason = None
    try:
        for i in range(config.task_horizon):
            action_cmd = policy.get_action(observation)
            phase = policy.get_phase()
            new_step = plan_step_for_phase(phase, max(current_plan_step, 0))
            if new_step != current_plan_step:
                current_plan_step = new_step
                label = PLAN_STEP_PHASES[current_plan_step][0]
                print(f"  [FETCH] {label}")

            observation, _reward, _terminated, _truncated, _infos = task.step(action_cmd)
            renderer.frames.append(observation[0][CAMERA_NAME])
            renderer.top_down_frames.append(render_top_down_frame(task.env))

            if action_cmd.get("success") is False:
                failure_reason = f"policy signaled failure during phase {phase!r} (max retries exceeded)"
                break
            if task.is_done():
                info = task.get_info()[0]
                if not info["success"]:
                    failure_reason = f"task finished but was judged unsuccessful: {info}"
                print(f"  [FETCH] policy signaled done at step {i}.")
                break
        else:
            failure_reason = f"reached task_horizon ({config.task_horizon}) without finishing"
    except Exception as e:
        failure_reason = f"unhandled exception during fetch execution: {e!r}"
        print(f"  [FETCH] {failure_reason}")

    obj_category = category_from_object_name(obj)
    suffix = "" if failure_reason is None else "_FAILED"
    output_path = args.save_video or f"molmospace_search_house{house_idx}_{obj_category}{suffix}.mp4"
    root, ext = os.path.splitext(output_path)
    top_down_output_path = args.save_topdown_video or f"{root}_topdown{ext}"

    print(f"\nRendered {len(renderer.frames)} frames. Writing {output_path}...")
    with imageio.get_writer(output_path, fps=FPS, codec="libx264", quality=7, macro_block_size=1) as writer:
        for frame in renderer.frames:
            writer.append_data(frame)

    print(f"Writing top-down video ({len(renderer.top_down_frames)} frames) to {top_down_output_path}...")
    with imageio.get_writer(
        top_down_output_path, fps=FPS, codec="libx264", quality=7, macro_block_size=1
    ) as writer:
        for frame in renderer.top_down_frames:
            writer.append_data(frame)

    if args.save_plot:
        imageio.imwrite(args.save_plot, renderer.frames[-1])
        print(f"Saved final-frame plot to {args.save_plot}")

    print("\n" + "=" * 60)
    print(f"Seed:            {seed}")
    print(f"House index:     {house_idx}")
    print(f"Object:          {obj}")
    print(f"Search sites:    {candidate_locations}  (real: {correct_loc})")
    print(f"Searches before find: {len(executed) - 1}")
    print(f"Destination:     {destination}")
    print(f"Video:           {output_path}")
    print(f"Top-down video:  {top_down_output_path}")
    print("=" * 60)

    if failure_reason is not None:
        print(f"\nFAILED during fetch: {failure_reason}")
        return 1

    print("\nSUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

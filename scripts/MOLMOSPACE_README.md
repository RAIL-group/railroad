# MolmoSpaces plan + simulation pipeline

Bridges the `railroad` symbolic planner to a real MuJoCo/MolmoSpaces
simulation: plan the box-station gift-wrap task symbolically, then replay
the resulting plan as an actual MobileFranka manipulation in a ProcTHOR
scene, with real collision-aware navigation and real physics-based
grasping.

Five files, each independently reusable:

| File | Responsibility |
|---|---|
| [molmospace_scene.py](molmospace_scene.py) | Builds the MuJoCo scene: 4 tables, 2 gift cubes, scissors, 2 MobileFranka robots. Knows nothing about planning. |
| [molmospace_domain.py](molmospace_domain.py) | Builds a `common.Domain` (see [README.md](README.md)) from that scene's real table positions, so `railroad` can plan over it. |
| [molmospace_navigation.py](molmospace_navigation.py) | Collision-aware Theta* paths for `move`, used by the executor. |
| [molmospace_executor.py](molmospace_executor.py) | Replays a planner's `PlanResult.steps` against a built scene: drives real IK, weld-constraint grasping, and MuJoCo physics, and renders a video. |
| [molmospace_demo.py](molmospace_demo.py) | Wires the above four together into one runnable script. |

**Pipeline:** `build_scene()` → `build_domain(scene)` → `plan_reservation(domain)` (or any of the other 4 planners in [decentralized.py](decentralized.py)/[centralized.py](centralized.py)) → `PlanExecutor(scene).render_video(result.steps)`.

The scene only needs to be built once — a single `MolmoSpaceScene` can be
replayed against any number of different plans (different planners,
different goals) via a fresh `PlanExecutor`.

---

## molmospace_scene.py — scene construction

Builds a standard ProcTHOR home layout with the dining table/chairs
stripped out, replaced by 4 free-standing tables named directly after the
box-station domain's own locations (`workstation1`/`workstation2` — each
robot's own table; `tool_space`/`box_station` — shared), 2 gift cubes, a
scissors tool, and 2 MobileFranka robots.

| Function / class | What it does |
|---|---|
| `resting_spot(obj, loc)` | World xyz where `obj` sits when resting on table `loc` (table center + a small per-object offset so cube_1/cube_2 don't spawn on top of each other). |
| `stand_pose(loc, robot)` | `(x, y, theta_deg)` where `robot` stands (0.7m south, facing north) to work at table `loc`, offset laterally per robot (`ROBOT_LATERAL_OFFSET`, ±0.25m) so two robots sent to the same *shared* table (tool_space/box_station) get distinct side-by-side spots instead of the identical point — they used to be sent to literally the same coordinate, which is what caused two robots to visually overlap/pass through each other. |
| `grasp_anchor_name(obj_id)` | Name of the site on `obj_id`'s body that a robot's gripper weld attaches to (`"{obj_id}/grasp"`). |
| `weld_name(robot, obj_id)` | Name of the (initially inactive) weld equality constraint between `robot`'s gripper and `obj_id` (`"{robot}_{obj_id}_weld"`). |
| `ObjectRegistry` | Maps clear string IDs (`"cube_1"`, `"scissors"`, ...) to compiled MuJoCo body names. `.register()`, `.body_name()`, `.body_id(model)`, `.world_pos(model, data)`, `.ids()`. |
| `remove_clashing_furniture(spec, scene_meta, root_names)` | Deletes the source scene's dining table/chairs and everything logically resting on them (walks the scene metadata's parent/child graph, since deleting a table body alone would leave its tabletop items behind as orphaned clutter). |
| `add_table(spec, table_id, pos_xy)` | Spawns a static 4-legged table body at `pos_xy`. |
| `add_cube(spec, cube_id, pos, rgba)` | Spawns a free-jointed 6cm cube with a grasp anchor site. |
| `add_scissors(spec, scissors_id, pos)` | Spawns a free-jointed scissors-like tool (two crossed blades + handles + pivot, built from primitive geoms) with a grasp anchor site. |
| `add_robot_near(spec, prefix, base_xy_theta_deg)` | Spawns a MobileFranka robot at the given base pose, namespaced under `prefix`. Returns its `MobileFrankaRobotConfig`. |
| `MolmoSpaceScene` | Dataclass bundling everything a plan replay needs: `model`, `data`, `registry`, `robot_configs`, `robot_prefixes`, `location_positions`, `kinematics`. |
| `build_scene(settle=True)` | Top-level entry point. Installs the ProcTHOR scene, strips the clashing dining set, spawns the 4 tables + cubes + scissors + 2 robots, wires up all 6 (robot × object) weld equality constraints (inactive by default), compiles the model, and lets physics settle for 1s so objects rest naturally before any plan executes. Returns a `MolmoSpaceScene`. |

## molmospace_domain.py — symbolic planning domain

Builds the same box-station gift-wrap `Domain` as
[pick_and_place_astar_boxstation.py](pick_and_place_astar_boxstation.py),
but `move`'s duration comes from the *real* distances between the scene's
tables instead of a hardcoded layout. Only the domain-specific operators
live here — pick/place/wait/no_op are deliberately left out; each planning
method in decentralized.py/centralized.py builds whichever variant of
those it needs (see [common.Domain](common.py)'s docstring).

| Function | What it does |
|---|---|
| `_make_move_time(location_positions)` | Returns a `Numeric` duration function: real Euclidean distance between two tables' positions, divided by `ROBOT_SPEED` (0.5 m/s). |
| `construct_accessible_move_operator(location_positions)` | The `move` operator — 3-stage effect (clear `free`/`at-from` at t=0, set `free`/`at-to`/`just-moved` at `move_time`, clear `just-moved` 0.1s later as a re-move debounce). Gated by `accessible ?r ?to` so a robot can't enter the other robot's workstation. |
| `construct_cut_paper_operator()` | 20s action: robot must be at a workstation, holding scissors, with the gift box present. |
| `construct_wrap_gift_operator()` | 10s action: empty hands, paper already cut. |
| `construct_cut_ribbon_operator()` | 20s action: holding scissors again, gift already wrapped. |
| `construct_complete_job_operator()` | 10s action: empty hands, ribbon already cut → `wrapped_gift`. |
| `default_robot_goals()` | robot1 wraps `cube_1`, robot2 wraps `cube_2`, both also returning the scissors to `tool_space` so the other robot can acquire it. |
| `build_domain(scene, robot_goals=None, pick_time=1.0, place_time=1.0)` | Assembles `objects_by_type`, `initial_state` (scissors at tool_space, both cubes at box_station, each robot at its own home workstation with the right `accessible`/`is_workstation` fluents), the 5 operators above (with `move` bound to the scene's real table positions), and `contested_resources={"scissors": "tool_space"}` into a `common.Domain`. |

## molmospace_navigation.py — collision-aware move paths

A straight line between two tables' stand-points can clip diagonally
through whichever table sits in between (e.g. `workstation1` →
`box_station` cuts across the 2x2 grid). This module builds a small
occupancy grid from the table footprints and routes `move` through it with
Theta* (`railroad.navigation.pathing`, the same any-angle planner already
used elsewhere in this repo for ProcTHOR navigation) instead. Kinematic
only — no dynamics, just a smarter path for the executor's qpos writes to
follow.

| Method (on `OccupancyGrid`) | What it does |
|---|---|
| `__init__()` | Builds a static grid covering the table layout (+ margin), marking each table's footprint (inflated by a clearance margin) as an obstacle. |
| `world_to_grid(x, y)` / `grid_to_world(row, col)` | Convert between world meters and grid cell indices. |
| `_mark_rect(grid, cx, cy, half_x, half_y)` | Marks a rectangular region as an obstacle (used both for tables at init time and for the other robot's position per-query). |
| `path(start_xy, end_xy, blocked_xy=None)` | Returns a `(2, N)` array of world-space waypoints from `start_xy` to `end_xy` via Theta*, optionally treating `blocked_xy` (the other robot's current position) as an extra temporary obstacle. Falls back to a direct 2-point line if no path is found. Called repeatedly per move (see `_path_for_move` below), not just once, so `blocked_xy` reflects a reasonably current position rather than a single stale snapshot. |

## molmospace_executor.py — plan replay + physics

The core of the pipeline: takes a compiled `MolmoSpaceScene` and any
planner's `PlanResult.steps` and actually drives the simulation.

**Fidelity model:**
- `move` — real interpolated base motion along the Theta* path above (kinematic: direct qpos writes, not actuator-driven — this is what keeps it "collision-aware path" rather than full dynamic locomotion).
- `pick`/`place` — real IK (`MlSpacesKinematics`) drives the arm down to the object/table, then a weld equality constraint between the gripper's grasp site and the object's grasp anchor site is activated/deactivated while **real physics** (`mj_step`, not just `mj_forward`) runs — so a held object has genuine gravity/contact/collision behavior, and a released object settles onto the table for real. The gripper fingers are deliberately never actuated closed around the object (see below).
- Every other action (`cut_paper`, `wrap_gift`, `cut_ribbon`, `complete_job`, `wait_for_resource`, `no_op`) — arm/base hold their current pose; a held object just keeps following the gripper via its already-active weld.

| Method (on `PlanExecutor`) | What it does |
|---|---|
| `__init__(scene)` | Caches per-robot base pose, neutral arm qpos, current arm qpos target, held-object state, the `OccupancyGrid`, and looks up all 6 weld equality IDs by name. |
| `_set_weld(robot, obj, active)` | Toggles a (robot, object) weld constraint on/off (`data.eq_active`). |
| `_reach_qpos(robot, obj, loc)` | Solves IK once (cached per robot/location/object) for the arm pose that puts the gripper's grasp site at `resting_spot(obj, loc)`, keeping the neutral pose's orientation. Falls back to the neutral pose if IK fails to converge. |
| `_apply_move(step, frac)` | Works out which checkpoint window `frac` falls in (`_checkpoint_window`), fetches that segment's Theta* path (`_path_for_move`), interpolates the robot's world (x, y) along it by arc-length using the *local* fraction within that segment, and updates the cached base-pose target. |
| `_apply_pick(step, frac)` / `_apply_place(step, frac)` | Parse `loc`/`obj` from the action name and delegate to `_drive_grasp`. |
| `_apply_static(step, frac)` | No-op — base/arm targets are simply left at whatever they currently are. |
| `_drive_grasp(robot, obj, loc, frac, picking)` | The 2-phase pick/place animation: for the first half of the action (`frac <= REACH_PHASE`), the arm interpolates from neutral toward the IK reach pose; for the second half, it interpolates back to neutral. The weld is toggled active/inactive at the `REACH_PHASE` midpoint (active while "grasped", i.e. retracting after a pick or still descending during a place). |
| `_checkpoint_window(frac)` | Which `[checkpoint, next_checkpoint)` window `frac` falls in, per `CHECKPOINT_FRACS = (0.0, 0.25, 0.5, 0.75)`. |
| `_path_for_move(step, robot, loc_from, loc_to, checkpoint)` | Theta* path for the segment starting at `checkpoint` (cached per move-step-instance *and* checkpoint). Re-snapshots the other robot's *current* position each time a new checkpoint is reached instead of relying on one stale snapshot from the move's start — the original one-shot version let two robots each independently plan a clear path at t=0 and then walk straight through each other once their paths actually crossed mid-transit (both instrumental in the "robot moves through the other robot" bug). Every checkpoint after the first starts from the robot's current interpolated position, not the move's original start, so the rendered path stays continuous across a re-route. |
| `_active_step(steps, robot, t)` | Finds whichever of `robot`'s steps is currently in progress at plan-time `t`. |
| `_sync_completed_pick_place(steps, robot, t)` | Self-correcting catch-up: recomputes weld/held state from the most recently *finished* pick/place (by `end <= t`), independent of whether any rendered frame happened to land inside that step's active window — a short pick/place relative to frame spacing can otherwise fall between two frames and leave the weld stuck in the wrong state (this is what caused the original "cube stuck to hand" bug). |
| `_apply_step(step, t)` | Computes `frac` for the currently active step and dispatches to the right handler via `ACTION_HANDLERS`. |
| `_write_kinematics()` | Re-asserts both robots' base/arm qpos at their current cached target and zeroes their velocity — called before every physics substep so the robots stay kinematically puppeteered while the 3 grabbable objects (genuinely free bodies) evolve under real dynamics/contact in between. |
| `render_video(steps, out_path, fps=20, seconds_per_plan_second=0.15, ..., physics_substeps=None)` | Main loop: for each output frame, syncs weld state, drives the active step for each robot, then runs `physics_substeps` real `mj_step` calls (re-asserting kinematics before each one) before rendering. `physics_substeps` auto-computes from how much *plan* time each frame represents divided by the model's own timestep — a fixed small count would make a welded object move at an unrealistic velocity to keep up with the kinematically-teleported gripper (this is what caused the "cubes get dropped/flung" bug — see the module docstring for the full story). |

## molmospace_demo.py — entry point

```python
scene = build_scene()
domain = build_domain(scene, robot_goals=default_robot_goals())
result = plan_reservation(domain, verbose=True)   # <- swap for any of the other 4 planners
executor = PlanExecutor(scene)
executor.render_video(result.steps, "molmospace_demo.mp4")
```

Run: `uv run python scripts/molmospace_demo.py` → `molmospace_demo.mp4`.

To try a different planner, swap the `PLANNER` module-level constant for
any of `plan_reactive`/`plan_reservation`/`plan_no_op_blind`
(decentralized.py) or `plan_joint_astar`/`plan_joint_mcts`
(centralized.py) — the scene doesn't need to be rebuilt. To try different
goals, pass a different `robot_goals` dict to `build_domain`.

## Known limitations / things to be aware of

- `move`'s planned duration (molmospace_domain.py) is still based on
  straight-line distance between tables, but the physical path
  (molmospace_navigation.py) is longer since it detours around tables —
  so the rendered robot covers the real path slightly faster than a truly
  physical robot moving at `ROBOT_SPEED` would. Not fixed; would need
  `_make_move_time` to route through `OccupancyGrid.path` too.
- The other robot's position is re-checked at 4 fixed checkpoints per move
  (`CHECKPOINT_FRACS`), not continuously-updated spacetime planning — a
  pathological case where the other robot changes direction sharply
  between two checkpoints could still get closer than intended.
- The *symbolic* plan still has no notion that a physical location has
  limited capacity — `tool_space`/`box_station` are `accessible` to both
  robots with no concurrency limit, so the plan itself can (and, in the
  default 2-gift scenario, does) schedule both robots to the same shared
  table at the same time. `ROBOT_LATERAL_OFFSET` + the checkpoint
  re-routing above keep this physically plausible at execution time (they
  end up ~0.5m apart, not overlapping), but the plan doesn't know to avoid
  or stagger it. Fixing that would mean modeling each shared location as a
  contested resource the same way `scissors` already is (see
  `common.Domain`'s `contested_resources`), so `plan_reservation` would
  serialize concurrent visits with a real wait — deliberately not done
  yet, since it changes plan timing/cost and wasn't in scope.
- Object orientation while held isn't explicitly controlled (the weld's
  `eq_data` relpose is left at identity) — MuJoCo's site-type weld forces
  the two sites to coincide positionally regardless of `eq_data`, so this
  mostly doesn't matter, but exact held-object rotation isn't guaranteed.

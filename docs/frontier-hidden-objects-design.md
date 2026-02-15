# Frontier Hidden Objects: Standalone Design Specification (vs `origin/main`)

Date: 2026-02-14  
Baseline: `origin/main` at `42c3139`

## 1. Problem Statement

Build a planning environment where multiple robots can:

1. Explore an initially unknown occupancy map.
2. Continuously update the map while moving.
3. Track and refresh frontiers (free cells adjacent to unknown space).
4. Unlock hidden search sites once the corresponding map cells become observed.
5. Search unlocked sites to find target objects.
6. Replan whenever new information invalidates/changes the best plan.

This document is a **self-contained functional + technical spec**. It defines required behavior, data contracts, algorithms, and test criteria without requiring access to prior implementation history.

## 2. Scope

### In scope

- Unknown-space navigation and sensing loop.
- Frontier extraction and lifecycle.
- Symbolic-planning integration (objects, fluents, actions).
- Multi-robot concurrency controls for navigation/search.
- End-to-end hidden-object scenario: find all target objects.
- Integration against provided starter modules:
  - `packages/railroad/src/railroad/environment/navigation/constants.py`
  - `packages/railroad/src/railroad/environment/navigation/mapping.py`
  - `packages/railroad/src/railroad/environment/navigation/pathing.py`

### Out of scope

- GUI/dashboard polish.
- Benchmarking UX and profiling tooling.
- New planning algorithm research; planner can remain MCTS-compatible.

## 2.1 Provided Starter Files (Assumed Existing)

This spec presumes the following files are already provided and should be treated as integration dependencies:

- `packages/railroad/src/railroad/environment/navigation/constants.py`
- `packages/railroad/src/railroad/environment/navigation/mapping.py`
- `packages/railroad/src/railroad/environment/navigation/pathing.py`

Required contracts from provided files:

- `constants.py`:
  - `FREE_VAL`, `COLLISION_VAL`, `UNOBSERVED_VAL`, `OBSTACLE_THRESHOLD`
- `mapping.py`:
  - `insert_scan(...) -> (updated_grid, newly_observed_count)`
- `pathing.py`:
  - `build_traversal_costs(...)`
  - `get_cost_and_path(...)` (or equivalent path+cost function)
  - `path_total_length(...)`
  - `get_coordinates_at_distance(...)`

If starter files expose different function names/signatures, add a thin adapter layer; do not rewrite their internals unless required for correctness.

## 3. Reference Scenario (Ground Truth)

Use this canonical scenario to validate behavior:

- Robots: `robot1`, `robot2`.
- Target objects: `Mug`, `Knife`.
- Map: branching corridor grid with obstacles/walls.
- Hidden sites: `stash_east`, `stash_north`, `stash_west` (fixed coordinates in map frame).
- Object placement example:
  - `stash_east` contains `Mug`.
  - `stash_north` contains `Knife`.
  - `stash_west` empty.
- Goal: `found Mug AND found Knife`.

Initial symbolic state (minimum):

- `(at robot1 start)`, `(free robot1)`
- `(at robot2 start2)`, `(free robot2)`
- `(revealed start)`, `(revealed start2)`
- `(candidate-site stash_east)`, `(candidate-site stash_north)`, `(candidate-site stash_west)`

Initial map state:

- `true_grid`: full occupancy (known only to simulator/runtime internals).
- `observed_grid`: all `UNOBSERVED` except cells seen by initial sensor sweep.

Alternate map source (supported):

- A ProcTHOR scene may be used as the source occupancy grid for search/navigation.
- Seed `8617` is a known large, high-coverage candidate for this scenario.
- When using ProcTHOR:
  - derive `true_grid` from scene traversability/obstacle geometry,
  - keep the same planner-facing contracts (`observed_grid`, frontiers, `navigable`, hidden-site unlock/search).

## 4. Core Concepts and Data Model

## 4.1 Time and State Semantics

Runtime state is:

- `time: float`
- `fluents: Set[Fluent]`
- `upcoming_effects: List[(effect_time, grounded_effect)]`

Action effects may be:

- Immediate (`time=0`)
- Delayed (`time>0`)
- Probabilistic branch effects

Execution is event-driven: the environment advances to the next scheduled skill/effect time.

## 4.2 Types

Required object types:

- `robot`: mobile agents.
- `location`: all planner-usable locations (base locations + dynamic frontiers + transient robot locations).
- `frontier`: dynamic subset of `location` generated from observed map.
- `object`: target object symbols (`Mug`, `Knife`, etc.).

Hidden-object candidates such as `stash_*` are regular `location` objects (no separate site type).

## 4.3 Fluents

Required fluent vocabulary:

Navigation/execution:

- `(at ?r ?loc)`
- `(free ?r)`
- `(navigable ?loc)`
- `(claimed ?loc)`

Observation/discovery:

- `(looked-for-site ?frontier ?site)`
- `(revealed ?loc)`
- `(candidate-site ?loc)` (static marker for hidden-object candidate locations)

Search bookkeeping:

- `(searched ?loc ?obj)`
- `(lock-search ?loc)`

Task success:

- `(found ?obj)`
- Optional completion signal: `(exploration-complete)` when no frontiers remain.

## 4.4 Map Representation

Use occupancy grid conventions:

- `FREE_VAL = 0.0`
- `COLLISION_VAL = 1.0`
- `UNOBSERVED_VAL = -1.0`
- `OBSTACLE_THRESHOLD = 0.5`

Structures:

- `true_grid: np.ndarray[rows, cols]` with free/occupied truth.
- `observed_grid: np.ndarray[rows, cols]` with unknown + fused occupancy values.

## 4.5 Robot Pose Model

Each robot has continuous pose in grid coordinates:

- `x: float` (row axis)
- `y: float` (col axis)
- `yaw: float` radians

Pose is updated during move execution micro-steps.

Coordinate convention:

- A grid cell `(row, col)` maps to pose `(x=row, y=col)`.
- Row increases downward in array indexing, but visualization may render with origin at bottom-left.
- All pathing/frontier logic must use a single consistent convention to avoid swapped axes.

## 4.6 Navigation Runtime Config (Recommended Defaults)

Use the following config parameters and defaults unless scenario-specific tuning is needed:

- `sensor_range: 9.0` cells
- `sensor_fov_rad: 2*pi` (360 degrees)
- `sensor_num_rays: 181`
- `sensor_dt: 0.08` sec
- `speed_cells_per_sec: 2.0`
- `occupied_prob: 0.9`
- `unoccupied_prob: 0.1`
- `connect_neighbor_distance: 2` (ray endpoint stitching)
- `interrupt_min_new_cells: 20`
- `interrupt_min_dt: 1.0` sec
- `correct_with_known_map: true` (sim-only stabilization)
- `record_frames: true` (optional diagnostics)

Tuning guidance:

- Increase `interrupt_min_new_cells` to reduce replanning churn.
- Increase `sensor_dt` to reduce compute cost (less frequent sensing).
- Lower `sensor_num_rays` for speed if map fidelity remains acceptable.

## 5. Required Module-Level Additions vs `origin/main`

Note: items in provided starter files are assumed present. The work described below is the remaining implementation and integration delta.

## 5.1 Environment Runtime Hooks

Add interrupt extension points in base environment runtime:

- `should_interrupt_active_skills() -> bool`
- `clear_interrupt_request() -> None`

`act()` loop contract:

1. Start skill for selected action.
2. Apply immediate effects.
3. While no robot is free:
   - If interrupt requested: stop loop.
   - Advance all skills to nearest next event time.
   - Apply due effects.
4. If interrupted:
   - call `interrupt()` on active interruptible skills.
   - clear interrupt request.
5. Return current assembled state.

Action validity filters in runtime action generation:

- Reject `move` where source == destination.
- Reject `move` whose effects are all zero-duration (degenerate).
- Reject move/search/place targeting transient `*_loc` symbols.

## 5.2 Optional Navigation Package

Add optional package `railroad.environment.navigation` with extras dependency group:

- `scipy`
- `scikit-image`
- `imageio`
- `imageio-ffmpeg`

Package responsibilities:

- Laser simulation.
- Occupancy-map scan insertion/fusion (via provided `mapping.py`).
- Path planning utilities (via provided `pathing.py`).
- Frontier extraction.
- Unknown-space environment class.
- Navigation move skill.

Implementation expectation in this effort:

- Reuse provided `constants.py`, `mapping.py`, and `pathing.py`.
- Implement missing modules around them (`types.py`, `laser.py`, `frontiers.py`, `environment.py`, `skill.py`).

## 5.3 Unknown-Space Environment Contract

### Required fields

- `true_grid`
- `observed_grid`
- `robot_poses: Dict[robot, pose]`
- `frontiers: Dict[frontier_id, Frontier]`
- `base_locations: Set[str]` (static scenario locations)
- `location_registry` (maps symbolic locations to numeric coordinates)
- interrupt counters/timestamps:
  - `new_cells_since_interrupt`
  - `last_interrupt_time`

### Required methods

- `observe_from_pose(robot, pose, time, allow_interrupt=True) -> newly_observed_count`
- `observe_all_robots(time=None, allow_interrupt=True) -> total_new_cells`
- `refresh_frontiers() -> None`
- `compute_move_path(loc_from, loc_to) -> 2xN path`
- `estimate_move_time(...)` (observed-map-based)
- `is_cell_observed(row, col) -> bool`

### Object model updates performed by `refresh_frontiers()`

Given the newly extracted reachable frontiers:

1. `objects_by_type["frontier"] = current_frontier_ids`
2. `objects_by_type["location"] = base_locations U frontier_ids U current_robot_locations`
3. Register each frontier id in `location_registry` with frontier centroid coordinates
4. Set `(exploration-complete)` iff `frontier_ids` is empty

### Required side effects

After observation updates:

- recompute frontiers,
- refresh dynamic object sets (`frontier`, `location`),
- update `exploration-complete` fluent,
- optionally record frame snapshots,
- request interrupt if thresholds are crossed and no robot is free.

## 5.4 Motion Skill + Environment Cadence Contract

A move skill must:

1. Parse action identity: `move robot from to`.
2. Build path at start.
3. On each advance:
   - update robot pose by interpolating along path,
   - apply due symbolic effects.
4. Expose motion state for environment-level sensing cadence:
   - controlled robot identity,
   - whether motion is active at absolute time `t`.
5. On interruption:
   - place robot at current intermediate pose,
   - create/update transient location `<robot>_loc` in registry,
   - rewrite pending deterministic effects from old destination to new transient target,
   - clear any stale claim on previous target,
   - end skill.

`UnknownSpaceEnvironment` owns sensing cadence and interrupt triggering:

- cap scheduler advance while any motion skill is active so time increments are at most `sensor_dt`,
- after each scheduler advance, sense for robots currently in active motion skills,
- use observed-cell deltas to set interrupt requests via existing thresholds.

Important: probabilistic pending effects should not be silently misapplied on interrupt. Preferred clean behavior is to drop or explicitly handle them; do not leave ambiguous half-applied probabilistic branches.

## 6. Frontier and Mapping Algorithms

## 6.1 Frontier Extraction

Frontier definition: free observed cell with at least one unknown neighbor (8-connected neighborhood).

Algorithm:

1. `free_mask = observed_grid in [FREE_VAL, OBSTACLE_THRESHOLD)`
2. `unknown_mask = observed_grid == UNOBSERVED_VAL`
3. Dilate `unknown_mask` by 3x3 ones kernel.
4. `frontier_mask = free_mask AND dilated_unknown_mask`
5. Connected-components label on `frontier_mask` with 8-connectivity.
6. For each component:
   - compute centroid,
   - snap centroid to nearest member cell,
   - create stable id: `frontier_<row>_<col>`.
7. Sort by id for deterministic ordering.

Reachability filter:

- Keep only frontiers reachable from at least one robot on planning grid where unknown cells are obstacles.

## 6.2 Scan Insertion / Map Fusion

For each robot observation:

1. Simulate laser ranges by ray casting on `true_grid`.
2. Truncate ranges to sensor max range.
3. Project rays into world coordinates using pose transform.
4. Mark interior polygon as free observations.
5. Mark ray endpoints as occupied observations.
6. Fuse projected measurement into `observed_grid` with occupancy blending:
   - free update via `unoccupied_prob`,
   - occupied update via `occupied_prob`.
7. Return count of newly observed cells (`previously UNOBSERVED`, now observed).

Optional correction step:

- For already observed cells, snap values to true map values to remove rasterization artifacts.

Integration note:

- This section defines the behavior contract expected from provided `mapping.insert_scan(...)`.
- Consume that implementation as-is and validate via integration tests.

## 6.3 Dynamic Symbolic Synchronization

This is the most important consistency rule.

At every planning cycle and after each action:

1. Compute `desired_navigable`:
   - all current frontier ids,
   - all hidden search sites whose map cells are now observed,
   - previously unlocked hidden sites (if unlock should persist).
2. Replace all existing `(navigable *)` fluents with new set.
3. Remove stale `(claimed *)` fluents for targets no longer valid.
4. Ensure `objects_by_type["frontier"]` and `objects_by_type["location"]` reflect current frontier lifecycle.

If this sync is skipped, planner actions become stale and can reference removed frontiers.

Reference pseudocode:

```text
function sync_dynamic_targets(env, hidden_sites):
    desired = set(env.objects_by_type["frontier"])

    # Persist previously unlocked hidden sites
    for fluent in env.fluents:
        if fluent matches (navigable site) and site in hidden_sites:
            desired.add(site)

    # Unlock hidden sites whose true map cell is now observed
    for site, (r, c) in hidden_sites:
        if env.is_cell_observed(r, c):
            desired.add(site)

    # Replace navigable fluents
    remove all (navigable *)
    for loc in desired:
        add (navigable loc)

    # Prune stale claims
    for fluent in all (claimed x):
        if x not in desired:
            remove fluent
```

## 6.4 Move-Time Estimation Model

Movement duration should be computable from current map knowledge and symbolic endpoints.

Required behavior:

- If destination is unreachable in mixed-known/unknown planning graph: return `inf` (filter action).
- If path exists through known space only:
  - `known_distance = path_length`
  - `unknown_distance = 0`
- If path requires unknown cells:
  - split path at first unknown cell into known and unknown segments.

Recommended dispatch-time estimate:

- `move_time = max(min_duration, known_distance / speed_cells_per_sec)`

Rationale:

- Encourages planner to act on known progress and rely on interrupts for newly revealed map.
- Avoids overcommitting to speculative unknown traversal durations.

Integration note:

- Use path-cost/path-geometry outputs from provided `pathing.py`.
- If needed, wrap starter pathing functions to produce the exact values this move-time contract requires.

## 7. Action Model Specification (PDDL-Style Semantics)

## 7.1 Move to Navigable Target

Parameters:

- `(?r - robot, ?from - location, ?to - location)`

Preconditions:

- `(at ?r ?from)`
- `(free ?r)`
- `(navigable ?to)`
- `(not (claimed ?to))`

Effects:

- At `t+0`:
  - `(not (free ?r))`
  - `(not (at ?r ?from))`
  - `(claimed ?to)`
- At `t+move_time(...)`:
  - `(free ?r)`
  - `(at ?r ?to)`
  - `(not (claimed ?to))`

## 7.2 Observe Site from Frontier

Parameters:

- `(?r - robot, ?frontier - frontier, ?site - location)`

Preconditions:

- `(at ?r ?frontier)`
- `(free ?r)`
- `(candidate-site ?site)`
- `(not (navigable ?site))`
- `(not (looked-for-site ?frontier ?site))`

Effects:

- At `t+0`: `(not (free ?r))`
- At `t+observe_time(...)`:
  - `(free ?r)`
  - `(looked-for-site ?frontier ?site)`
  - probabilistic branch:
    - success: add `(navigable ?site)`
    - failure: no additional fluents

## 7.3 Search at Site

Parameters:

- `(?r - robot, ?loc - location, ?obj - object)`

Preconditions:

- `(at ?r ?loc)`
- `(free ?r)`
- `(candidate-site ?loc)`
- `(not (revealed ?loc))`
- `(not (searched ?loc ?obj))`
- `(not (found ?obj))`
- `(not (lock-search ?loc))`

Effects:

- At `t+0`:
  - `(not (free ?r))`
  - `(lock-search ?loc)`
- At `t+search_time(...)`:
  - `(free ?r)`
  - `(searched ?loc ?obj)`
  - `(not (lock-search ?loc))`
  - probabilistic branch:
    - success: `(found ?obj)`, `(at ?obj ?loc)`
    - failure: no additional fluents

## 7.4 Ground-Truth Resolution Rule for Search

If the runtime has `true_object_locations`, probabilistic search outcomes should be resolved consistently with truth:

- If object truly at searched location: force success branch.
- Else: force failure branch.

This avoids sampled contradictions and is required for parity with deterministic object-search examples.

## 8. Planning and Execution Loop (Reference Pseudocode)

```text
initialize env with true_grid, observed_grid, state, operators
observe_all_robots(allow_interrupt=False)
sync_dynamic_navigable_targets()

while step < max_steps:
    if goal_satisfied(state): break

    sync_dynamic_navigable_targets()
    actions = env.get_actions()

    assert initial/ongoing heuristic finite for goal (guard)

    action_name = planner(state, goal, actions)
    if action_name == NONE: break

    action = lookup(actions, action_name)
    env.act(action)  # may interrupt moving skills due to new observations

    sync_dynamic_navigable_targets()
```

## 9. Coordination, Concurrency, and Correctness Invariants

## 9.1 Invariants

Must always hold:

1. `observed_grid` never reverts observed cell back to `UNOBSERVED`.
2. `objects_by_type["frontier"]` equals current extracted+reachable frontier ids.
3. Every frontier id in objects has coordinates in location registry.
4. Every `(navigable x)` corresponds to a currently valid target policy.
5. No stale `(claimed x)` for removed/invalid targets.
6. At most one active search lock per location due to `(lock-search ?loc)`.

## 9.2 Failure Modes and Required Guards

- Infinite/NaN move times: filter invalid actions before planning.
- Stale frontier references: enforce sync before and after every act.
- Thrashing interrupts: enforce `interrupt_min_new_cells` + `interrupt_min_dt`.
- Planner dead-end (`NONE`): terminate loop with explicit reason.
- Heuristic infinity at start: raise model-consistency error.

## 10. Recommended File Layout

Provided starter files (assumed existing):

- `packages/railroad/src/railroad/environment/navigation/constants.py`
- `packages/railroad/src/railroad/environment/navigation/mapping.py`
- `packages/railroad/src/railroad/environment/navigation/pathing.py`

Files to implement in this effort:

- `packages/railroad/src/railroad/environment/navigation/types.py`
- `packages/railroad/src/railroad/environment/navigation/laser.py`
- `packages/railroad/src/railroad/environment/navigation/frontiers.py`
- `packages/railroad/src/railroad/environment/navigation/environment.py`
- `packages/railroad/src/railroad/environment/navigation/skill.py`

Cross-cutting integration points:

- `packages/railroad/src/railroad/environment/environment.py` (interrupt hooks + act loop)
- `packages/railroad/src/railroad/environment/__init__.py` (exports)
- `packages/railroad/src/railroad/operators/*` (if adding helper constructors)
- `packages/railroad/pyproject.toml` (navigation extras)

## 11. Phased Delivery Plan

Phase 1: Runtime hooks and optional navigation package

- Add base interrupt hooks and action filters.
- Add navigation extras and package scaffold.

Phase 2: Starter-module integration + frontiers + unknown-space environment

- Integrate provided `constants.py`, `mapping.py`, and `pathing.py` through stable adapters if needed.
- Implement frontier extraction and reachable filtering.
- Implement frontier/object synchronization in environment.

Phase 3: Interruptible movement and dynamic replanning

- Implement environment-managed sensing cadence for active motion skills.
- Integrate interrupt policy into `act()` runtime.

Phase 4: Hidden-site domain + end-to-end task

- Implement observe/search operators and sync policy.
- Implement complete planning loop and goal checks.

Phase 5: Hardening

- Add guards for finite heuristic and stale actions.
- Add tests below.

## 12. Acceptance Test Matrix

Required tests (minimum):

1. Mapping growth test:
   - repeated scans monotonically increase observed fraction.
2. Frontier extraction test:
   - deterministic IDs and clusters for fixed synthetic grids.
3. Frontier lifecycle test:
   - frontiers empty when observed map becomes fully known.
4. Move interrupt test:
   - interrupted move yields `(at robot robot_loc)` and robot becomes free.
5. No interrupt test:
   - high thresholds allow full move completion to intended destination.
6. Hidden-site unlock test:
   - site becomes navigable only after corresponding cell observed.
7. Search lock test:
   - concurrent searches on same site are prevented.
8. End-to-end hidden-object smoke test:
   - scenario finds all target objects within bounded planning steps.
9. Heuristic guard test:
   - initial state is relaxed-reachable for task goal.

## 13. Essential vs Optional

Essential:

- Runtime interrupt support.
- Unknown-space map + frontier subsystem.
- Dynamic fluents/object synchronization.
- Interruptible move with sensing.
- Hidden-site observe/search model and locks.
- Validation tests above.

Optional:

- Dashboard map overlays.
- Video rendering and profiling outputs.
- Experimental duplicate utility packages.

## 14. Fixed Policy Decisions

1. Search semantics:
   - Must be deterministic via ground-truth resolution when truth data is available.
   - Do not sample probabilistic search outcomes in that mode.
2. Site unlock persistence:
   - `(navigable site)` must persist once a site is known navigable.
   - If a location can be navigated to, this predicate should be present.
3. Interrupt aggressiveness defaults:
   - Tune for fewer replans vs faster adaptation.
4. Unknown traversal in move-time estimation:
   - observed-only cost vs optimistic mixed known/unknown estimate.

Remaining tunables (3-4) must be documented in code comments and tests to avoid behavioral drift.

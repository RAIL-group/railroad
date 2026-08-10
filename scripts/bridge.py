"""Glue between railroad's symbolic MCTS planner and MolmoSpaces' pick-and-place
task/policy, used by plan_and_render.py.

Coordinate frames: railroad's own ProcTHOR loader (railroad.environment.procthor,
ThorInterface/ProcTHORScene) targets AI2-THOR's Unity-based simulator and builds
an occupancy grid in *that* simulator's coordinate frame and resolution. That is
not the same coordinate frame, unit scale, or asset addressing as MolmoSpaces'
MuJoCo scenes (which are separately exported from the same underlying ProcTHOR
houses) -- there is no valid grid-cell <-> MuJoCo-pose mapping between the two,
so railroad's occupancy-grid module is not usable here. Rather than fabricate a
mapping, this pipeline has railroad reason only over *order* (move, then pick,
then move, then place); MolmoSpaces owns every real coordinate, grasp pose, and
placement pose for the house it loads. Locations are named with real receptacle
names whenever they're known (see `plan_move_one_item`'s `pickup_location`), so
a printed plan reads as "move robot0 start diningtable_..." rather than opaque
placeholders; a placeholder ("pickup_site") is used only as a fallback when the
object's current real location isn't known at planning time. The one fact that
does have to cross the boundary -- that the object railroad's plan is built
around actually exists, at a real, finite pose, in the MolmoSpaces scene -- is
checked explicitly by `assert_pickup_pose_matches_sim` below.
"""

from __future__ import annotations

import numpy as np

ROBOT = "robot0"
START_LOC = "start"
PICKUP_SITE = "pickup_site"


# Assumed constant robot base cruising speed (m/s), used only to turn a real
# measured distance into a move duration. MolmoSpaces has no config constant
# for this (MobileFranka's base is position-controlled, not speed-limited by
# a single published number), so this is a placeholder in the same spirit as
# the `velocity=1.0` used by LocationRegistry.move_time_fn elsewhere in the
# repo (e.g. planned_single_robot_demo.py) -- not a value read from the robot.
ASSUMED_ROBOT_BASE_SPEED_M_S = 0.5


def _build_move_one_item_problem(obj: str, destination: str, env, pickup_location: str | None):
    """Shared setup for plan_move_one_item / plan_move_one_item_astar: a
    SymbolicEnvironment + goal for moving `obj` to the real `destination`
    receptacle, with move costs grounded in real measured distance.

    `destination` is always a real receptacle name (its position comes
    straight from the scene). `pickup_location` should be the object's real
    current receptacle when the caller knows it (e.g. from
    build_initial_fluents_from_scene's fluents) -- if omitted, a generic
    "pickup_site" token is used instead, positioned at the object's own
    location, since the object's real current receptacle isn't always known
    at planning time (e.g. before a MolmoSpaces task has settled the scene).
    A LocationRegistry built from these positions feeds
    LocationRegistry.move_time_fn, which turns straight-line distance into a
    duration via ASSUMED_ROBOT_BASE_SPEED_M_S.

    Returns (SymbolicEnvironment, goal Fluent, resolved pickup_location).
    """
    from railroad.core import Fluent as F, State
    from railroad.environment.symbolic import SymbolicEnvironment, LocationRegistry
    from railroad import operators

    om = env.object_managers[env.current_batch_index]
    robot_base_xy = np.asarray(env.current_robot.robot_view.base.pose[:2, 3], dtype=float)
    if pickup_location is None:
        pickup_location = PICKUP_SITE
        pickup_xy = np.asarray(om.get_object_by_name(obj).position[:2], dtype=float)
    else:
        pickup_xy = np.asarray(om.get_object_by_name(pickup_location).position[:2], dtype=float)
    destination_xy = np.asarray(om.get_object_by_name(destination).position[:2], dtype=float)

    registry = LocationRegistry(
        {
            START_LOC: robot_base_xy,
            pickup_location: pickup_xy,
            destination: destination_xy,
        }
    )
    # Plain (non-blocking) move: the "_blocking" variant adds a "just-moved"
    # precondition meant to stop a robot thrashing back and forth in
    # multi-robot coordination problems, which doesn't apply to this
    # single-robot, single-object, pure-ordering problem. Using it here
    # created a real dead end: "just-moved" is set immediately on arrival but
    # only clears via a delayed effect, which Environment.act() does not
    # advance to on its own (it stops as soon as a robot is free again) --
    # normally a subsequent pick/place's own longer duration flushes it as a
    # side effect of time passing, but if MCTS's first move ever goes to the
    # wrong site (plausible when pickup_site and destination_site happen to
    # be nearly equidistant from the start), `pick` then fails (object isn't
    # there) while `move` is blocked by `just-moved`, and the planner
    # correctly reports no action is available.
    move_op = operators.construct_move_operator(
        move_time=registry.move_time_fn(velocity=ASSUMED_ROBOT_BASE_SPEED_M_S)
    )
    pick_op = operators.construct_pick_operator_blocking(pick_time=5.0)
    place_op = operators.construct_place_operator_blocking(place_time=5.0)

    initial_fluents = {
        F(f"at {ROBOT} {START_LOC}"),
        F(f"free {ROBOT}"),
        F(f"at {obj} {pickup_location}"),
    }
    state = State(0.0, initial_fluents, [])
    senv = SymbolicEnvironment(
        state=state,
        objects_by_type={
            "robot": {ROBOT},
            "location": {START_LOC, pickup_location, destination},
            "object": {obj},
        },
        operators=[move_op, pick_op, place_op],
    )
    goal = F(f"at {obj} {destination}")
    return senv, goal, pickup_location


def plan_move_one_item(
    obj: str,
    destination: str,
    env,
    pickup_location: str | None = None,
    max_steps: int = 10,
) -> list[str]:
    """Railroad MCTS plan for moving a single named object to the destination.

    MCTSPlanner only returns the single next best action per call (see
    railroad.planner.MCTSPlanner), so this re-plans from scratch after each
    action until the goal is reached. See _build_move_one_item_problem for
    how the problem (locations, move costs) is constructed.
    """
    from railroad.core import get_action_by_name
    from railroad.planner import MCTSPlanner

    senv, goal, _pickup_location = _build_move_one_item_problem(
        obj, destination, env, pickup_location
    )

    plan: list[str] = []
    for _ in range(max_steps):
        if goal.evaluate(senv.fluents):
            return plan
        all_actions = senv.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(senv.state, goal, max_iterations=5000, max_depth=15)
        if action_name == "NONE":
            raise RuntimeError("Planner could not find a next action toward the goal.")
        action = get_action_by_name(all_actions, action_name)
        senv.act(action)
        plan.append(action_name)

    raise RuntimeError(f"Plan did not reach the goal within {max_steps} steps.")


def plan_move_one_item_astar(
    obj: str,
    destination: str,
    env,
    pickup_location: str | None = None,
) -> list[str]:
    """Railroad A* plan for moving a single named object to the destination.

    Unlike plan_move_one_item (MCTS), AStarPlanner.plan_sequence runs once and
    returns the full, cost-optimal plan directly -- a good way to sanity-check
    that the real measured move costs from _build_move_one_item_problem are
    wired up correctly (A* is deterministic and provably optimal, so a wrong
    or missing cost would show up as a wrong action ordering, not just
    flakiness).
    """
    from railroad.planner import AStarPlanner

    senv, goal, _pickup_location = _build_move_one_item_problem(
        obj, destination, env, pickup_location
    )
    if goal.evaluate(senv.fluents):
        return []

    astar = AStarPlanner(senv.get_actions())
    plan = astar.plan_sequence(senv.state, goal)
    if not plan:
        raise RuntimeError("A* found no plan to the goal.")
    return [action.name for action in plan]


# Translation layer: PickAndPlacePlannerPolicy runs the whole pick+place task as
# one continuous low-level trajectory (see BaseObjectManipulationPlannerPolicy)
# rather than exposing discrete move/pick/place calls, so railroad's four plan
# steps are attributed to ranges of the policy's own named phases
# (BaseObjectManipulationPlannerPolicy.get_all_phases) for step-by-step logging
# and failure attribution.
PLAN_STEP_PHASES: list[tuple[str, tuple[str, ...]]] = [
    ("move to object", ("gripper-open", "pregrasp")),
    ("pick", ("grasp", "gripper-close")),
    ("move to destination", ("lift", "preplace")),
    ("place", ("place", "retreat", "go_home", "unknown")),
]


def plan_step_for_phase(phase: str, current_step: int) -> int:
    """Map a policy phase name to a plan-step index. Never moves backward, since
    phase names (e.g. "gripper-open") recur across plan steps."""
    for idx, (_label, phases) in enumerate(PLAN_STEP_PHASES):
        if phase in phases and idx >= current_step:
            return idx
    return current_step


def assert_pickup_pose_matches_sim(
    observation: dict, env, pickup_obj_name: str, atol: float = 1e-3
) -> None:
    """Confirm the object railroad's plan assumes at `pickup_site` really exists,
    at a finite, non-degenerate pose, in the loaded MolmoSpaces scene -- and that
    the frozen "obj_start_pose" sensor the task itself reasons from agrees with
    the live simulator body pose (these come from two different code paths and
    would only diverge if the scene/task were set up inconsistently).
    """
    from molmo_spaces.utils.pose import pose_mat_to_7d

    expected = np.asarray(observation["obj_start_pose"], dtype=np.float64)
    om = env.object_managers[env.current_batch_index]
    actual = pose_mat_to_7d(om.get_object_by_name(pickup_obj_name).pose)

    if not np.all(np.isfinite(actual)):
        raise AssertionError(
            f"Pickup object {pickup_obj_name!r} has a non-finite pose in the "
            f"simulator: {actual}"
        )
    if not np.allclose(expected, actual, atol=atol):
        raise AssertionError(
            f"Pickup object {pickup_obj_name!r} pose mismatch: task's assumed "
            f"start pose {expected} vs. live simulator pose {actual}"
        )


def category_from_object_name(name: str) -> str:
    """Best-effort human-readable category from a MolmoSpaces object/body name,
    e.g. 'cup_4697b73732349085e30935e0f35566dc_1_0_2' -> 'Cup'."""
    token = name.split("_")[0]
    return token[:1].upper() + token[1:] if token else name


def build_initial_fluents_from_scene(
    om, robot: str = ROBOT, start_loc: str = START_LOC
) -> tuple[set, list[str]]:
    """Construct real railroad Fluents describing the current MolmoSpaces scene:
    the robot free at `start_loc`, plus "at <object> <receptacle>" for every
    pickupable object that is physically resting on a named receptacle/surface
    (per ObjectManager.objects_on_receptacle's contact-based check).

    Returns (fluents, unplaced) where `unplaced` lists pickupable objects for
    which no supporting receptacle was detected (e.g. on the floor, inside a
    closed container, or below the contact-detection threshold) -- these are
    real objects in the scene, just not ones this fluent set can localize.
    """
    from railroad.core import Fluent as F

    pickupable = om.get_pickup_candidates()
    receptacles = om.get_receptacles()

    fluents = {F(f"at {robot} {start_loc}"), F(f"free {robot}")}

    remaining = list(pickupable)
    unplaced_names: list[str] = []
    for recep in receptacles:
        if not remaining:
            break
        on_recep = om.objects_on_receptacle(remaining, recep.geom_ids)
        on_recep_names = {obj.name for obj in on_recep}
        for obj_name in on_recep_names:
            fluents.add(F(f"at {obj_name} {recep.name}"))
        remaining = [obj for obj in remaining if obj.name not in on_recep_names]

    unplaced_names = [obj.name for obj in remaining]
    return fluents, unplaced_names


def compute_pairwise_travel_distances(
    env,
    location_names: list[str],
    position_overrides: dict[str, np.ndarray] | None = None,
) -> dict[tuple[str, str], float]:
    """Shortest collision-free travel distance (meters) between every pair of
    named locations, routed around real obstacles in the loaded MolmoSpaces
    house.

    This reuses two *existing* pieces rather than implementing pathfinding
    from scratch: MolmoSpaces' own cached navigability map (`env.get_thormap()`,
    the same one `env.place_robot_near` uses to place the robot) supplies the
    real occupancy grid and the world-meters <-> grid-pixel transform; railroad's
    own Dijkstra/MCP shortest-path routine (`railroad.navigation.pathing.
    compute_cost_grid_from_position`, the lower-level function backing
    `get_cost_and_path`, which railroad.environment.procthor.ThorInterface
    uses for its own known-cost table) computes the paths. The two were never
    wired together before (see the module docstring's note on why railroad's
    own occupancy-grid *loader* isn't used here -- this only reuses its
    *pathing* function, which just operates on a plain numpy array, once fed
    a grid in its expected convention):

      - MolmoSpaces' occupancy array is truthy at *free* cells (see
        THORMap.get_free_points: `np.argwhere(self.occupancy)`).
      - railroad's pathing.build_traversal_costs/get_cost_and_path expect the
        opposite convention: FREE_VAL=0.0 at free cells, COLLISION_VAL=1.0 at
        obstacles (railroad.navigation.constants). The grid built below inverts
        MolmoSpaces' array into that convention.
      - compute_cost_grid_from_position is called once per location (not once
        per pair) with every remaining location passed as an `ends` target, so
        skimage.graph.MCP_Geometric solves a single-source, multi-target
        Dijkstra per location (N-1 solves total) instead of one full solve per
        pair (N*(N-1)/2) -- the naive pairwise version timed out past 5
        minutes on a 36-receptacle house.
      - Its returned cost is in grid-cell units (Dijkstra path length,
        diagonal steps included), not meters; it is converted to meters here
        via the map's own px_per_m (meters per cell = 1 / px_per_m).

    A receptacle's own position is itself inside its physical footprint
    (e.g. a table's center sits on the "occupied" cells of the table itself),
    so it must be snapped to the nearest actually-free floor cell before
    pathing -- otherwise it is an isolated obstacle-locked island with no
    route out. railroad.environment.procthor.ThorInterface solves this exact
    problem for its own (AI2-THOR) occupancy grid via
    `railroad.environment.procthor.utils.get_nearest_free_point`, but that
    helper does a plain per-point linear scan over every free point, which
    doesn't scale to MolmoSpaces' native-resolution map (millions of free
    cells vs. AI2-THOR's thousands of reachable positions). This uses
    `scipy.ndimage.distance_transform_edt` to do the same nearest-free-cell
    snap for every queried point in one vectorized pass.

    Args:
        env: MolmoSpaces environment with a loaded scene (task.env).
        location_names: Object/receptacle names (as returned by
            ObjectManager.get_receptacles()) to compute distances between.
        position_overrides: Explicit (x, y[, z]) position for any name in
            `location_names` that isn't a real scene object -- e.g. a robot's
            current base position under a synthetic name like "start" --
            bypassing the `om.get_object_by_name` lookup for that name.

    Returns:
        Dict mapping each unordered pair (name_a, name_b), name_a < name_b,
        to the shortest travel distance in meters, or float('inf') if no
        collision-free path exists between them.
    """
    import scipy.ndimage

    from railroad.navigation import pathing
    from railroad.navigation.constants import COLLISION_VAL, FREE_VAL

    om = env.object_managers[env.current_batch_index]
    thormap = env.get_thormap()

    # MolmoSpaces: truthy = free. railroad: FREE_VAL (0.0) = free, COLLISION_VAL
    # (1.0) = obstacle -- the two conventions are inverted.
    free_mask = thormap.occupancy.astype(bool)
    railroad_grid = np.where(free_mask, FREE_VAL, COLLISION_VAL)
    meters_per_cell = 1.0 / thormap.px_per_m

    # For every obstacle cell, the (row, col) of its nearest free cell (free
    # cells map to themselves at distance 0).
    nearest_free_row, nearest_free_col = scipy.ndimage.distance_transform_edt(
        ~free_mask, return_distances=False, return_indices=True
    )

    grid_coords: dict[str, tuple[int, int]] = {}
    for name in location_names:
        if position_overrides is not None and name in position_overrides:
            position = position_overrides[name]
        else:
            position = om.get_object_by_name(name).position
        px = thormap.pos_m_to_px(np.asarray(position, dtype=float))
        row = int(np.clip(px[0], 0, free_mask.shape[0] - 1))
        col = int(np.clip(px[1], 0, free_mask.shape[1] - 1))
        grid_coords[name] = (int(nearest_free_row[row, col]), int(nearest_free_col[row, col]))

    # One single-source Dijkstra run per location, covering every remaining
    # location as an `ends` target in that same run (skimage.graph.MCP_Geometric
    # early-terminates once all ends are settled), rather than get_cost_and_path's
    # one-pair-at-a-time interface -- for N locations this is N-1 Dijkstra runs
    # instead of N*(N-1)/2, which matters once a house has 30+ receptacles on a
    # multi-thousand-pixel native-resolution grid.
    distances: dict[tuple[str, str], float] = {}
    sorted_names = sorted(location_names)
    for i, name_a in enumerate(sorted_names):
        remaining = sorted_names[i + 1 :]
        if not remaining:
            break
        cost_grid = pathing.compute_cost_grid_from_position(
            railroad_grid,
            start=grid_coords[name_a],
            use_soft_cost=False,
            unknown_as_obstacle=False,
            ends=[grid_coords[name_b] for name_b in remaining],
            only_return_cost_grid=True,
        )
        for name_b in remaining:
            cost_cells = cost_grid[grid_coords[name_b]]
            distance_m = cost_cells * meters_per_cell if np.isfinite(cost_cells) else float("inf")
            distances[(name_a, name_b)] = distance_m

    return distances

#!/usr/bin/env python3
"""
Load a random ProcTHOR house in MolmoSpaces and print:
  - every object in the scene that can be picked up (has a free joint)
  - every named location the robot can navigate to and interact with
    (receptacle furniture/surfaces -- tables, counters, shelves, etc.)
  - the railroad initial-fluent set built from that real scene (bridge.py's
    build_initial_fluents_from_scene): robot free at "start", plus
    "at <object> <receptacle>" for every pickupable object physically
    resting on a detected receptacle

Scene loading bypasses MolmoSpaces' task sampler's own object-selection/
physics-feasibility check (see comment at the call site) since high-level
symbolic planning doesn't need it; no MolmoSpaces task or policy is created
or executed here, only scene introspection plus a railroad MCTS plan.

Run: python scripts/test_pipeline.py
"""

import heapq
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

warnings.filterwarnings("ignore")

from bridge import (
    ASSUMED_ROBOT_BASE_SPEED_M_S,
    ROBOT,
    START_LOC,
    build_initial_fluents_from_scene,
    compute_pairwise_travel_distances,
)
from planner_interface import call_planner, CoordinationAStarPlanner # pyrefly: ignore [missing-import]

HOUSE_INDEX_POOL = 50


def _fluent_str(f) -> str:
    args_str = " ".join(f.args)
    s = f"{f.name} {args_str}" if args_str else f.name
    return f"not {s}" if f.negated else s


@dataclass(order=True)
class _Event:
    time: float
    robot_id: str
    event_type: str = field(compare=False)
    data: Any = field(compare=False)


def _check_preconditions(action, state) -> Optional[str]:
    for pre in action.preconditions:
        if pre.startswith("not "):
            if pre[4:] in state:
                return pre
        else:
            if pre not in state:
                return pre
    return None


def run_two_robot_simulation(world_state, objects_by_type, my_operators, tasks) -> None:
    """Decentralized event-driven simulation for two independently-tasked robots.

    Same structure as pick_and_place_astar.py's main(): each robot re-plans with
    CoordinationAStarPlanner (Python A* with an injectable extra-cost hook,
    used here with no extra-cost function, i.e. plain A*) from a snapshot of
    the world (its own state plus every robot's committed future effects)
    whenever a task arrives or another robot's action completes.
    """
    world_state = set(world_state)
    event_queue: list[_Event] = []

    robots = list(objects_by_type["robot"])
    robot_plans: dict = {r: [] for r in robots}
    robot_blocked: dict = {r: False for r in robots}
    robot_pending_goal: dict = {r: None for r in robots}

    for arrival_t, robot, goal in tasks:
        heapq.heappush(
            event_queue,
            _Event(time=arrival_t, robot_id=robot, event_type="TASK_ARRIVAL", data=goal),
        )

    def broadcast_state():
        state = set(world_state)
        for other_plan in robot_plans.values():
            for action in other_plan:
                for d in action.del_effects:
                    state.discard(d)
                for a in action.add_effects:
                    state.add(a)
        return state

    def plan_and_start(t, robot, goal):
        plan = call_planner(
            broadcast_state(), goal, objects_by_type, my_operators, robot,
            planner_cls=CoordinationAStarPlanner,
        )
        if plan is None:
            print(f"  t={t:06.1f} | {robot} goal already satisfied")
        elif not plan:
            print(f"  t={t:06.1f} | {robot} FAILED — cannot plan for '{goal}'")
        else:
            robot_plans[robot] = plan
            try_start_next(t, robot)

    def try_start_next(t, robot):
        if not robot_plans[robot]:
            if robot_pending_goal[robot] is not None:
                goal = robot_pending_goal[robot]
                robot_pending_goal[robot] = None
                print(f"  t={t:06.1f} | {robot} idle — planning deferred task '{goal}'")
                plan_and_start(t, robot, goal)
            else:
                print(f"  t={t:06.1f} | {robot} finished — idle")
            return

        action = robot_plans[robot][0]
        fail = _check_preconditions(action, world_state)

        if fail is None:
            robot_blocked[robot] = False
            for d in action.del_effects:
                world_state.discard(d)
            print(f"  t={t:06.1f} | {robot} START  {action.name}")
            heapq.heappush(
                event_queue,
                _Event(time=t + action.duration, robot_id=robot, event_type="ACTION_END", data=action),
            )
        else:
            robot_blocked[robot] = True
            print(f"  t={t:06.1f} | {robot} WAITING — '{fail}' not yet true, will retry")

    while event_queue:
        event = heapq.heappop(event_queue)
        t, robot = event.time, event.robot_id

        if event.event_type == "TASK_ARRIVAL":
            goal = event.data
            print(f"\nt={t:06.1f} | [TASK_ARRIVAL] {robot} → '{goal}'")
            if robot_plans[robot]:
                print(f"  t={t:06.1f} | {robot} has active plan — deferring until plan completes")
                robot_pending_goal[robot] = goal
            else:
                plan_and_start(t, robot, goal)

        elif event.event_type == "ACTION_END":
            action = event.data
            for a in action.add_effects:
                world_state.add(a)
            print(f"  t={t:06.1f} | {robot} END    {action.name}")
            robot_plans[robot].pop(0)
            try_start_next(t, robot)
            for other in robots:
                if other != robot and robot_blocked[other]:
                    print(f"  t={t:06.1f} | {other} retrying after {robot} completed {action.name}")
                    try_start_next(t, other)

    print("\n--- Final world state ---")
    for f in sorted(world_state):
        print(f"  {f}")


def main() -> None:
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        FrankaPickAndPlaceDroidDataGenConfig,
    )

    seed = 472808211 # random.SystemRandom().randint(0, 2**31 - 1)
    print(f"[SEED] {seed}")

    config = FrankaPickAndPlaceDroidDataGenConfig()
    config.use_passive_viewer = False
    config.profile = False
    config.use_wandb = False
    config.seed = seed
    config.task_sampler_config.pickup_types = None
    config.task_sampler_config.enable_texture_randomization = False

    task_sampler = config.task_sampler_config.task_sampler_class(config)
    task_sampler.reset()

    house_idx = 36 # random.Random(seed).randrange(HOUSE_INDEX_POOL)
    print(f"[HOUSE] index {house_idx}")

    # Load the house via update_scene() directly rather than sample_task().
    # sample_task() also runs the sampler's task-specific object selection
    # (_select_pickup_object), which applies physics-feasibility filtering
    # (grasp/mass/IK-oriented) meant for *physical execution* and can raise
    # HouseInvalidForTask on houses with no candidate passing that filter.
    # None of that is needed for symbolic MCTS planning -- we only need the
    # scene loaded so we can introspect real objects/receptacles/positions
    # ourselves and hand them straight to the planner.
    task_sampler._house_inds = [house_idx]
    task_sampler._house_iterator_index = 0
    task_sampler.update_scene(task_sampler._current_house_scene_path())
    env = task_sampler.env
    om = env.object_managers[env.current_batch_index]

    pickupable = om.get_pickup_candidates()
    print(f"\n[PICKUPABLE OBJECTS] {len(pickupable)} object(s) with a free joint:")
    for obj in pickupable:
        print(f"  - {obj.name}")

    receptacles = om.get_receptacles()
    print(f"\n[NAMED LOCATIONS] {len(receptacles)} receptacle/surface(s) the robot can navigate to:")
    for recep in receptacles:
        print(f"  - {recep.name}")

    fluents, unplaced = build_initial_fluents_from_scene(om)
    print(f"\n[INITIAL FLUENTS] {len(fluents)} fluent(s) built from the real scene:")
    for fluent in sorted(fluents, key=str):
        print(f"  - {fluent}")
    if unplaced:
        print(
            f"\n  ({len(unplaced)} pickupable object(s) had no detected supporting "
            f"receptacle, so are not localized above -- e.g. on the floor, inside "
            f"a closed container, or below the contact-detection threshold):"
        )
        for name in unplaced:
            print(f"    - {name}")

    # Real collision-aware distance between every receptacle *and* the robots'
    # shared starting point (added via position_overrides since "start" isn't
    # a real scene object) -- this is the one set of distances used both for
    # the printout below and as the actual move cost for the two-robot plan,
    # rather than a separate straight-line estimate.
    # pos_m_to_px (used inside compute_pairwise_travel_distances) needs a full
    # 3D (x, y, z) position, unlike the 2D xy bridge.py's straight-line-distance
    # helpers use.
    robot_base_xyz = np.asarray(env.current_robot.robot_view.base.pose[:3, 3], dtype=float)
    receptacle_names = [recep.name for recep in receptacles]
    distances = compute_pairwise_travel_distances(
        env, receptacle_names + [START_LOC], position_overrides={START_LOC: robot_base_xyz}
    )
    print(
        f"\n[PAIRWISE TRAVEL DISTANCES] shortest collision-free distance (m) "
        f"between all {len(receptacle_names) + 1} named locations (incl. robot start):"
    )
    for (name_a, name_b), distance_m in sorted(distances.items(), key=lambda kv: -kv[1]):
        distance_str = f"{distance_m:.2f}m" if distance_m != float("inf") else "unreachable"
        print(f"  - {name_a}  <->  {name_b}: {distance_str}")

    # ------------------------------------------------------------------
    # Two-robot concurrent task planning -- same decentralized event-driven
    # setup as pick_and_place_astar.py (call_planner + CoordinationAStarPlanner
    # per robot, world state advanced by an event queue), grounded in only the
    # objects/locations the two tasks actually reference (plus each task
    # object's real current receptacle, so the robot has somewhere to pick it
    # up from) rather than the whole scene, to keep the search space small --
    # with move cost equal to the real collision-aware distance computed above
    # instead of a straight-line one:
    #   robot1 @ t=0  ClearBed  -- basketball/tennis racket off the bed, pillow on it
    #   robot2 @ t=5  SetTable  -- bowl and mug on the dining table
    # ------------------------------------------------------------------
    from railroad import operators

    ROBOT1, ROBOT2 = "robot1", "robot2"

    BASKETBALL = "basketball_13f63d02d4de4b9acc00b38afd72266b_1_0_2"
    TENNISRACKET = "tennisracket_42c6d99c18843075f3d477d62ad8c5ab_1_0_2"
    PILLOW = "pillow_e6c6ee75a221e60aa884f04d694f5c5d_1_0_2"
    BED = "bed_d62dc7f87a3c3074e5ad6b8e6cd4ed2f_1_0_2"
    BOWL = "bowl_ab2df3cd7ba6e36f1f725f462a622119_1_0_2"
    MUG = "mug_92c42ca5366a9cee73d1919f977d8b28_1_0_2"
    DININGTABLE = "diningtable_9ad22c85731f44adaa87de6c1f5efb23_1_0_2"

    pickupable_names = {obj.name for obj in pickupable}
    receptacle_names_set = set(receptacle_names)
    missing_objects = {BASKETBALL, TENNISRACKET, PILLOW, BOWL, MUG} - pickupable_names
    missing_receptacles = {BED, DININGTABLE} - receptacle_names_set
    if missing_objects or missing_receptacles:
        print(
            f"\n[TASK] scene is missing required object(s) {missing_objects or set()} "
            f"or receptacle(s) {missing_receptacles or set()}; skipping two-robot demo."
        )
        return

    def move_time(_robot: str, loc_from: str, loc_to: str) -> float:
        """Real collision-aware travel time from the pairwise distances
        computed above, not a straight-line estimate."""
        if loc_from == loc_to:
            return 0.0
        key = (loc_from, loc_to) if loc_from < loc_to else (loc_to, loc_from)
        return distances.get(key, float("inf")) / ASSUMED_ROBOT_BASE_SPEED_M_S

    move_op = operators.construct_move_operator(move_time)
    pick_op = operators.construct_pick_operator_blocking(pick_time=1.0)
    place_op = operators.construct_place_operator_blocking(place_time=1.0)
    my_operators = [move_op, pick_op, place_op]

    def _current_location(obj_name: str) -> str | None:
        for f in fluents:
            if f.name == "at" and len(f.args) == 2 and f.args[0] == obj_name:
                return f.args[1]
        return None

    task_objects = {BASKETBALL, TENNISRACKET, PILLOW, BOWL, MUG}
    # Goal receptacles, plus each task object's real current receptacle (so
    # the robot has somewhere to pick it up from), plus the shared start.
    task_locations = {BED, DININGTABLE, START_LOC}
    task_locations |= {
        loc for loc in (_current_location(obj) for obj in task_objects) if loc is not None
    }

    # Small world: only the objects/locations the two tasks reference, not
    # every pickupable object and receptacle in the scene.
    objects_by_type = {
        "robot": {ROBOT1, ROBOT2},
        "location": task_locations,
        "object": task_objects,
    }

    initial_world_state = {
        _fluent_str(f)
        for f in fluents
        if ROBOT not in f.args and set(f.args) <= (task_objects | task_locations)
    }
    initial_world_state |= {
        f"free {ROBOT1}", f"at {ROBOT1} {START_LOC}",
        f"free {ROBOT2}", f"at {ROBOT2} {START_LOC}",
    }

    clear_bed_goal = (
        f"not at {BASKETBALL} {BED} & not at {TENNISRACKET} {BED} & at {PILLOW} {BED} & not hand-full {ROBOT1}"
    )
    set_table_goal = f"at {BASKETBALL} {DININGTABLE} & at {TENNISRACKET} {DININGTABLE} & not hand-full {ROBOT2}"
    tasks = [
        (0.0, ROBOT1, clear_bed_goal),
        (5.0, ROBOT2, set_table_goal),
    ]
    print(f"\n[INITIAL WORLD STATE] {len(initial_world_state)} fluent(s):")
    for fluent_str in sorted(initial_world_state):
        print(f"  - {fluent_str}")
    print(f"\n[TASK] {ROBOT1} @ t=0.0 ClearBed: {clear_bed_goal}")
    print(f"[TASK] {ROBOT2} @ t=5.0 SetTable: {set_table_goal}")

    run_two_robot_simulation(initial_world_state, objects_by_type, my_operators, tasks)


if __name__ == "__main__":
    main()

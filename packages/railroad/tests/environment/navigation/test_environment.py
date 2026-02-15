"""Tests 4-9: Integration tests for UnknownSpaceEnvironment."""

from __future__ import annotations

import math

import numpy as np

from railroad._bindings import Fluent, State
from railroad.environment.navigation.constants import COLLISION_VAL, FREE_VAL
from railroad.environment.navigation.environment import UnknownSpaceEnvironment
from railroad.environment.navigation.types import NavigationConfig, Pose
from railroad.environment.symbolic import LocationRegistry
from railroad.operators import (
    construct_move_navigable_operator,
    construct_observe_site_operator,
    construct_search_at_site_operator,
)

F = Fluent


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _make_branching_grid() -> np.ndarray:
    """Create a 30x30 branching corridor grid for tests.

    Layout (0=free, 1=wall, conceptual):
        - Central corridor from (5,5) to (5,25)
        - East branch from (5,25) to (15,25)  -> stash_east at (15,25)
        - North branch from (5,15) to (25,15) -> stash_north at (25,15)
        - West branch from (5,5) to (15,5)    -> stash_west at (15,5)
    """
    grid = COLLISION_VAL * np.ones((30, 30))

    # Central horizontal corridor (rows 4-6, cols 4-26)
    grid[4:7, 4:27] = FREE_VAL

    # East branch (rows 4-16, cols 24-26)
    grid[4:17, 24:27] = FREE_VAL

    # North branch (rows 4-26, cols 14-16)
    grid[4:27, 14:17] = FREE_VAL

    # West branch (rows 4-16, cols 4-6)
    grid[4:17, 4:7] = FREE_VAL

    return grid


def _make_environment(
    config: NavigationConfig | None = None,
    two_robots: bool = True,
) -> UnknownSpaceEnvironment:
    """Create a test UnknownSpaceEnvironment with the branching corridor grid."""
    true_grid = _make_branching_grid()

    if config is None:
        config = NavigationConfig(
            sensor_range=9.0,
            sensor_fov_rad=2 * math.pi,
            sensor_num_rays=181,
            sensor_dt=0.08,
            speed_cells_per_sec=2.0,
            interrupt_min_new_cells=20,
            interrupt_min_dt=1.0,
        )

    # Robot starting poses
    robot_initial_poses: dict[str, Pose] = {"robot1": Pose(5.0, 5.0, 0.0)}
    if two_robots:
        robot_initial_poses["robot2"] = Pose(5.0, 7.0, 0.0)

    # Location registry
    locations: dict[str, np.ndarray] = {
        "start": np.array([5, 5]),
    }
    if two_robots:
        locations["start2"] = np.array([5, 7])

    registry = LocationRegistry(locations)

    # Hidden sites
    hidden_sites = {
        "stash_east": (15, 25),
        "stash_north": (25, 15),
        "stash_west": (15, 5),
    }

    # Object types
    robots = {"robot1"}
    if two_robots:
        robots.add("robot2")

    objects_by_type: dict[str, set[str]] = {
        "robot": robots,
        "location": {"start"} | ({"start2"} if two_robots else set())
                    | set(hidden_sites.keys()),
        "container": set(hidden_sites.keys()),
        "frontier": set(),
        "object": {"Mug", "Knife"},
    }

    # True object locations
    true_object_locations = {
        "stash_east": {"Mug"},
        "stash_north": {"Knife"},
    }

    # Initial fluents
    fluents: set[Fluent] = {
        F("at robot1 start"),
        F("free robot1"),
        F("revealed start"),
    }
    if two_robots:
        fluents |= {
            F("at robot2 start2"),
            F("free robot2"),
            F("revealed start2"),
        }

    state = State(0.0, fluents, [])

    # Operators with environment-based move time
    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        # Will be overridden by NavigationMoveSkill path-based duration
        return 5.0

    operators = [
        construct_move_navigable_operator(move_time_fn),
        construct_observe_site_operator(0.8, 1.0, container_type="container"),
        construct_search_at_site_operator(0.9, 2.0, container_type="container"),
    ]

    return UnknownSpaceEnvironment(
        state=state,
        objects_by_type=objects_by_type,
        operators=operators,
        true_grid=true_grid,
        robot_initial_poses=robot_initial_poses,
        location_registry=registry,
        hidden_sites=hidden_sites,
        true_object_locations=true_object_locations,
        config=config,
    )


# ---------------------------------------------------------------------------
# Test 4: Move interrupt yields (at robot robot_loc) and robot becomes free
# ---------------------------------------------------------------------------


def test_move_interrupt_creates_robot_loc():
    """Interrupted move should yield (at robot robot_loc) and free robot."""
    config = NavigationConfig(
        sensor_range=9.0,
        sensor_fov_rad=2 * math.pi,
        sensor_num_rays=91,
        sensor_dt=0.05,
        speed_cells_per_sec=2.0,
        # Very aggressive interrupt: trigger after just 1 new cell
        interrupt_min_new_cells=1,
        interrupt_min_dt=0.0,
    )
    env = _make_environment(config=config, two_robots=False)

    state = env.state

    # Get actions and find a move action whose preconditions are met
    actions = env.get_actions()
    move_actions = [
        a for a in actions
        if a.name.startswith("move") and state.satisfies_precondition(a)
    ]
    assert len(move_actions) > 0, "Should have at least one valid move action"

    action = move_actions[0]
    result = env.act(action)

    # Check: robot should be free
    assert F("free robot1") in result.fluents, "Robot should be free after interrupt"

    # Check: robot should be at some location
    at_fluents = [f for f in result.fluents if f.name == "at" and f.args[0] == "robot1"]
    assert len(at_fluents) == 1, f"Robot should be at exactly one location, got {at_fluents}"
    assert F("just-moved", "robot1") not in result.fluents


# ---------------------------------------------------------------------------
# Test 5: High thresholds allow full move completion
# ---------------------------------------------------------------------------


def test_move_completes_with_high_thresholds():
    """High interrupt thresholds allow move to complete to destination."""
    config = NavigationConfig(
        sensor_range=9.0,
        sensor_fov_rad=2 * math.pi,
        sensor_num_rays=91,
        sensor_dt=0.08,
        speed_cells_per_sec=2.0,
        # Very high thresholds — should never interrupt
        interrupt_min_new_cells=100000,
        interrupt_min_dt=100000.0,
    )
    env = _make_environment(config=config, two_robots=False)

    state = env.state
    actions = env.get_actions()
    move_actions = [
        a for a in actions
        if a.name.startswith("move") and state.satisfies_precondition(a)
    ]
    assert len(move_actions) > 0

    action = move_actions[0]
    destination = action.name.split()[3]

    result = env.act(action)

    assert F("free robot1") in result.fluents
    assert F("at", "robot1", destination) in result.fluents, (
        f"Robot should be at destination {destination}"
    )
    assert F("just-moved", "robot1") not in result.fluents


# ---------------------------------------------------------------------------
# Regression: Unreachable moves must be filtered (no obstacle-crossing fallback)
# ---------------------------------------------------------------------------


def test_unreachable_move_is_filtered_out():
    """Moves to unreachable destinations should not be instantiated."""
    # Two disconnected free regions separated by walls.
    true_grid = COLLISION_VAL * np.ones((14, 14))
    true_grid[1:13, 1:5] = FREE_VAL
    true_grid[1:13, 9:13] = FREE_VAL

    location_registry = LocationRegistry(
        {
            "start": np.array([6, 2], dtype=float),
            "goal": np.array([6, 10], dtype=float),
        }
    )

    env_ref: list[UnknownSpaceEnvironment | None] = [None]

    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        assert env_ref[0] is not None
        return env_ref[0].estimate_move_time(robot, loc_from, loc_to)

    operators = [construct_move_navigable_operator(move_time_fn)]

    env = UnknownSpaceEnvironment(
        state=State(0.0, {F("at robot1 start"), F("free robot1")}, []),
        objects_by_type={
            "robot": {"robot1"},
            "location": {"start", "goal"},
            "frontier": set(),
            "object": set(),
        },
        operators=operators,
        true_grid=true_grid,
        robot_initial_poses={"robot1": Pose(6.0, 2.0, 0.0)},
        location_registry=location_registry,
        hidden_sites={},
        config=NavigationConfig(
            sensor_range=2.0,
            correct_with_known_map=False,
            interrupt_min_new_cells=100000,
            interrupt_min_dt=100000.0,
        ),
    )
    env_ref[0] = env

    actions = env.get_actions()
    move_names = {a.name for a in actions}
    assert "move robot1 start goal" not in move_names


# ---------------------------------------------------------------------------
# Regression: just-moved move precondition gate
# ---------------------------------------------------------------------------


def test_just_moved_blocks_followup_move_actions():
    """(just-moved robot) should make move preconditions fail for that robot."""
    env = _make_environment(two_robots=False)

    env.fluents.add(F("just-moved", "robot1"))
    state = env.state
    actions = env.get_actions()
    move_actions_with_satisfied_preconditions = [
        a for a in actions
        if a.name.startswith("move robot1 ") and state.satisfies_precondition(a)
    ]
    assert not move_actions_with_satisfied_preconditions


# ---------------------------------------------------------------------------
# Regression: clear just-moved at transient robot_loc anchors
# ---------------------------------------------------------------------------


def test_sync_clears_just_moved_when_robot_at_robot_loc():
    """When robot is at robot_loc, sync should clear just-moved for that robot."""
    env = _make_environment(two_robots=False)

    env.fluents.discard(F("at robot1 start"))
    env.fluents.add(F("at robot1 robot1_loc"))
    env.fluents.add(F("just-moved", "robot1"))
    env.objects_by_type.setdefault("location", set()).add("robot1_loc")

    env.sync_dynamic_targets()

    assert F("just-moved", "robot1") not in env.fluents


# ---------------------------------------------------------------------------
# Regression: move-time cache must respect location coordinate updates
# ---------------------------------------------------------------------------


def test_move_time_cache_updates_when_location_moves():
    """Updating location coordinates should invalidate cached move-time grid."""
    true_grid = FREE_VAL * np.ones((24, 24))
    true_grid[0, :] = COLLISION_VAL
    true_grid[-1, :] = COLLISION_VAL
    true_grid[:, 0] = COLLISION_VAL
    true_grid[:, -1] = COLLISION_VAL

    location_registry = LocationRegistry(
        {
            "robot1_loc": np.array([2, 2], dtype=float),
            "goal": np.array([20, 20], dtype=float),
        }
    )

    env_ref: list[UnknownSpaceEnvironment | None] = [None]

    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        assert env_ref[0] is not None
        return env_ref[0].estimate_move_time(robot, loc_from, loc_to)

    env = UnknownSpaceEnvironment(
        state=State(
            0.0,
            {F("at robot1 robot1_loc"), F("free robot1")},
            [],
        ),
        objects_by_type={
            "robot": {"robot1"},
            "location": {"robot1_loc", "goal"},
            "frontier": set(),
            "object": set(),
        },
        operators=[construct_move_navigable_operator(move_time_fn)],
        true_grid=true_grid,
        robot_initial_poses={"robot1": Pose(2.0, 2.0, 0.0)},
        location_registry=location_registry,
        hidden_sites={},
        config=NavigationConfig(
            sensor_range=2.0,
            correct_with_known_map=False,
            interrupt_min_new_cells=100000,
            interrupt_min_dt=100000.0,
        ),
    )
    env_ref[0] = env

    # Use a fully-known map to isolate cache behavior from observations.
    env._observed_grid[:] = true_grid

    t1 = env.estimate_move_time("robot1", "robot1_loc", "goal")
    location_registry.register("robot1_loc", np.array([10, 10], dtype=float))
    t2 = env.estimate_move_time("robot1", "robot1_loc", "goal")

    assert t1 != t2, "Move-time cache should refresh after location coordinate changes"


# ---------------------------------------------------------------------------
# Test 6: Observation should not reintroduce legacy navigable fluents
# ---------------------------------------------------------------------------


def test_hidden_site_observation_does_not_add_navigable():
    """Observed hidden cells should not add legacy (navigable *) fluents."""
    env = _make_environment(two_robots=False)

    assert all(
        not (f.name == "navigable" and not f.negated)
        for f in env.fluents
    )

    # Simulate observing a hidden site cell by moving closer.
    env._robot_poses["robot1"] = Pose(12.0, 5.0, math.pi / 2)
    env.observe_from_pose("robot1", env._robot_poses["robot1"], env.time + 0.1,
                          allow_interrupt=False)
    env.refresh_frontiers()
    env.sync_dynamic_targets()
    assert all(
        not (f.name == "navigable" and not f.negated)
        for f in env.fluents
    )


# ---------------------------------------------------------------------------
# Test 7: Search lock prevents concurrent searches
# ---------------------------------------------------------------------------


def test_search_lock_prevents_concurrent():
    """The (lock-search ?loc) fluent prevents two robots from searching the same site."""
    env = _make_environment(two_robots=True)

    # Manually place both robots at stash_east
    env._fluents.add(F("at robot1 stash_east"))
    env._fluents.add(F("at robot2 stash_east"))
    env._fluents.discard(F("at robot1 start"))
    env._fluents.discard(F("at robot2 start2"))

    state = env.state

    # Get search actions at stash_east whose preconditions are satisfied
    actions = env.get_actions()
    search_actions = [
        a for a in actions
        if a.name.startswith("search") and "stash_east" in a.name
        and state.satisfies_precondition(a)
    ]

    # There should be search actions for both robots
    robot1_searches = [a for a in search_actions if "robot1" in a.name]
    robot2_searches = [a for a in search_actions if "robot2" in a.name]
    assert len(robot1_searches) > 0, "Robot1 should have search actions"
    assert len(robot2_searches) > 0, "Robot2 should have search actions"

    # Simulate search start: set (lock-search stash_east) and mark robot1 busy
    env._fluents.add(F("lock-search stash_east"))
    env._fluents.discard(F("free robot1"))

    # Now check: robot2's search actions should fail preconditions
    state_after = env.state
    actions_after = env.get_actions()
    robot2_stash_searches = [
        a for a in actions_after
        if a.name.startswith("search") and "stash_east" in a.name
        and "robot2" in a.name
        and state_after.satisfies_precondition(a)
    ]
    assert len(robot2_stash_searches) == 0, (
        "Robot2 should not be able to search stash_east while lock-search is active"
    )


# ---------------------------------------------------------------------------
# Test 8: End-to-end smoke test
# ---------------------------------------------------------------------------


def test_end_to_end_smoke():
    """Two robots find Mug and Knife across hidden sites in bounded steps."""
    config = NavigationConfig(
        sensor_range=9.0,
        sensor_fov_rad=2 * math.pi,
        sensor_num_rays=91,
        sensor_dt=0.1,
        speed_cells_per_sec=3.0,
        interrupt_min_new_cells=100000,
        interrupt_min_dt=100000.0,
    )
    env = _make_environment(config=config, two_robots=True)

    # We'll manually drive the robots through the scenario rather than
    # using the planner (to test environment mechanics, not the planner).

    max_steps = 50
    found_mug = False
    found_knife = False

    for step in range(max_steps):
        if F("found Mug") in env.fluents:
            found_mug = True
        if F("found Knife") in env.fluents:
            found_knife = True
        if found_mug and found_knife:
            break

        # Manually advance: move robots towards hidden sites, unlock, search
        # Step 1: Move robot1 towards east (stash_east has Mug)
        if step == 0:
            # Teleport robot1 near stash_east and observe
            env._robot_poses["robot1"] = Pose(15.0, 25.0, 0.0)
            env.observe_from_pose("robot1", env._robot_poses["robot1"],
                                  env.time + 1.0, allow_interrupt=False)
            env.refresh_frontiers()
            env.sync_dynamic_targets()

            # Place robot at stash_east symbolically
            env._fluents.discard(F("at robot1 start"))
            env._fluents.add(F("at robot1 stash_east"))

        if step == 1:
            # Search stash_east with robot1 for Mug
            actions = env.get_actions()
            search_mug = [
                a for a in actions
                if a.name.startswith("search") and "robot1" in a.name
                and "stash_east" in a.name and "Mug" in a.name
            ]
            if search_mug:
                env.act(search_mug[0])

        if step == 2:
            # Teleport robot2 near stash_north and observe
            env._robot_poses["robot2"] = Pose(25.0, 15.0, 0.0)
            env.observe_from_pose("robot2", env._robot_poses["robot2"],
                                  env.time + 1.0, allow_interrupt=False)
            env.refresh_frontiers()
            env.sync_dynamic_targets()

            env._fluents.discard(F("at robot2 start2"))
            env._fluents.add(F("at robot2 stash_north"))

        if step == 3:
            # Search stash_north with robot2 for Knife
            actions = env.get_actions()
            search_knife = [
                a for a in actions
                if a.name.startswith("search") and "robot2" in a.name
                and "stash_north" in a.name and "Knife" in a.name
            ]
            if search_knife:
                env.act(search_knife[0])

    assert F("found Mug") in env.fluents, "Should have found Mug"
    assert F("found Knife") in env.fluents, "Should have found Knife"


# ---------------------------------------------------------------------------
# Test 9: Heuristic guard — initial state has finite relaxed-plan cost
# ---------------------------------------------------------------------------


def test_heuristic_finite_initial():
    """The initial state should be relaxed-reachable for the task goal.

    We check this by verifying the planner's heuristic returns finite cost.
    """
    from railroad.planner import MCTSPlanner

    env = _make_environment(two_robots=True)

    goal = F("found Mug") & F("found Knife")

    actions = env.get_actions()
    assert len(actions) > 0, "Should have available actions"

    planner = MCTSPlanner(actions)

    state = env.state
    h_value = planner.heuristic(state, goal)

    assert h_value < float("inf"), (
        f"Heuristic should be finite for initial state, got {h_value}"
    )

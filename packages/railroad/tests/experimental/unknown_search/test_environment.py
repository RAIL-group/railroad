"""Tests 4-9: Integration tests for UnknownSpaceEnvironment."""

from __future__ import annotations

import math

import numpy as np

from railroad._bindings import Fluent, State
from railroad.navigation.constants import COLLISION_VAL, FREE_VAL
from railroad.experimental.unknown_search.environment import UnknownSpaceEnvironment
from railroad.experimental.unknown_search.types import NavigationConfig, Pose
from railroad.environment.symbolic import LocationRegistry
from railroad.experimental.unknown_search.operators import (
    construct_move_navigable_operator,
    construct_search_at_site_operator,
)
from railroad.operators import construct_no_op_operator

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
    include_no_op: bool = False,
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
        construct_search_at_site_operator(0.9, 2.0, container_type="container"),
    ]
    if include_no_op:
        operators.append(construct_no_op_operator(2.0))

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


def _first_valid_move_action(env: UnknownSpaceEnvironment):
    state = env.state
    actions = env.get_actions()
    move_actions = [
        a for a in actions
        if a.name.startswith("move") and state.satisfies_precondition(a)
    ]
    assert move_actions, "Should have at least one valid move action"
    return move_actions[0]


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

    action = _first_valid_move_action(env)
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
    """Move ends at a currently valid location after frontier refresh."""
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

    action = _first_valid_move_action(env)
    destination = action.name.split()[3]

    result = env.act(action)

    assert F("free robot1") in result.fluents
    if destination in env.objects_by_type.get("location", set()):
        assert F("at", "robot1", destination) in result.fluents
    else:
        assert F("at", "robot1", "robot1_loc") in result.fluents
        assert F("at", "robot1", destination) not in result.fluents
    assert F("just-moved", "robot1") not in result.fluents


def test_environment_manages_sensing_cadence_during_motion():
    """Movement should trigger repeated sensing under env-managed sensor_dt."""
    config = NavigationConfig(
        sensor_range=9.0,
        sensor_fov_rad=2 * math.pi,
        sensor_num_rays=91,
        sensor_dt=0.05,
        speed_cells_per_sec=2.0,
        interrupt_min_new_cells=100000,
        interrupt_min_dt=100000.0,
    )
    env = _make_environment(config=config, two_robots=False)
    action = _first_valid_move_action(env)

    observed_times: list[float] = []
    original_observe_from_pose = env.observe_from_pose

    def wrapped_observe_from_pose(
        robot: str,
        pose: Pose,
        time: float,
        allow_interrupt: bool = True,
    ) -> int:
        observed_times.append(time)
        return original_observe_from_pose(
            robot, pose, time, allow_interrupt=allow_interrupt
        )

    env.observe_from_pose = wrapped_observe_from_pose  # type: ignore[assignment]
    env.act(action)

    assert len(observed_times) > 1, "Expected repeated sensing while moving"
    diffs = np.diff(np.array(observed_times, dtype=float))
    assert np.all(diffs <= config.sensor_dt + 1e-6), (
        "Observed sensing intervals should be bounded by sensor_dt"
    )


def test_no_motion_action_not_capped_by_sensor_dt():
    """Non-motion actions should not be forced into sensor_dt micro-steps."""
    config = NavigationConfig(
        sensor_range=9.0,
        sensor_fov_rad=2 * math.pi,
        sensor_num_rays=91,
        sensor_dt=0.05,
        speed_cells_per_sec=2.0,
        interrupt_min_new_cells=100000,
        interrupt_min_dt=100000.0,
    )
    env = _make_environment(config=config, two_robots=False, include_no_op=True)

    state = env.state
    no_op_actions = [
        a for a in env.get_actions()
        if a.name.startswith("no_op") and state.satisfies_precondition(a)
    ]
    assert no_op_actions, "Expected a valid no-op action"

    result = env.act(no_op_actions[0])
    assert math.isclose(result.time, 2.0, rel_tol=0.0, abs_tol=1e-9)


def test_non_interruptible_move_completes_under_aggressive_interrupt_thresholds():
    """Non-interruptible move still remaps if destination becomes stale."""
    config = NavigationConfig(
        sensor_range=9.0,
        sensor_fov_rad=2 * math.pi,
        sensor_num_rays=91,
        sensor_dt=0.05,
        speed_cells_per_sec=2.0,
        interrupt_min_new_cells=1,
        interrupt_min_dt=0.0,
        move_execution_interruptible=False,
    )
    env = _make_environment(config=config, two_robots=False)
    action = _first_valid_move_action(env)
    destination = action.name.split()[3]

    result = env.act(action)

    if destination in env.objects_by_type.get("location", set()):
        assert F("at", "robot1", destination) in result.fluents
    else:
        assert F("at", "robot1", destination) not in result.fluents
        assert F("at", "robot1", "robot1_loc") in result.fluents


def test_stale_destination_is_handled_only_on_interrupt_request(monkeypatch):
    """Stale move destination is rewritten when an interrupt request is processed."""
    import railroad.experimental.unknown_search.environment as nav_env_module

    config = NavigationConfig(
        sensor_range=9.0,
        sensor_fov_rad=2 * math.pi,
        sensor_num_rays=91,
        sensor_dt=0.05,
        speed_cells_per_sec=2.0,
        interrupt_min_new_cells=100000,
        interrupt_min_dt=100000.0,
        move_execution_interruptible=True,
    )
    env = _make_environment(config=config, two_robots=False)

    state = env.state
    frontier_moves = [
        a for a in env.get_actions()
        if a.name.startswith("move")
        and state.satisfies_precondition(a)
        and a.name.split()[3].startswith("frontier_")
    ]
    assert frontier_moves, "Expected at least one move to a frontier destination"

    action = frontier_moves[0]
    stale_destination = action.name.split()[3]
    skill = env.create_skill(action, env.time)
    env._active_skills.append(skill)

    for s in env._active_skills:
        s.advance(env.time, env)

    env._time += 0.5
    skill.advance(env.time, env)

    monkeypatch.setattr(nav_env_module, "extract_frontiers", lambda _grid: [])
    monkeypatch.setattr(
        nav_env_module,
        "filter_reachable_frontiers",
        lambda _raw, _grid, _robot_positions: [],
    )

    env.refresh_frontiers()

    # Frontier refresh alone should not interrupt active skills.
    assert not skill.is_done
    assert F("at", "robot1", "robot1_loc") not in env.fluents

    # Explicit interrupt request should process stale-destination rewrite.
    env.interrupt_skills()

    assert skill.is_done
    assert F("free robot1") in env.fluents

    env.refresh_frontiers()

    assert F("at", "robot1", "robot1_loc") in env.fluents
    assert F("at", "robot1", stale_destination) not in env.fluents


def test_robot_pose_updates_without_continuous_robot_loc_registry_writes():
    """Move pose updates continuously; robot_loc writes remain bounded."""
    config = NavigationConfig(
        sensor_range=9.0,
        sensor_fov_rad=2 * math.pi,
        sensor_num_rays=91,
        sensor_dt=0.05,
        speed_cells_per_sec=2.0,
        interrupt_min_new_cells=100000,
        interrupt_min_dt=100000.0,
        move_execution_interruptible=False,
    )
    env = _make_environment(config=config, two_robots=False)
    action = _first_valid_move_action(env)
    registry = env.location_registry
    assert registry is not None

    registered_keys: list[str] = []
    original_register = registry.register

    def wrapped_register(key: str, coords) -> None:  # noqa: ANN001
        registered_keys.append(key)
        original_register(key, coords)

    registry.register = wrapped_register  # type: ignore[assignment]

    sensed_positions: list[tuple[float, float]] = []
    original_observe_from_pose = env.observe_from_pose

    def wrapped_observe_from_pose(
        robot: str,
        pose: Pose,
        time: float,
        allow_interrupt: bool = True,
    ) -> int:
        sensed_positions.append((pose.x, pose.y))
        return original_observe_from_pose(
            robot, pose, time, allow_interrupt=allow_interrupt
        )

    env.observe_from_pose = wrapped_observe_from_pose  # type: ignore[assignment]
    env.act(action)

    assert len({(round(x, 3), round(y, 3)) for x, y in sensed_positions}) > 1
    robot_loc_writes = registered_keys.count("robot1_loc")
    assert robot_loc_writes <= 3
    assert robot_loc_writes < len(sensed_positions)


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

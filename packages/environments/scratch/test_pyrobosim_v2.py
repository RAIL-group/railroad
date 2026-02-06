"""Tests for PyRoboSimEnvironment (pyrobosim_v2)."""

from environments.pyrobosim_v2 import PyRoboSimEnvironment, MatplotlibWorldCanvas


def test_pyrobosim_environment_init():
    """Test PyRoboSimEnvironment loads world and derives initial state."""
    env = PyRoboSimEnvironment(
        world_file='./resources/pyrobosim_worlds/test_world.yaml',
        show_plot=False,
    )

    # Check initial fluents derived from world
    assert any(f.name == "at" for f in env.fluents)
    assert any(f.name == "free" for f in env.fluents)
    assert any(f.name == "revealed" for f in env.fluents)

    # Check objects_by_type
    assert "robot" in env.objects_by_type
    assert "location" in env.objects_by_type
    assert "object" in env.objects_by_type

    # Check robots are present
    assert len(env.objects_by_type["robot"]) >= 1

    # Check locations include robot starting locations
    robot_locs = {f"{r}_loc" for r in env.objects_by_type["robot"]}
    assert robot_locs.issubset(env.objects_by_type["location"])


def test_pyrobosim_environment_fluents_structure():
    """Test that initial fluents have correct structure."""
    env = PyRoboSimEnvironment(
        world_file='./resources/pyrobosim_worlds/test_world.yaml',
        show_plot=False,
    )

    # For each robot, check we have at, free, and revealed fluents
    for robot in env.objects_by_type["robot"]:
        robot_loc = f"{robot}_loc"

        # Check at fluent
        at_fluents = [f for f in env.fluents if f.name == "at" and robot in f.args]
        assert len(at_fluents) >= 1, f"Missing 'at' fluent for robot {robot}"

        # Check free fluent
        free_fluents = [f for f in env.fluents if f.name == "free" and robot in f.args]
        assert len(free_fluents) == 1, f"Missing 'free' fluent for robot {robot}"

        # Check revealed fluent for robot's starting location
        revealed_fluents = [f for f in env.fluents if f.name == "revealed" and robot_loc in f.args]
        assert len(revealed_fluents) == 1, f"Missing 'revealed' fluent for location {robot_loc}"


def test_pyrobosim_environment_get_move_cost_fn():
    """Test that get_move_cost_fn returns a callable."""
    env = PyRoboSimEnvironment(
        world_file='./resources/pyrobosim_worlds/test_world.yaml',
        show_plot=False,
    )

    move_cost_fn = env.get_move_cost_fn()
    assert callable(move_cost_fn)

    # Get a robot and some locations to test
    robot = next(iter(env.objects_by_type["robot"]))
    robot_loc = f"{robot}_loc"

    # Get another location from the world
    other_locs = [loc for loc in env.objects_by_type["location"] if not loc.endswith("_loc")]
    if other_locs:
        other_loc = other_locs[0]
        cost = move_cost_fn(robot, robot_loc, other_loc)
        assert isinstance(cost, float)
        assert cost > 0  # Should be positive cost


def test_pyrobosim_environment_get_objects_at_location():
    """Test get_objects_at_location returns correct structure."""
    env = PyRoboSimEnvironment(
        world_file='./resources/pyrobosim_worlds/test_world.yaml',
        show_plot=False,
    )

    # Robot locations should have no objects
    robot = next(iter(env.objects_by_type["robot"]))
    robot_loc = f"{robot}_loc"
    objects = env.get_objects_at_location(robot_loc)
    assert "object" in objects
    assert len(objects["object"]) == 0

    # Non-robot locations may have objects
    other_locs = [loc for loc in env.objects_by_type["location"] if not loc.endswith("_loc")]
    if other_locs:
        objects = env.get_objects_at_location(other_locs[0])
        assert "object" in objects


def test_pyrobosim_environment_canvas():
    """Test that canvas is initialized."""
    env = PyRoboSimEnvironment(
        world_file='./resources/pyrobosim_worlds/test_world.yaml',
        show_plot=False,
    )

    assert hasattr(env, "canvas")
    assert isinstance(env.canvas, MatplotlibWorldCanvas)

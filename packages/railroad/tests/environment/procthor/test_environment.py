"""Tests for ProcTHOREnvironment."""

import pytest

from railroad.environment.procthor import ProcTHORScene, ProcTHOREnvironment
from railroad.core import Fluent as F
from railroad._bindings import State
from railroad import operators


@pytest.fixture
def scene():
    """Create ProcTHORScene for testing."""
    return ProcTHORScene(seed=4001)


@pytest.mark.timeout(30)
def test_scene_locations(scene):
    """Test scene exposes locations."""
    assert 'start_loc' in scene.locations
    assert len(scene.locations) > 1


@pytest.mark.timeout(30)
def test_scene_objects(scene):
    """Test scene exposes objects."""
    assert len(scene.objects) > 0
    # Objects should be formatted as name_idx
    for obj in scene.objects:
        assert '_' in obj


@pytest.mark.timeout(30)
def test_scene_object_locations(scene):
    """Test scene provides ground truth object locations."""
    assert len(scene.object_locations) > 0
    for loc, objs in scene.object_locations.items():
        assert loc in scene.locations
        for obj in objs:
            assert obj in scene.objects


@pytest.mark.timeout(30)
def test_scene_move_cost_fn(scene):
    """Test move cost function."""
    cost_fn = scene.get_move_cost_fn()

    # Cost to self is 0
    cost = cost_fn("robot1", "start_loc", "start_loc")
    assert cost == 0.0

    # Cost to other location is positive
    other_loc = next(iter(scene.locations.keys() - {"start_loc"}))
    cost = cost_fn("robot1", "start_loc", other_loc)
    assert cost > 0


@pytest.mark.timeout(30)
def test_environment_creation(scene):
    """Test ProcTHOREnvironment can be created."""
    move_op = operators.construct_move_operator_blocking(scene.get_move_cost_fn())

    initial_state = State(0.0, {
        F("at robot1 start_loc"),
        F("free robot1"),
        F("revealed start_loc"),
    })

    # Pick first available object
    target_obj = next(iter(scene.objects))

    env = ProcTHOREnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={
            "robot": {"robot1"},
            "location": set(scene.locations.keys()),
            "object": {target_obj},
        },
        operators=[move_op],
    )

    assert env.scene is scene
    assert len(env.get_actions()) > 0


@pytest.mark.timeout(30)
def test_environment_validation_invalid_location(scene):
    """Test validation catches invalid locations."""
    initial_state = State(0.0, {F("at robot1 start_loc")})

    with pytest.raises(ValueError, match="not found in scene"):
        ProcTHOREnvironment(
            scene=scene,
            state=initial_state,
            objects_by_type={
                "robot": {"robot1"},
                "location": {"nonexistent_location"},
                "object": set(),
            },
            operators=[],
            validate=True,
        )


@pytest.mark.timeout(30)
def test_environment_validation_invalid_object(scene):
    """Test validation catches invalid objects."""
    initial_state = State(0.0, {F("at robot1 start_loc")})

    with pytest.raises(ValueError, match="not found in scene"):
        ProcTHOREnvironment(
            scene=scene,
            state=initial_state,
            objects_by_type={
                "robot": {"robot1"},
                "location": set(scene.locations.keys()),
                "object": {"nonexistent_object"},
            },
            operators=[],
            validate=True,
        )

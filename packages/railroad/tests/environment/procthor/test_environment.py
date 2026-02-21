"""Tests for ProcTHOREnvironment."""

import pytest

from railroad.environment.procthor import ProcTHORScene, ProcTHOREnvironment
from railroad.core import Fluent as F
from railroad._bindings import State
from railroad import operators


class _TestProcTHOREnvironment(ProcTHOREnvironment):
    """Test-local concrete ProcTHOR environment."""

    def define_operators(self):
        move_op = operators.construct_move_operator_blocking(self.estimate_move_time)
        return [move_op]


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
def test_environment_creation(scene):
    """Test ProcTHOREnvironment can be created."""
    initial_state = State(0.0, {
        F("at robot1 start_loc"),
        F("free robot1"),
        F("revealed start_loc"),
    })

    # Pick first available object
    target_obj = next(iter(scene.objects))

    env = _TestProcTHOREnvironment(
        seed=4001,
        state=initial_state,
        objects_by_type={
            "robot": {"robot1"},
            "location": set(scene.locations.keys()),
            "object": {target_obj},
        },
    )

    assert env.scene.locations == scene.locations
    assert env.location_registry is not None
    path = env.compute_move_path("start_loc", "start_loc")
    assert path.shape[0] == 2
    assert path.shape[1] >= 1
    assert len(env.get_actions()) > 0


@pytest.mark.timeout(30)
def test_environment_validation_invalid_location(scene):
    """Test validation catches invalid locations."""
    initial_state = State(0.0, {F("at robot1 start_loc")})

    with pytest.raises(ValueError, match="not found in scene"):
        _TestProcTHOREnvironment(
            seed=4001,
            state=initial_state,
            objects_by_type={
                "robot": {"robot1"},
                "location": {"nonexistent_location"},
                "object": set(),
            },
            validate=True,
        )


@pytest.mark.timeout(30)
def test_environment_validation_invalid_object(scene):
    """Test validation catches invalid objects."""
    initial_state = State(0.0, {F("at robot1 start_loc")})

    with pytest.raises(ValueError, match="not found in scene"):
        _TestProcTHOREnvironment(
            seed=4001,
            state=initial_state,
            objects_by_type={
                "robot": {"robot1"},
                "location": set(scene.locations.keys()),
                "object": {"nonexistent_object"},
            },
            validate=True,
        )


@pytest.mark.timeout(30)
def test_base_environment_requires_define_operators_override():
    """Base ProcTHOREnvironment should be abstract."""
    initial_state = State(0.0, {F("at robot1 start_loc"), F("free robot1")})
    with pytest.raises(TypeError):
        ProcTHOREnvironment(
            seed=4001,
            state=initial_state,
            objects_by_type={"robot": {"robot1"}, "location": {"start_loc"}, "object": set()},
        )

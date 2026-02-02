"""Tests for SimpleSymbolicEnvironment."""
import pytest
from railroad._bindings import Fluent as F, GroundedEffect
from railroad.core import Effect, Operator


def test_simple_symbolic_environment_construction():
    """Test basic construction of SimpleSymbolicEnvironment."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment

    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "bedroom"},
    }
    objects_at_locations = {"kitchen": {"Knife"}}

    env = SimpleSymbolicEnvironment(
        fluents=fluents,
        objects_by_type=objects_by_type,
        objects_at_locations=objects_at_locations,
    )

    assert env.fluents == fluents
    assert env.objects_by_type == objects_by_type


def test_simple_symbolic_environment_apply_effect():
    """Test applying effects modifies fluents."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment

    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    env = SimpleSymbolicEnvironment(
        fluents=fluents,
        objects_by_type={},
        objects_at_locations={},
    )

    # Create a grounded effect that removes free
    effect = GroundedEffect(
        time=0.0,
        resulting_fluents={~F("free", "robot1")},
    )

    env.apply_effect(effect)

    assert F("free", "robot1") not in env.fluents


def test_simple_symbolic_environment_apply_effect_add():
    """Test applying effects that add fluents."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment

    fluents = {F("at", "robot1", "kitchen")}
    env = SimpleSymbolicEnvironment(
        fluents=fluents,
        objects_by_type={},
        objects_at_locations={},
    )

    # Create a grounded effect that adds a fluent
    effect = GroundedEffect(
        time=0.0,
        resulting_fluents={F("free", "robot1")},
    )

    env.apply_effect(effect)

    assert F("free", "robot1") in env.fluents


def test_simple_symbolic_environment_create_skill():
    """Test skill creation via factory method."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment
    from railroad.environment.skill import SymbolicSkill

    env = SimpleSymbolicEnvironment(
        fluents=set(),
        objects_by_type={},
        objects_at_locations={},
    )

    op = Operator(
        name="test",
        parameters=[("?robot", "robot")],
        preconditions=[],
        effects=[Effect(time=1.0, resulting_fluents={F("done", "?robot")})]
    )
    action = op.instantiate({"robot": ["r1"]})[0]

    skill = env.create_skill(action, time=0.0)

    assert isinstance(skill, SymbolicSkill)
    assert skill.robot == "r1"


def test_simple_symbolic_environment_create_move_skill():
    """Test move skill creation via factory method."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment
    from railroad.environment.skill import SymbolicMoveSkill

    env = SimpleSymbolicEnvironment(
        fluents=set(),
        objects_by_type={},
        objects_at_locations={},
    )

    op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=10.0, resulting_fluents={
                ~F("at", "?robot", "?from"),
                F("at", "?robot", "?to"),
                F("free", "?robot")
            }),
        ]
    )
    actions = op.instantiate({"robot": ["r1"], "location": ["kitchen", "bedroom"]})
    action = [a for a in actions if "kitchen" in a.name and "bedroom" in a.name][0]

    skill = env.create_skill(action, time=0.0)

    assert isinstance(skill, SymbolicMoveSkill)
    assert skill.robot == "r1"
    assert skill.start == "kitchen"
    assert skill.end == "bedroom"


def test_simple_symbolic_environment_revelation():
    """Test that searching a location reveals objects at that location."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment

    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    objects_at_locations = {"kitchen": {"Knife", "Fork"}}

    env = SimpleSymbolicEnvironment(
        fluents=fluents,
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen"}},
        objects_at_locations=objects_at_locations,
    )

    # Simulate a search completing (searched fluent added)
    effect = GroundedEffect(
        time=0.0,
        resulting_fluents={F("searched", "kitchen")},
    )
    env.apply_effect(effect)

    # Verify objects were revealed
    assert F("revealed", "kitchen") in env.fluents
    assert F("found", "Knife") in env.fluents
    assert F("found", "Fork") in env.fluents
    assert F("at", "Knife", "kitchen") in env.fluents
    assert F("at", "Fork", "kitchen") in env.fluents


def test_simple_symbolic_environment_get_objects_at_location():
    """Test getting objects at a location."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment

    objects_at_locations = {"kitchen": {"Knife", "Fork"}}

    env = SimpleSymbolicEnvironment(
        fluents=set(),
        objects_by_type={},
        objects_at_locations=objects_at_locations,
    )

    result = env.get_objects_at_location("kitchen")
    assert result == {"object": {"Knife", "Fork"}}

    # Test empty location
    result = env.get_objects_at_location("bedroom")
    assert result == {"object": set()}


def test_simple_symbolic_environment_remove_object_from_location():
    """Test removing an object from a location (for pick)."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment

    objects_at_locations = {"kitchen": {"Knife", "Fork"}}

    env = SimpleSymbolicEnvironment(
        fluents=set(),
        objects_by_type={},
        objects_at_locations=objects_at_locations,
    )

    env.remove_object_from_location("Knife", "kitchen")

    assert "Knife" not in env._objects_at_locations["kitchen"]
    assert "Fork" in env._objects_at_locations["kitchen"]


def test_simple_symbolic_environment_add_object_at_location():
    """Test adding an object at a location (for place)."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment

    env = SimpleSymbolicEnvironment(
        fluents=set(),
        objects_by_type={},
        objects_at_locations={},
    )

    env.add_object_at_location("Knife", "kitchen")

    assert "Knife" in env._objects_at_locations["kitchen"]


def test_simple_symbolic_environment_resolve_probabilistic_effect():
    """Test resolving probabilistic effects based on ground truth."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment

    # Create an environment where "obj" IS at "loc"
    env = SimpleSymbolicEnvironment(
        fluents=set(),
        objects_by_type={},
        objects_at_locations={"loc": {"obj"}},  # obj is at loc
    )

    # Create a deterministic effect
    det_effect = GroundedEffect(
        time=1.0,
        resulting_fluents={F("done")},
    )

    # For non-probabilistic, should return unchanged
    effects, fluents = env.resolve_probabilistic_effect(det_effect, set())
    assert effects == [det_effect]

    # Create a probabilistic effect with proper search structure
    # Success branch has both "found obj" and "at obj loc"
    branch1_effect = GroundedEffect(
        time=0.0,
        resulting_fluents={F("found", "obj"), F("at", "obj", "loc")}
    )
    branch2_effect = GroundedEffect(time=0.0, resulting_fluents={F("searched", "loc")})
    prob_effect = GroundedEffect(
        time=2.0,
        resulting_fluents=set(),
        prob_effects=[
            (0.6, [branch1_effect]),  # success branch
            (0.4, [branch2_effect]),  # failure branch
        ],
    )

    # Since obj IS at loc in ground truth, should return success branch
    effects, fluents = env.resolve_probabilistic_effect(prob_effect, set())
    assert len(effects) == 1
    assert effects[0] == branch1_effect

    # Now test when object is NOT at location
    env2 = SimpleSymbolicEnvironment(
        fluents=set(),
        objects_by_type={},
        objects_at_locations={"other_loc": {"obj"}},  # obj is at other_loc, not loc
    )
    effects2, _ = env2.resolve_probabilistic_effect(prob_effect, set())
    assert len(effects2) == 1
    assert effects2[0] == branch2_effect  # failure branch


def test_search_skill_resolves_probabilistically():
    """Test that search skill resolves probabilistic effects via environment."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment
    from railroad.environment.interface_v2 import EnvironmentInterfaceV2
    from railroad.core import get_action_by_name
    from railroad import operators

    # Object IS at kitchen - search should succeed
    env = SimpleSymbolicEnvironment(
        fluents={F("at", "robot1", "kitchen"), F("free", "robot1")},
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen"}, "object": {"Knife"}},
        objects_at_locations={"kitchen": {"Knife"}},
    )

    search_op = operators.construct_search_operator(
        object_find_prob=0.5,  # Probability doesn't matter - ground truth does
        search_time=3.0,
    )

    interface = EnvironmentInterfaceV2(environment=env, operators=[search_op])
    actions = interface.get_actions()
    search_action = get_action_by_name(actions, "search robot1 kitchen Knife")

    interface.advance(search_action, do_interrupt=False)

    # Since Knife IS at kitchen, search should succeed
    assert F("searched", "kitchen", "Knife") in interface.state.fluents
    assert F("found", "Knife") in interface.state.fluents
    assert F("at", "Knife", "kitchen") in interface.state.fluents
    assert F("free", "robot1") in interface.state.fluents


def test_search_skill_fails_when_object_not_at_location():
    """Test that search fails when object is NOT at the searched location."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment
    from railroad.environment.interface_v2 import EnvironmentInterfaceV2
    from railroad.core import get_action_by_name
    from railroad import operators

    # Object is NOT at kitchen (it's at bedroom)
    env = SimpleSymbolicEnvironment(
        fluents={F("at", "robot1", "kitchen"), F("free", "robot1")},
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}, "object": {"Knife"}},
        objects_at_locations={"bedroom": {"Knife"}},  # Knife is NOT at kitchen
    )

    search_op = operators.construct_search_operator(
        object_find_prob=0.9,  # High probability, but ground truth says object NOT here
        search_time=3.0,
    )

    interface = EnvironmentInterfaceV2(environment=env, operators=[search_op])
    actions = interface.get_actions()
    search_action = get_action_by_name(actions, "search robot1 kitchen Knife")

    interface.advance(search_action, do_interrupt=False)

    # Search should complete but NOT find the object
    assert F("searched", "kitchen", "Knife") in interface.state.fluents
    assert F("found", "Knife") not in interface.state.fluents
    assert F("free", "robot1") in interface.state.fluents


def test_simple_symbolic_environment_create_search_skill():
    """Test search skill creation via factory method."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment
    from railroad.environment.skill import SymbolicSearchSkill
    from railroad import operators

    env = SimpleSymbolicEnvironment(
        fluents=set(),
        objects_by_type={"robot": {"r1"}, "location": {"kitchen"}, "object": {"Knife"}},
        objects_at_locations={},
    )

    search_op = operators.construct_search_operator(
        object_find_prob=0.5,
        search_time=3.0,
    )
    actions = search_op.instantiate(env.objects_by_type)
    search_action = [a for a in actions if "r1" in a.name and "kitchen" in a.name and "Knife" in a.name][0]

    skill = env.create_skill(search_action, time=0.0)

    assert isinstance(skill, SymbolicSearchSkill)
    assert skill.robot == "r1"
    assert skill.location == "kitchen"
    assert skill.target_object == "Knife"
    assert not skill.is_interruptible

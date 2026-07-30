"""Tests for SymbolicEnvironment."""

import pytest
from railroad._bindings import Fluent as F, GroundedEffect, State
from railroad.core import Effect, Operator
from railroad.environment import LocationRegistry, SymbolicEnvironment


# =============================================================================
# Construction Tests
# =============================================================================


def test_symbolic_environment_construction():
    """Test basic construction of SymbolicEnvironment."""
    initial_fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}

    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=5.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to"), F("free", "?robot")}),
        ]
    )

    env = SymbolicEnvironment(
        state=State(0.0, initial_fluents, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
        operators=[move_op],
    )

    assert env.time == 0.0
    assert F("at", "robot1", "kitchen") in env.state.fluents


# =============================================================================
# Action Execution Tests
# =============================================================================


def test_symbolic_environment_act():
    """Test acting (advancing state) with an action."""
    from railroad.core import get_action_by_name

    initial_fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}

    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=5.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to"), F("free", "?robot")}),
        ]
    )

    env = SymbolicEnvironment(
        state=State(0.0, initial_fluents, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
        operators=[move_op],
    )
    actions = env.get_actions()
    move_action = get_action_by_name(actions, "move robot1 kitchen bedroom")

    env.act(move_action)

    assert env.time == pytest.approx(5.0, abs=0.1)
    assert F("at", "robot1", "bedroom") in env.state.fluents
    assert F("free", "robot1") in env.state.fluents


def test_symbolic_environment_multi_robot_interrupt():
    """Test that robot1's move is interrupted when robot2 becomes free."""
    import numpy as np
    from railroad.environment import InterruptibleNavigationMoveSkill, LocationRegistry
    from railroad.core import get_action_by_name

    # Two robots: robot1 at kitchen, robot2 at bedroom
    initial_fluents = {
        F("at", "robot1", "kitchen"),
        F("at", "robot2", "bedroom"),
        F("free", "robot1"),
        F("free", "robot2"),
    }

    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=10.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to"), F("free", "?robot")}),
        ]
    )
    # Short action for robot2
    wait_op = Operator(
        name="wait",
        parameters=[("?robot", "robot")],
        preconditions=[F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(time=2.0, resulting_fluents={F("free", "?robot")}),
        ]
    )

    env = SymbolicEnvironment(
        state=State(0.0, initial_fluents, []),
        objects_by_type={"robot": {"robot1", "robot2"}, "location": {"kitchen", "bedroom", "living_room"}},
        operators=[move_op, wait_op],
        skill_overrides={"move": InterruptibleNavigationMoveSkill},
        location_registry=LocationRegistry({
            "kitchen": np.array([0.0, 0.0]),
            "bedroom": np.array([10.0, 0.0]),
            "living_room": np.array([10.0, 5.0]),
        }),
    )
    actions = env.get_actions()

    # Robot1 starts long move (10s)
    move_action = get_action_by_name(actions, "move robot1 kitchen living_room")
    env.act(move_action)

    # Now robot1 is busy, robot2 is still free
    assert F("free", "robot2") in env.state.fluents
    assert F("free", "robot1") not in env.state.fluents

    # Robot2 starts short wait (2s), with interrupt enabled
    actions = env.get_actions()
    wait_action = get_action_by_name(actions, "wait robot2")
    env.act(wait_action)

    # At t=2, robot2 becomes free, robot1's move should be interrupted
    assert env.time == pytest.approx(2.0, abs=0.1)
    assert F("free", "robot2") in env.state.fluents

    # Robot1 should now be at intermediate location and free
    assert F("at", "robot1", "robot1_loc") in env.state.fluents
    assert F("free", "robot1") in env.state.fluents
    assert F("at", "robot1", "living_room") not in env.state.fluents  # Did NOT reach destination


def test_symbolic_environment_apply_effect():
    """Test applying effects modifies fluents."""
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    env = SymbolicEnvironment(
        state=State(0.0, fluents, []),
        objects_by_type={},
        operators=[],
    )

    # Create a grounded effect that removes free
    effect = GroundedEffect(
        time=0.0,
        resulting_fluents={~F("free", "robot1")},
    )

    env.apply_effect(effect)

    assert F("free", "robot1") not in env.fluents


def test_symbolic_environment_apply_effect_add():
    """Test applying effects that add fluents."""
    fluents = {F("at", "robot1", "kitchen")}
    env = SymbolicEnvironment(
        state=State(0.0, fluents, []),
        objects_by_type={},
        operators=[],
    )

    # Create a grounded effect that adds a fluent
    effect = GroundedEffect(
        time=0.0,
        resulting_fluents={F("free", "robot1")},
    )

    env.apply_effect(effect)

    assert F("free", "robot1") in env.fluents


def test_symbolic_environment_apply_effect_deletes_before_adds():
    """An effect that deletes and adds the same fluent leaves it present."""
    env = SymbolicEnvironment(
        state=State(0.0, {F("at", "robot1", "kitchen")}, []),
        objects_by_type={},
        operators=[],
    )

    effect = GroundedEffect(
        time=0.0,
        resulting_fluents={
            F("at", "robot1", "kitchen"),
            ~F("at", "robot1", "kitchen"),
        },
    )
    env.apply_effect(effect)

    assert F("at", "robot1", "kitchen") in env.fluents


def test_symbolic_environment_apply_effect_conditional():
    """Conditional branches fire iff their conditions hold pre-effect."""
    env = SymbolicEnvironment(
        state=State(0.0, {F("in", "doc")}, []),
        objects_by_type={},
        operators=[],
    )

    # The effect deletes its own condition fluent; the branch must still fire
    # because conditions read the state as it was when the effect fired.
    effect = GroundedEffect(
        time=0.0,
        resulting_fluents={~F("in", "doc")},
        cond_effects=[
            ({F("in", "doc")}, [GroundedEffect(0.0, {F("moved", "doc")})]),
            ({F("in", "pen")}, [GroundedEffect(0.0, {F("moved", "pen")})]),
        ],
    )
    env.apply_effect(effect)

    assert F("moved", "doc") in env.fluents
    assert F("moved", "pen") not in env.fluents
    assert F("in", "doc") not in env.fluents


def test_symbolic_environment_act_conditional_effects():
    """env.act() applies conditional branches, matching the planner's model.

    Briefcase-style: moving relocates exactly the items inside when the move
    completes (mirrors test_forall_effect_expands_per_object in test_core).
    """
    from railroad.core import ForallEffect, get_action_by_name

    move_op = Operator(
        name="move",
        parameters=[("?from", "location"), ("?to", "location")],
        preconditions=[F("free briefcase"), F("at briefcase ?from")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free briefcase")}),
            Effect(
                time=2.0,
                resulting_fluents={
                    F("free briefcase"),
                    F("at briefcase ?to"),
                    ~F("at briefcase ?from"),
                },
                forall_effects=[ForallEffect(
                    variables=[("?obj", "item")],
                    conditions={F("in ?obj")},
                    effects=[Effect(time=0, resulting_fluents={
                        F("at ?obj ?to"), ~F("at ?obj ?from")})],
                )],
            ),
        ],
    )
    env = SymbolicEnvironment(
        state=State(0.0, {
            F("free briefcase"), F("at briefcase home"),
            F("at doc home"), F("at pen home"), F("in doc"),
        }, []),
        objects_by_type={"location": {"home", "office"}, "item": {"doc", "pen"}},
        operators=[move_op],
    )
    move = get_action_by_name(env.get_actions(), "move home office")
    env.act(move)

    assert F("at briefcase office") in env.state.fluents
    assert F("at doc office") in env.state.fluents
    assert F("at doc home") not in env.state.fluents
    assert F("at pen home") in env.state.fluents
    assert F("at pen office") not in env.state.fluents


# =============================================================================
# Skill Creation Tests
# =============================================================================


def test_symbolic_environment_create_skill():
    """Test skill creation via factory method."""
    from railroad.environment import SymbolicSkill

    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
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


def test_symbolic_environment_create_move_skill():
    """Test move skill creation via factory method."""
    import numpy as np
    from railroad.environment import InterruptibleNavigationMoveSkill, SymbolicSkill

    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
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

    # Move skills use SymbolicSkill by default (not interruptible)
    assert isinstance(skill, SymbolicSkill)
    assert not isinstance(skill, InterruptibleNavigationMoveSkill)
    assert not skill.is_interruptible

    # Can use skill_overrides to make moves interruptible
    env_with_override = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
        skill_overrides={"move": InterruptibleNavigationMoveSkill},
        location_registry=LocationRegistry({
            "kitchen": np.array([0.0, 0.0]),
            "bedroom": np.array([10.0, 0.0]),
        }),
    )
    skill_interruptible = env_with_override.create_skill(action, time=0.0)
    assert isinstance(skill_interruptible, InterruptibleNavigationMoveSkill)
    assert skill_interruptible.is_interruptible


def test_symbolic_environment_interruptible_override_requires_location_registry():
    """Env-aware interruptible move skill requires registry-backed pathing."""
    from railroad.environment import InterruptibleNavigationMoveSkill

    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
        skill_overrides={"move": InterruptibleNavigationMoveSkill},
    )

    op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(
                time=10.0,
                resulting_fluents={
                    ~F("at", "?robot", "?from"),
                    F("at", "?robot", "?to"),
                    F("free", "?robot"),
                },
            ),
        ],
    )
    actions = op.instantiate({"robot": ["r1"], "location": ["kitchen", "bedroom"]})
    action = [a for a in actions if "kitchen" in a.name and "bedroom" in a.name][0]

    with pytest.raises(TypeError, match="requires location_registry"):
        env.create_skill(action, time=0.0)


def test_symbolic_environment_create_search_skill():
    """Test search skill creation via factory method."""
    from railroad.environment import SymbolicSkill
    from railroad import operators

    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={"robot": {"r1"}, "location": {"kitchen"}, "object": {"Knife"}},
        operators=[],
    )

    search_op = operators.construct_search_operator(
        object_find_prob=0.5,
        search_time=3.0,
    )
    actions = search_op.instantiate(env.objects_by_type)
    search_action = [a for a in actions if "r1" in a.name and "kitchen" in a.name and "Knife" in a.name][0]

    skill = env.create_skill(search_action, time=0.0)

    assert isinstance(skill, SymbolicSkill)
    assert not skill.is_interruptible  # Search skills are not interruptible


def test_symbolic_environment_create_pick_skill():
    """Test pick skill creation via factory method."""
    from railroad.environment import SymbolicSkill
    from railroad import operators

    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={"robot": {"r1"}, "location": {"kitchen"}, "object": {"Knife"}},
        operators=[],
    )

    pick_op = operators.construct_pick_operator_blocking(pick_time=2.0)
    actions = pick_op.instantiate(env.objects_by_type)
    pick_action = [a for a in actions if "r1" in a.name and "kitchen" in a.name and "Knife" in a.name][0]

    skill = env.create_skill(pick_action, time=0.0)

    assert isinstance(skill, SymbolicSkill)
    assert not skill.is_interruptible  # Pick skills are not interruptible


def test_symbolic_environment_create_place_skill():
    """Test place skill creation via factory method."""
    from railroad.environment import SymbolicSkill
    from railroad import operators

    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={"robot": {"r1"}, "location": {"bedroom"}, "object": {"Knife"}},
        operators=[],
    )

    place_op = operators.construct_place_operator_blocking(place_time=2.0)
    actions = place_op.instantiate(env.objects_by_type)
    place_action = [a for a in actions if "r1" in a.name and "bedroom" in a.name and "Knife" in a.name][0]

    skill = env.create_skill(place_action, time=0.0)

    assert isinstance(skill, SymbolicSkill)
    assert not skill.is_interruptible  # Place skills are not interruptible
# =============================================================================
# Generality Tests (no domain conventions on the base class)
# =============================================================================


def test_symbolic_environment_no_special_fluent_names():
    """The generic base gives no fluent name special meaning (no revelation)."""
    env = SymbolicEnvironment(
        state=State(0.0, set(), []),
        objects_by_type={},
        operators=[],
    )

    env.apply_effect(GroundedEffect(0.0, {F("searched", "kitchen")}))

    assert F("searched", "kitchen") in env.fluents
    assert F("revealed", "kitchen") not in env.fluents


def test_symbolic_environment_no_action_name_filtering():
    """The generic base keeps actions regardless of name (e.g. PDDL `move`)."""
    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            # Zero-duration move: ObjectSearchEnvironment filters this,
            # the generic base must not.
            Effect(time=0.0, resulting_fluents={~F("at", "?robot", "?from"), F("at", "?robot", "?to")}),
        ],
    )
    env = SymbolicEnvironment(
        state=State(0.0, {F("at", "robot1", "kitchen"), F("free", "robot1")}, []),
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
        operators=[move_op],
    )
    names = {a.name for a in env.get_actions()}
    assert "move robot1 kitchen bedroom" in names


def test_symbolic_environment_seeded_probabilistic_sampling():
    """Probabilistic branches sample from the env's seeded RNG, reproducibly."""
    def make_env(seed):
        return SymbolicEnvironment(
            state=State(0.0, set(), []),
            objects_by_type={},
            operators=[],
            seed=seed,
        )

    effect = GroundedEffect(
        time=0.0,
        prob_effects=[
            (0.5, [GroundedEffect(0.0, {F("heads")})]),
            (0.5, [GroundedEffect(0.0, {F("tails")})]),
        ],
    )

    def outcomes(env, n=8):
        result = []
        for _ in range(n):
            env._fluents.discard(F("heads"))
            env._fluents.discard(F("tails"))
            env.apply_effect(effect)
            result.append(F("heads") in env.fluents)
        return result

    assert outcomes(make_env(seed=7)) == outcomes(make_env(seed=7))

# =============================================================================
# Grounding Integration (static preconditions, caching)
# =============================================================================


def test_grounding_cache_regrounds_on_universe_change():
    """get_actions() reflects objects revealed after construction."""
    from railroad.environment import ObjectSearchEnvironment
    from railroad import operators

    pick_op = operators.construct_pick_operator_blocking(pick_time=2.0)
    env = ObjectSearchEnvironment(
        state=State(0.0, {F("at r1 kitchen"), F("free r1")}, []),
        objects_by_type={"robot": {"r1"}, "location": {"kitchen"}, "object": set()},
        operators=[pick_op],
        true_object_locations={"kitchen": {"Knife"}},
    )
    assert env.get_actions() == []  # no objects known yet

    # Revelation discovers the Knife (mutating objects_by_type out-of-band);
    # the cached grounding must pick it up on the next call.
    env.apply_effect(GroundedEffect(0.0, {F("searched kitchen")}))
    names = {a.name for a in env.get_actions()}
    assert "pick r1 kitchen Knife" in names


def test_action_filter_sees_live_state_on_a_grounding_cache_hit():
    """_is_valid_action runs outside the cache, so it may read dynamic state.

    The cache key covers the object universe and static facts — exactly what
    grounding reads — but _is_valid_action is a subclass hook that can consult
    anything. UnknownSpaceEnvironment's filters on live `at` fluents and on
    observed-grid move costs are the real instance. Caching the filtered list
    would make those depend on some unrelated caller invalidating the cache
    often enough; here nothing changes the key at all between calls, so a
    cached filter verdict would be visible immediately.
    """
    move_op = Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
            Effect(time=2.0, resulting_fluents={F("free ?r"), F("at ?r ?to")}),
        ],
    )

    class BlockingEnvironment(SymbolicEnvironment):
        blocked: set[str] = set()

        def _is_valid_action(self, action):
            if not super()._is_valid_action(action):
                return False
            return action.name.split()[-1] not in self.blocked

    env = BlockingEnvironment(
        state=State(0.0, {F("at r1 kitchen"), F("free r1")}, []),
        objects_by_type={"robot": {"r1"}, "location": {"kitchen", "hall", "office"}},
        operators=[move_op],
    )

    # `at` is dynamic, so grounding emits every ordered location pair and the
    # runtime precondition decides which are applicable.
    everything = {a.name for a in env.get_actions()}
    assert everything == {
        "move r1 kitchen hall", "move r1 kitchen office",
        "move r1 hall kitchen", "move r1 hall office",
        "move r1 office kitchen", "move r1 office hall",
    }

    # Nothing the cache key tracks has changed — no new objects, no static
    # fact touched, no invalidate_grounding() call.
    env.blocked = {"office"}
    assert {a.name for a in env.get_actions()} == {
        "move r1 kitchen hall", "move r1 hall kitchen", "move r1 office kitchen",
        "move r1 office hall",
    }

    env.blocked = set()
    assert {a.name for a in env.get_actions()} == everything


def test_environment_static_preconditions_end_to_end():
    """Connected-domain exemplar: static facts constrain grounding and are
    stripped from grounded actions, while staying in the environment state."""
    from railroad.core import get_action_by_name

    move_op = Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r"), F("connected ?from ?to")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
            Effect(time=2.0, resulting_fluents={F("free ?r"), F("at ?r ?to")}),
        ],
    )
    env = SymbolicEnvironment(
        state=State(0.0, {
            F("at r1 kitchen"), F("free r1"),
            F("connected kitchen hall"), F("connected hall office"),
        }, []),
        objects_by_type={"robot": {"r1"},
                         "location": {"kitchen", "hall", "office"}},
        operators=[move_op],
    )

    before = set(env.state.fluents)
    actions = env.get_actions()
    names = {a.name for a in actions}
    assert names == {"move r1 kitchen hall", "move r1 hall office"}

    # Verified static preconditions are gone from the grounded actions...
    move = get_action_by_name(actions, "move r1 kitchen hall")
    assert not any(f.name == "connected" for f in move.preconditions)
    # ...but grounding never mutates the environment's state. Dropping
    # search-irrelevant fluents belongs to the planner, which (unlike the
    # environment) can see the goal and the state's queued effects.
    assert set(env.state.fluents) == before
    assert F("connected kitchen hall") in env.state.fluents
    assert F("connected kitchen hall") in env.static_facts

    env.act(get_action_by_name(env.get_actions(), "move r1 kitchen hall"))
    env.act(get_action_by_name(env.get_actions(), "move r1 hall office"))
    assert F("at r1 office") in env.state.fluents
    assert env.is_goal_reached([F("at r1 office"), F("connected hall office")])


def test_initial_upcoming_effects_keep_their_branches():
    """State-carried effects survive conversion into the bootstrap skill.

    Probabilistic and conditional branches are rebuilt alongside the effect's
    own fluents; only the top-level time is rebased onto the start time.
    """
    from railroad.core import get_action_by_name

    effect = GroundedEffect(
        time=2.0,
        resulting_fluents={F("started")},
        prob_effects=[(1.0, [GroundedEffect(0.0, {F("finished")})])],
        cond_effects=[({F("armed")}, [GroundedEffect(0.0, {F("fired")})])],
    )
    # Occupies the robot until t=3, so act() runs past the t=2 initial effect.
    wait = Operator(
        name="wait", parameters=[("?r", "robot")], preconditions=[F("free ?r")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free ?r")}),
            Effect(time=3.0, resulting_fluents={F("free ?r")}),
        ],
    )
    # Keeps `armed` dynamic so grounding does not compile it out of the state;
    # this test is about branch preservation, not static elimination.
    disarm = Operator(
        name="disarm", parameters=[], preconditions=[F("armed")],
        effects=[Effect(time=1.0, resulting_fluents={~F("armed")})],
    )
    env = SymbolicEnvironment(
        state=State(0.0, {F("free r1"), F("armed")}, [(2.0, effect)]),
        objects_by_type={"robot": {"r1"}},
        operators=[wait, disarm],
    )

    ((_, rebuilt),) = env._active_skills[0].upcoming_effects
    assert rebuilt.is_probabilistic
    assert rebuilt.is_conditional

    # And they actually fire when the effect comes due.
    env.act(get_action_by_name(env.get_actions(), "wait r1"))
    assert F("started") in env.state.fluents
    assert F("finished") in env.state.fluents
    assert F("fired") in env.state.fluents


def test_grounding_keeps_facts_read_by_queued_effect_conditions():
    """Regression: grounding must not compile out a fact a `when` reads.

    `armed` is touched by no operator effect, so grounding classifies it as
    static and unreferenced — it cannot see that an effect carried in the
    state conditions on it. Removing it from the runtime state used to make
    the branch silently not fire.
    """
    from railroad.core import get_action_by_name

    effect = GroundedEffect(
        time=2.0,
        resulting_fluents={F("started")},
        cond_effects=[({F("armed")}, [GroundedEffect(0.0, {F("fired")})])],
    )
    wait = Operator(
        name="wait", parameters=[("?r", "robot")], preconditions=[F("free ?r")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free ?r")}),
            Effect(time=3.0, resulting_fluents={F("free ?r")}),
        ],
    )
    env = SymbolicEnvironment(
        state=State(0.0, {F("free r1"), F("armed")}, [(2.0, effect)]),
        objects_by_type={"robot": {"r1"}},
        operators=[wait],
    )

    env.act(get_action_by_name(env.get_actions(), "wait r1"))
    assert F("started") in env.state.fluents
    assert F("fired") in env.state.fluents

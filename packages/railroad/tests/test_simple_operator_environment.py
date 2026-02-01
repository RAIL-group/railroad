"""Tests for SimpleOperatorEnvironment."""

import pytest
from railroad.core import Fluent as F, State, get_action_by_name
from railroad.environment import SimpleOperatorEnvironment, EnvironmentInterface, SkillStatus
from railroad import operators


class TestSimpleOperatorEnvironmentConstruction:
    """Tests for basic construction."""

    def test_basic_construction(self):
        """Test that environment can be constructed with operators and objects."""
        move_op = operators.construct_move_operator_blocking(move_time=5.0)
        search_op = operators.construct_search_operator(0.5, search_time=3.0)

        objects_at_locations = {
            "kitchen": {"Knife", "Mug"},
            "bedroom": {"Pillow"},
        }

        env = SimpleOperatorEnvironment(
            operators=[move_op, search_op],
            objects_at_locations=objects_at_locations,
        )

        assert env.time == 0.0
        assert len(env.operators) == 2

    def test_empty_locations(self):
        """Test construction with empty location sets."""
        move_op = operators.construct_move_operator_blocking(move_time=5.0)

        objects_at_locations = {
            "kitchen": {"Knife"},
            "bedroom": set(),
        }

        env = SimpleOperatorEnvironment(
            operators=[move_op],
            objects_at_locations=objects_at_locations,
        )

        assert env.get_objects_at_location("bedroom") == {"object": set()}
        assert env.get_objects_at_location("kitchen") == {"object": {"Knife"}}


class TestObjectTracking:
    """Tests for object location tracking."""

    def test_get_objects_at_location(self):
        """Test retrieving objects at a location."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={
                "kitchen": {"Knife", "Mug"},
                "bedroom": {"Pillow"},
            },
        )

        result = env.get_objects_at_location("kitchen")
        assert result == {"object": {"Knife", "Mug"}}

    def test_get_objects_at_unknown_location(self):
        """Test retrieving objects at a location that doesn't exist."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={"kitchen": {"Knife"}},
        )

        result = env.get_objects_at_location("unknown_room")
        assert result == {"object": set()}

    def test_remove_object_from_location(self):
        """Test removing an object (simulating pick)."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={"kitchen": {"Knife", "Mug"}},
        )

        env.remove_object_from_location("Knife", "kitchen")

        assert env.get_objects_at_location("kitchen") == {"object": {"Mug"}}

    def test_remove_nonexistent_object(self):
        """Test removing an object that doesn't exist (should not error)."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={"kitchen": {"Knife"}},
        )

        # Should not raise
        env.remove_object_from_location("Mug", "kitchen")
        assert env.get_objects_at_location("kitchen") == {"object": {"Knife"}}

    def test_add_object_at_location(self):
        """Test adding an object (simulating place)."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={"kitchen": {"Knife"}},
        )

        env.add_object_at_location("Mug", "kitchen")

        assert env.get_objects_at_location("kitchen") == {"object": {"Knife", "Mug"}}

    def test_add_object_at_new_location(self):
        """Test adding an object to a location that didn't exist."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={"kitchen": {"Knife"}},
        )

        env.add_object_at_location("Pillow", "bedroom")

        assert env.get_objects_at_location("bedroom") == {"object": {"Pillow"}}

    def test_pick_then_place_workflow(self):
        """Test a typical pick-then-place workflow."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={
                "kitchen": {"Knife"},
                "bedroom": set(),
            },
        )

        # Pick from kitchen
        env.remove_object_from_location("Knife", "kitchen")
        assert env.get_objects_at_location("kitchen") == {"object": set()}

        # Place in bedroom
        env.add_object_at_location("Knife", "bedroom")
        assert env.get_objects_at_location("bedroom") == {"object": {"Knife"}}


class TestSkillExecution:
    """Tests for skill execution and timing."""

    def test_execute_skill_with_duration(self):
        """Test executing a skill with duration kwarg."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={},
        )

        env.execute_skill("robot1", "move", "kitchen", "bedroom", duration=5.0)

        # Robot should be tracked as busy
        assert env._robot_busy["robot1"] is True
        assert env._robot_skill_tracking["robot1"] == (0.0, 5.0)

    def test_execute_skill_without_duration_raises(self):
        """Test that executing without duration raises ValueError."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={},
        )

        with pytest.raises(ValueError, match="requires 'duration' kwarg"):
            env.execute_skill("robot1", "move", "kitchen", "bedroom")

    def test_get_skills_time_fn_raises(self):
        """Test that get_skills_time_fn raises NotImplementedError."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={},
        )

        with pytest.raises(NotImplementedError):
            env.get_skills_time_fn("move")

    def test_stop_robot(self):
        """Test stopping a robot clears its tracking."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={},
        )

        env.execute_skill("robot1", "move", "kitchen", "bedroom", duration=5.0)
        assert env._robot_busy["robot1"] is True

        env.stop_robot("robot1")

        assert env._robot_busy["robot1"] is False
        assert "robot1" not in env._robot_skill_tracking


class TestMultiRobotCoordination:
    """Tests for multi-robot skill status tracking."""

    def test_single_robot_idle_before_skill(self):
        """Test robot is IDLE before any skill is executed."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={},
        )

        status = env.get_executed_skill_status("robot1", "move")
        assert status == SkillStatus.IDLE

    def test_single_robot_is_done(self):
        """Test single robot is DONE when it's the only one with a skill."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={},
        )

        # Only robot1 has a skill
        env.execute_skill("robot1", "move", "a", "b", duration=5.0)

        # With only one robot, it should be DONE (no one else to wait for)
        status = env.get_executed_skill_status("robot1", "move")
        assert status == SkillStatus.DONE

    def test_two_robots_done_order(self):
        """Test that the robot finishing first gets DONE status."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={},
        )

        # robot1 takes 5.0, robot2 takes 10.0
        env.execute_skill("robot1", "move", "a", "b", duration=5.0)
        env.execute_skill("robot2", "move", "c", "d", duration=10.0)

        # Both are busy, robot1 finishes first
        status1 = env.get_executed_skill_status("robot1", "move")
        status2 = env.get_executed_skill_status("robot2", "move")

        assert status1 == SkillStatus.DONE
        assert status2 == SkillStatus.RUNNING

    def test_two_robots_same_duration(self):
        """Test that both robots get DONE if they finish at the same time."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={},
        )

        env.execute_skill("robot1", "move", "a", "b", duration=5.0)
        env.execute_skill("robot2", "move", "c", "d", duration=5.0)

        status1 = env.get_executed_skill_status("robot1", "move")
        status2 = env.get_executed_skill_status("robot2", "move")

        assert status1 == SkillStatus.DONE
        assert status2 == SkillStatus.DONE

    def test_time_progression_affects_remaining(self):
        """Test that time progression affects remaining time calculation."""
        env = SimpleOperatorEnvironment(
            operators=[],
            objects_at_locations={},
        )

        # robot1: 5.0 duration, robot2: 8.0 duration
        env.execute_skill("robot1", "move", "a", "b", duration=5.0)
        env.execute_skill("robot2", "move", "c", "d", duration=8.0)

        # Initially robot1 finishes first
        assert env.get_executed_skill_status("robot1", "move") == SkillStatus.DONE
        assert env.get_executed_skill_status("robot2", "move") == SkillStatus.RUNNING

        # Advance time by 5.0 (robot1's duration) and stop robot1
        env.time = 5.0
        env.stop_robot("robot1")

        # Now start a new skill for robot1 with longer duration
        env.execute_skill("robot1", "pick", "b", "obj", duration=10.0)

        # robot2 has 3.0 remaining (8.0 - 5.0), robot1 has 10.0
        # robot2 should finish first now
        assert env.get_executed_skill_status("robot1", "pick") == SkillStatus.RUNNING
        assert env.get_executed_skill_status("robot2", "move") == SkillStatus.DONE


class TestOperatorTimingIntegration:
    """Integration tests verifying operator timing flows through to state time."""

    def test_move_time_affects_state_time(self):
        """Test that move_time parameter in operator affects elapsed time."""
        move_time = 7.5

        move_op = operators.construct_move_operator_blocking(move_time=move_time)
        env = SimpleOperatorEnvironment(
            operators=[move_op],
            objects_at_locations={},
        )

        initial_state = State(
            time=0,
            fluents={F("at robot1 kitchen"), F("free robot1")},
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["kitchen", "bedroom"],
        }

        sim = EnvironmentInterface(initial_state, objects_by_type, [move_op], env)
        actions = sim.get_actions()
        move_action = get_action_by_name(actions, "move robot1 kitchen bedroom")

        sim.advance(move_action, do_interrupt=False)

        # Time should have advanced by move_time (plus small epsilon for just-moved effect)
        assert sim.state.time == pytest.approx(move_time, abs=0.2)

    def test_different_move_times_produce_different_elapsed_times(self):
        """Test that different move_time values result in different elapsed times."""
        for move_time in [3.0, 10.0, 25.0]:
            move_op = operators.construct_move_operator_blocking(move_time=move_time)
            env = SimpleOperatorEnvironment(
                operators=[move_op],
                objects_at_locations={},
            )

            initial_state = State(
                time=0,
                fluents={F("at robot1 kitchen"), F("free robot1")},
            )
            objects_by_type = {
                "robot": ["robot1"],
                "location": ["kitchen", "bedroom"],
            }

            sim = EnvironmentInterface(initial_state, objects_by_type, [move_op], env)
            actions = sim.get_actions()
            move_action = get_action_by_name(actions, "move robot1 kitchen bedroom")

            sim.advance(move_action, do_interrupt=False)

            assert sim.state.time == pytest.approx(move_time, abs=0.2), (
                f"Expected time ~{move_time}, got {sim.state.time}"
            )

    def test_pick_time_affects_state_time(self):
        """Test that pick_time parameter in operator affects elapsed time."""
        pick_time = 12.0

        pick_op = operators.construct_pick_operator_blocking(pick_time=pick_time)
        env = SimpleOperatorEnvironment(
            operators=[pick_op],
            objects_at_locations={"kitchen": {"Knife"}},
        )

        initial_state = State(
            time=0,
            fluents={
                F("at robot1 kitchen"),
                F("free robot1"),
                F("at Knife kitchen"),
                F("found Knife"),
            },
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["kitchen"],
            "object": ["Knife"],
        }

        sim = EnvironmentInterface(initial_state, objects_by_type, [pick_op], env)
        actions = sim.get_actions()
        pick_action = get_action_by_name(actions, "pick robot1 kitchen Knife")

        sim.advance(pick_action, do_interrupt=False)

        assert sim.state.time == pytest.approx(pick_time, abs=0.2)

    def test_search_time_affects_state_time(self):
        """Test that search_time parameter in operator affects elapsed time."""
        search_time = 8.0

        search_op = operators.construct_search_operator(
            object_find_prob=1.0,  # Always find
            search_time=search_time,
        )
        env = SimpleOperatorEnvironment(
            operators=[search_op],
            objects_at_locations={"kitchen": {"Knife"}},
        )

        initial_state = State(
            time=0,
            fluents={F("at robot1 kitchen"), F("free robot1")},
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["kitchen"],
            "object": ["Knife"],
        }

        sim = EnvironmentInterface(initial_state, objects_by_type, [search_op], env)
        actions = sim.get_actions()
        search_action = get_action_by_name(actions, "search robot1 kitchen Knife")

        sim.advance(search_action, do_interrupt=False)

        assert sim.state.time == pytest.approx(search_time, abs=0.2)

    @pytest.mark.parametrize("search_time", [3.0, 7.0, 15.0])
    def test_different_search_times_produce_different_elapsed_times(self, search_time: float):
        """Test that different search_time values result in different elapsed times."""
        search_op = operators.construct_search_operator(
            object_find_prob=1.0,
            search_time=search_time,
        )
        env = SimpleOperatorEnvironment(
            operators=[search_op],
            objects_at_locations={"kitchen": {"Knife"}},
        )

        initial_state = State(
            time=0,
            fluents={F("at robot1 kitchen"), F("free robot1")},
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["kitchen"],
            "object": ["Knife"],
        }

        sim = EnvironmentInterface(initial_state, objects_by_type, [search_op], env)
        actions = sim.get_actions()
        search_action = get_action_by_name(actions, "search robot1 kitchen Knife")

        sim.advance(search_action, do_interrupt=False)

        assert sim.state.time == pytest.approx(search_time, abs=0.2), (
            f"Expected time ~{search_time}, got {sim.state.time}"
        )

    def test_search_time_same_for_success_and_failure(self):
        """Test that construct_search_operator uses same time for success/failure.

        The search operator has probabilistic outcomes but both take the same time.
        This test verifies that by checking elapsed time regardless of find probability.
        """
        search_time = 6.0

        # Test with 100% find probability (success)
        search_op_success = operators.construct_search_operator(
            object_find_prob=1.0,
            search_time=search_time,
        )
        env_success = SimpleOperatorEnvironment(
            operators=[search_op_success],
            objects_at_locations={"kitchen": {"Knife"}},
        )
        state_success = State(time=0, fluents={F("at robot1 kitchen"), F("free robot1")})
        sim_success = EnvironmentInterface(
            state_success, {"robot": ["robot1"], "location": ["kitchen"], "object": ["Knife"]},
            [search_op_success], env_success
        )
        action_success = get_action_by_name(sim_success.get_actions(), "search robot1 kitchen Knife")
        sim_success.advance(action_success, do_interrupt=False)

        # Test with 0% find probability (failure)
        search_op_fail = operators.construct_search_operator(
            object_find_prob=0.0,
            search_time=search_time,
        )
        env_fail = SimpleOperatorEnvironment(
            operators=[search_op_fail],
            objects_at_locations={"kitchen": set()},  # No knife here
        )
        state_fail = State(time=0, fluents={F("at robot1 kitchen"), F("free robot1")})
        sim_fail = EnvironmentInterface(
            state_fail, {"robot": ["robot1"], "location": ["kitchen"], "object": ["Knife"]},
            [search_op_fail], env_fail
        )
        action_fail = get_action_by_name(sim_fail.get_actions(), "search robot1 kitchen Knife")
        sim_fail.advance(action_fail, do_interrupt=False)

        # Both should take the same time
        assert sim_success.state.time == pytest.approx(search_time, abs=0.2)
        assert sim_fail.state.time == pytest.approx(search_time, abs=0.2)
        assert sim_success.state.time == pytest.approx(sim_fail.state.time, abs=0.1)


class TestSearchAndPickTiming:
    """Tests for search_and_pick operator timing behavior.

    The construct_search_and_pick_operator has different timing for success vs failure:
    - Success: move_time + pick_time (robot moves, finds object, picks it up)
    - Failure: move_time only (robot moves, doesn't find object, becomes free)
    """

    def test_search_and_pick_success_applies_all_effects(self):
        """Test that search_and_pick applies all effects on success (100% find prob)."""
        move_time = 5.0
        pick_time = 3.0

        search_pick_op = operators.construct_search_and_pick_operator(
            object_find_prob=1.0,  # Always find
            move_time=move_time,
            pick_time=pick_time,
        )
        env = SimpleOperatorEnvironment(
            operators=[search_pick_op],
            objects_at_locations={"kitchen": {"Knife"}},
        )

        initial_state = State(
            time=0,
            fluents={F("at robot1 living_room"), F("free robot1")},
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["living_room", "kitchen"],
            "object": ["Knife"],
        }

        sim = EnvironmentInterface(initial_state, objects_by_type, [search_pick_op], env)
        action = get_action_by_name(sim.get_actions(), "search robot1 living_room kitchen Knife")

        sim.advance(action, do_interrupt=False)

        # Verify success effects
        assert F("at robot1 kitchen") in sim.state.fluents, "Robot should be at kitchen"
        assert F("found Knife") in sim.state.fluents, "Knife should be found"
        assert F("holding robot1 Knife") in sim.state.fluents, "Robot should be holding Knife"
        assert F("free robot1") in sim.state.fluents, "Robot should be free"

    def test_search_and_pick_failure_applies_correct_effects(self):
        """Test that search_and_pick applies correct effects on failure (0% find prob)."""
        move_time = 5.0
        pick_time = 3.0

        search_pick_op = operators.construct_search_and_pick_operator(
            object_find_prob=0.0,  # Never find
            move_time=move_time,
            pick_time=pick_time,
        )
        env = SimpleOperatorEnvironment(
            operators=[search_pick_op],
            objects_at_locations={"kitchen": set()},
        )

        initial_state = State(
            time=0,
            fluents={F("at robot1 living_room"), F("free robot1")},
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["living_room", "kitchen"],
            "object": ["Knife"],
        }

        sim = EnvironmentInterface(initial_state, objects_by_type, [search_pick_op], env)
        action = get_action_by_name(sim.get_actions(), "search robot1 living_room kitchen Knife")

        sim.advance(action, do_interrupt=False)

        # Verify failure effects
        assert F("at robot1 kitchen") in sim.state.fluents, "Robot should be at kitchen"
        assert F("free robot1") in sim.state.fluents, "Robot should be free"
        assert F("found Knife") not in sim.state.fluents, "Knife should NOT be found"
        assert F("holding robot1 Knife") not in sim.state.fluents, "Robot should NOT be holding"

    def test_search_and_pick_duration_is_move_time_only(self):
        """Test that computed duration is move_time, not move_time + pick_time.

        This demonstrates that max(eff.time) only sees top-level effects,
        not nested probabilistic effect times.
        """
        move_time = 5.0
        pick_time = 3.0

        search_pick_op = operators.construct_search_and_pick_operator(
            object_find_prob=1.0,
            move_time=move_time,
            pick_time=pick_time,
        )

        # Check what duration would be computed
        actions = search_pick_op.instantiate({
            "robot": ["robot1"],
            "location": ["living_room", "kitchen"],
            "object": ["Knife"],
        })
        action = get_action_by_name(actions, "search robot1 living_room kitchen Knife")

        # Duration is max of top-level effect times
        computed_duration = max(eff.time for eff in action.effects)

        # This will be move_time, not move_time + pick_time
        # because pick_time is nested inside probabilistic effects
        assert computed_duration == pytest.approx(move_time, abs=0.1), (
            f"Duration should be move_time ({move_time}), not move_time + pick_time ({move_time + pick_time}). "
            f"Got {computed_duration}"
        )

    def test_sequential_actions_accumulate_time(self):
        """Test that sequential actions accumulate their times correctly."""
        move_time = 5.0
        pick_time = 3.0

        move_op = operators.construct_move_operator_blocking(move_time=move_time)
        pick_op = operators.construct_pick_operator_blocking(pick_time=pick_time)

        env = SimpleOperatorEnvironment(
            operators=[move_op, pick_op],
            objects_at_locations={"bedroom": {"Pillow"}},
        )

        initial_state = State(
            time=0,
            fluents={
                F("at robot1 kitchen"),
                F("free robot1"),
                F("at Pillow bedroom"),
                F("found Pillow"),
            },
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["kitchen", "bedroom"],
            "object": ["Pillow"],
        }

        sim = EnvironmentInterface(initial_state, objects_by_type, [move_op, pick_op], env)

        # Move from kitchen to bedroom
        actions = sim.get_actions()
        move_action = get_action_by_name(actions, "move robot1 kitchen bedroom")
        sim.advance(move_action, do_interrupt=False)

        time_after_move = sim.state.time
        assert time_after_move == pytest.approx(move_time, abs=0.2)

        # Pick up Pillow
        actions = sim.get_actions()
        pick_action = get_action_by_name(actions, "pick robot1 bedroom Pillow")
        sim.advance(pick_action, do_interrupt=False)

        # Total time should be move_time + pick_time
        expected_total = move_time + pick_time
        assert sim.state.time == pytest.approx(expected_total, abs=0.3)

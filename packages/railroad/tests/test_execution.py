"""Tests for the railroad.execution module.

Tests the simplified execution interface that handles:
- Basic action execution (move, pick, place)
- Simple probabilistic effects (search)
- Nested effects inside prob_effects (Issue #6 fix)
- Search resolution based on ground truth
- Multi-step planning workflows
"""

import pytest
from railroad.core import Fluent as F, State, get_action_by_name
from railroad.execution import Environment, EnvironmentInterface, OngoingAction
from railroad import operators


class TestEnvironmentConstruction:
    """Tests for Environment class construction."""

    def test_basic_construction(self):
        """Test Environment can be constructed with objects at locations."""
        objects_at_locations = {
            "kitchen": {"Knife", "Mug"},
            "bedroom": {"Pillow"},
        }
        env = Environment(objects_at_locations)

        assert env.get_objects_at_location("kitchen") == {"Knife", "Mug"}
        assert env.get_objects_at_location("bedroom") == {"Pillow"}

    def test_empty_locations(self):
        """Test construction with empty location sets."""
        objects_at_locations = {
            "kitchen": {"Knife"},
            "bedroom": set(),
        }
        env = Environment(objects_at_locations)

        assert env.get_objects_at_location("bedroom") == set()
        assert env.get_objects_at_location("kitchen") == {"Knife"}

    def test_unknown_location_returns_empty_set(self):
        """Test that unknown locations return empty set."""
        env = Environment({"kitchen": {"Knife"}})
        assert env.get_objects_at_location("unknown") == set()


class TestEnvironmentObjectTracking:
    """Tests for Environment object tracking."""

    def test_remove_object_from_location(self):
        """Test removing an object (simulating pick)."""
        env = Environment({"kitchen": {"Knife", "Mug"}})
        env.remove_object_from_location("Knife", "kitchen")
        assert env.get_objects_at_location("kitchen") == {"Mug"}

    def test_remove_nonexistent_object(self):
        """Test removing an object that doesn't exist (should not error)."""
        env = Environment({"kitchen": {"Knife"}})
        env.remove_object_from_location("Mug", "kitchen")
        assert env.get_objects_at_location("kitchen") == {"Knife"}

    def test_add_object_at_location(self):
        """Test adding an object (simulating place)."""
        env = Environment({"kitchen": {"Knife"}})
        env.add_object_at_location("Mug", "kitchen")
        assert env.get_objects_at_location("kitchen") == {"Knife", "Mug"}

    def test_add_object_at_new_location(self):
        """Test adding an object to a location that didn't exist."""
        env = Environment({"kitchen": {"Knife"}})
        env.add_object_at_location("Pillow", "bedroom")
        assert env.get_objects_at_location("bedroom") == {"Pillow"}


class TestEnvironmentInterfaceConstruction:
    """Tests for EnvironmentInterface construction."""

    def test_basic_construction(self):
        """Test EnvironmentInterface can be constructed."""
        move_op = operators.construct_move_operator_blocking(move_time=5.0)
        env = Environment({"kitchen": set()})

        initial_state = State(
            time=0,
            fluents={F("at robot1 kitchen"), F("free robot1")},
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["kitchen", "bedroom"],
        }

        interface = EnvironmentInterface(
            initial_state, objects_by_type, [move_op], env
        )

        assert interface.time == 0
        assert F("at robot1 kitchen") in interface.state.fluents


class TestBasicActionExecution:
    """Tests for basic action execution (move, pick, place)."""

    def test_move_action(self):
        """Test executing a move action."""
        move_time = 5.0
        move_op = operators.construct_move_operator_blocking(move_time=move_time)
        env = Environment({})

        initial_state = State(
            time=0,
            fluents={F("at robot1 kitchen"), F("free robot1")},
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["kitchen", "bedroom"],
        }

        interface = EnvironmentInterface(
            initial_state, objects_by_type, [move_op], env
        )

        actions = interface.get_actions()
        move_action = get_action_by_name(actions, "move robot1 kitchen bedroom")

        interface.advance(move_action)

        # Verify robot moved
        assert F("at robot1 bedroom") in interface.state.fluents
        assert F("at robot1 kitchen") not in interface.state.fluents
        assert F("free robot1") in interface.state.fluents

        # Verify time advanced
        assert interface.state.time == pytest.approx(move_time, abs=0.2)

    def test_pick_action(self):
        """Test executing a pick action."""
        pick_time = 3.0
        pick_op = operators.construct_pick_operator_blocking(pick_time=pick_time)
        env = Environment({"kitchen": {"Knife"}})

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

        interface = EnvironmentInterface(
            initial_state, objects_by_type, [pick_op], env
        )

        actions = interface.get_actions()
        pick_action = get_action_by_name(actions, "pick robot1 kitchen Knife")

        interface.advance(pick_action)

        # Verify object picked
        assert F("holding robot1 Knife") in interface.state.fluents
        assert F("at Knife kitchen") not in interface.state.fluents
        assert F("hand-full robot1") in interface.state.fluents

        # Verify time advanced
        assert interface.state.time == pytest.approx(pick_time, abs=0.2)

    def test_place_action(self):
        """Test executing a place action."""
        place_time = 3.0
        place_op = operators.construct_place_operator_blocking(place_time=place_time)
        env = Environment({})

        initial_state = State(
            time=0,
            fluents={
                F("at robot1 kitchen"),
                F("free robot1"),
                F("holding robot1 Knife"),
                F("hand-full robot1"),
            },
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["kitchen"],
            "object": ["Knife"],
        }

        interface = EnvironmentInterface(
            initial_state, objects_by_type, [place_op], env
        )

        actions = interface.get_actions()
        place_action = get_action_by_name(actions, "place robot1 kitchen Knife")

        interface.advance(place_action)

        # Verify object placed
        assert F("at Knife kitchen") in interface.state.fluents
        assert F("holding robot1 Knife") not in interface.state.fluents
        assert F("hand-full robot1") not in interface.state.fluents

        # Verify time advanced
        assert interface.state.time == pytest.approx(place_time, abs=0.2)


class TestSearchProbabilisticEffects:
    """Tests for search action with probabilistic effects."""

    def test_search_finds_object_when_present(self):
        """Test search finds object when it's at the location."""
        search_op = operators.construct_search_operator(
            object_find_prob=1.0,  # 100% find probability
            search_time=3.0,
        )
        env = Environment({"kitchen": {"Knife"}})  # Knife is at kitchen

        initial_state = State(
            time=0,
            fluents={F("at robot1 kitchen"), F("free robot1")},
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["kitchen"],
            "object": ["Knife"],
        }

        interface = EnvironmentInterface(
            initial_state, objects_by_type, [search_op], env
        )

        actions = interface.get_actions()
        search_action = get_action_by_name(actions, "search robot1 kitchen Knife")

        interface.advance(search_action)

        # Object should be found (because it's at the location)
        assert F("found Knife") in interface.state.fluents
        assert F("searched kitchen Knife") in interface.state.fluents

    def test_search_does_not_find_object_when_absent(self):
        """Test search doesn't find object when it's not at the location."""
        search_op = operators.construct_search_operator(
            object_find_prob=1.0,  # High probability but object isn't there
            search_time=3.0,
        )
        env = Environment({"kitchen": set()})  # Knife is NOT at kitchen

        initial_state = State(
            time=0,
            fluents={F("at robot1 kitchen"), F("free robot1")},
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["kitchen"],
            "object": ["Knife"],
        }

        interface = EnvironmentInterface(
            initial_state, objects_by_type, [search_op], env
        )

        actions = interface.get_actions()
        search_action = get_action_by_name(actions, "search robot1 kitchen Knife")

        interface.advance(search_action)

        # Object should NOT be found (it's not there)
        assert F("found Knife") not in interface.state.fluents
        assert F("searched kitchen Knife") in interface.state.fluents


class TestNestedEffects:
    """Tests for nested effects inside prob_effects (Issue #6 fix)."""

    def test_search_and_pick_success_applies_nested_effects(self):
        """Test search_and_pick applies nested pick effects on success.

        This is the key test for Issue #6 - the nested Effect with pick_time
        inside the success branch should be scheduled and applied.
        """
        move_time = 5.0
        pick_time = 3.0

        search_pick_op = operators.construct_search_and_pick_operator(
            object_find_prob=1.0,  # Always find
            move_time=move_time,
            pick_time=pick_time,
        )
        env = Environment({"kitchen": {"Knife"}})

        initial_state = State(
            time=0,
            fluents={F("at robot1 living_room"), F("free robot1")},
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["living_room", "kitchen"],
            "object": ["Knife"],
        }

        interface = EnvironmentInterface(
            initial_state, objects_by_type, [search_pick_op], env
        )

        actions = interface.get_actions()
        action = get_action_by_name(actions, "search robot1 living_room kitchen Knife")

        interface.advance(action)

        # Verify success: robot moved, found object, AND picked it up
        assert F("at robot1 kitchen") in interface.state.fluents
        assert F("found Knife") in interface.state.fluents
        assert F("holding robot1 Knife") in interface.state.fluents, \
            "Nested pick effect should be applied (Issue #6 fix)"
        assert F("free robot1") in interface.state.fluents

        # Time should include both move_time AND pick_time
        # (pick_time is the nested effect)
        expected_time = move_time + pick_time
        assert interface.state.time == pytest.approx(expected_time, abs=0.2), \
            f"Expected time ~{expected_time} (move + pick), got {interface.state.time}"

    def test_search_and_pick_failure_only_applies_failure_branch(self):
        """Test search_and_pick only applies failure branch when object not found."""
        move_time = 5.0
        pick_time = 3.0

        search_pick_op = operators.construct_search_and_pick_operator(
            object_find_prob=1.0,  # High prob but object not there
            move_time=move_time,
            pick_time=pick_time,
        )
        env = Environment({"kitchen": set()})  # No knife at kitchen

        initial_state = State(
            time=0,
            fluents={F("at robot1 living_room"), F("free robot1")},
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["living_room", "kitchen"],
            "object": ["Knife"],
        }

        interface = EnvironmentInterface(
            initial_state, objects_by_type, [search_pick_op], env
        )

        actions = interface.get_actions()
        action = get_action_by_name(actions, "search robot1 living_room kitchen Knife")

        interface.advance(action)

        # Verify failure: robot moved but didn't find or pick
        assert F("at robot1 kitchen") in interface.state.fluents
        assert F("free robot1") in interface.state.fluents
        assert F("found Knife") not in interface.state.fluents
        assert F("holding robot1 Knife") not in interface.state.fluents

        # Time should only be move_time (no pick_time on failure)
        assert interface.state.time == pytest.approx(move_time, abs=0.2)


class TestMultiStepWorkflows:
    """Tests for multi-step planning workflows."""

    def test_move_then_pick(self):
        """Test a workflow: move to location, then pick up object."""
        move_time = 5.0
        pick_time = 3.0

        move_op = operators.construct_move_operator_blocking(move_time=move_time)
        pick_op = operators.construct_pick_operator_blocking(pick_time=pick_time)

        env = Environment({"bedroom": {"Pillow"}})

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

        interface = EnvironmentInterface(
            initial_state, objects_by_type, [move_op, pick_op], env
        )

        # Step 1: Move to bedroom
        actions = interface.get_actions()
        move_action = get_action_by_name(actions, "move robot1 kitchen bedroom")
        interface.advance(move_action)

        assert F("at robot1 bedroom") in interface.state.fluents
        time_after_move = interface.state.time

        # Step 2: Pick up Pillow
        actions = interface.get_actions()
        pick_action = get_action_by_name(actions, "pick robot1 bedroom Pillow")
        interface.advance(pick_action)

        assert F("holding robot1 Pillow") in interface.state.fluents
        assert interface.state.time == pytest.approx(
            time_after_move + pick_time, abs=0.2
        )

    def test_search_then_pick(self):
        """Test a workflow: search for object, then pick it up."""
        search_time = 3.0
        pick_time = 2.0

        search_op = operators.construct_search_operator(
            object_find_prob=1.0,
            search_time=search_time,
        )
        pick_op = operators.construct_pick_operator_blocking(pick_time=pick_time)

        env = Environment({"kitchen": {"Knife"}})

        initial_state = State(
            time=0,
            fluents={F("at robot1 kitchen"), F("free robot1")},
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["kitchen"],
            "object": ["Knife"],
        }

        interface = EnvironmentInterface(
            initial_state, objects_by_type, [search_op, pick_op], env
        )

        # Step 1: Search for Knife
        actions = interface.get_actions()
        search_action = get_action_by_name(actions, "search robot1 kitchen Knife")
        interface.advance(search_action)

        assert F("found Knife") in interface.state.fluents
        assert F("at Knife kitchen") in interface.state.fluents
        time_after_search = interface.state.time

        # Step 2: Pick up Knife
        actions = interface.get_actions()
        pick_action = get_action_by_name(actions, "pick robot1 kitchen Knife")
        interface.advance(pick_action)

        assert F("holding robot1 Knife") in interface.state.fluents
        assert interface.state.time == pytest.approx(
            time_after_search + pick_time, abs=0.2
        )


class TestGoalChecking:
    """Tests for goal checking functionality."""

    def test_is_goal_reached_single_fluent(self):
        """Test checking a single goal fluent."""
        env = Environment({})
        initial_state = State(
            time=0,
            fluents={F("at robot1 kitchen"), F("free robot1")},
        )

        interface = EnvironmentInterface(
            initial_state,
            {"robot": ["robot1"], "location": ["kitchen"]},
            [],
            env,
        )

        assert interface.is_goal_reached([F("at robot1 kitchen")])
        assert not interface.is_goal_reached([F("at robot1 bedroom")])

    def test_is_goal_reached_multiple_fluents(self):
        """Test checking multiple goal fluents."""
        env = Environment({})
        initial_state = State(
            time=0,
            fluents={
                F("at robot1 kitchen"),
                F("free robot1"),
                F("holding robot1 Knife"),
            },
        )

        interface = EnvironmentInterface(
            initial_state,
            {"robot": ["robot1"], "location": ["kitchen"], "object": ["Knife"]},
            [],
            env,
        )

        # All fluents present
        assert interface.is_goal_reached([
            F("at robot1 kitchen"),
            F("holding robot1 Knife"),
        ])

        # One fluent missing
        assert not interface.is_goal_reached([
            F("at robot1 kitchen"),
            F("holding robot1 Mug"),
        ])


class TestActionPreconditionChecking:
    """Tests for action precondition validation."""

    def test_action_with_unsatisfied_precondition_raises(self):
        """Test that executing an action with unsatisfied preconditions raises."""
        move_op = operators.construct_move_operator_blocking(move_time=5.0)
        env = Environment({})

        initial_state = State(
            time=0,
            fluents={F("at robot1 kitchen")},  # Missing 'free robot1'
        )
        objects_by_type = {
            "robot": ["robot1"],
            "location": ["kitchen", "bedroom"],
        }

        interface = EnvironmentInterface(
            initial_state, objects_by_type, [move_op], env
        )

        actions = interface.get_actions()
        move_action = get_action_by_name(actions, "move robot1 kitchen bedroom")

        with pytest.raises(ValueError, match="preconditions not satisfied"):
            interface.advance(move_action)

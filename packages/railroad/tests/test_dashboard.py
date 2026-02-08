"""Unit tests for dashboard internal logic.

Tests pure functions and calculable numeric outputs — avoids testing
Rich panels or matplotlib rendering (fragile, not meaningful).
"""

import math

import pytest

from railroad._bindings import (
    Fluent,
    LiteralGoal,
    AndGoal,
    OrGoal,
    TrueGoal,
    FalseGoal,
    State,
)
from railroad.core import Fluent as F
from railroad.dashboard._goals import format_goal, get_satisfied_branch, get_best_branch
from railroad.dashboard._tui import _shorten_name, _generate_coordinates
from railroad.dashboard import PlannerDashboard
from railroad.environment import SymbolicEnvironment
from railroad import operators


# ------------------------------------------------------------------ #
# Goal analysis functions
# ------------------------------------------------------------------ #

class TestFormatGoal:
    def test_literal(self):
        goal = LiteralGoal(F("at r1 kitchen"))
        assert format_goal(goal) == "(at r1 kitchen)"

    def test_true_goal(self):
        assert format_goal(TrueGoal()) == "TRUE"

    def test_false_goal(self):
        assert format_goal(FalseGoal()) == "FALSE"

    def test_and_compact_two_literals(self):
        goal = F("at r1 kitchen") & F("at r2 bedroom")
        result = format_goal(goal, compact=True)
        assert result.startswith("AND(")
        assert "(at r1 kitchen)" in result
        assert "(at r2 bedroom)" in result
        # Compact: single line
        assert "\n" not in result

    def test_and_expanded_three_literals(self):
        goal = AndGoal([
            LiteralGoal(F("a")),
            LiteralGoal(F("b")),
            LiteralGoal(F("c")),
        ])
        result = format_goal(goal, compact=True)
        # 3 literals → multi-line even with compact=True
        assert "\n" in result

    def test_nested_and_or(self):
        goal = AndGoal([
            LiteralGoal(F("a")),
            OrGoal([LiteralGoal(F("b")), LiteralGoal(F("c"))]),
        ])
        result = format_goal(goal, compact=False)
        assert "AND(" in result
        assert "OR(" in result
        # Check indentation depth
        lines = result.split("\n")
        or_line = [l for l in lines if "OR(" in l][0]
        assert or_line.startswith("  ")  # indented one level


class TestGetSatisfiedBranch:
    def test_literal_satisfied(self):
        goal = LiteralGoal(F("at r1 kitchen"))
        fluents = {F("at r1 kitchen")}
        result = get_satisfied_branch(goal, fluents)
        assert result is not None

    def test_literal_unsatisfied(self):
        goal = LiteralGoal(F("at r1 kitchen"))
        fluents: set[Fluent] = set()
        result = get_satisfied_branch(goal, fluents)
        assert result is None

    def test_or_first_satisfied(self):
        goal = OrGoal([LiteralGoal(F("a")), LiteralGoal(F("b"))])
        fluents = {F("a")}
        result = get_satisfied_branch(goal, fluents)
        assert result is not None

    def test_and_both_satisfied(self):
        goal = AndGoal([LiteralGoal(F("a")), LiteralGoal(F("b"))])
        fluents = {F("a"), F("b")}
        result = get_satisfied_branch(goal, fluents)
        assert result is not None

    def test_and_partial_returns_none(self):
        goal = AndGoal([LiteralGoal(F("a")), LiteralGoal(F("b"))])
        fluents = {F("a")}
        result = get_satisfied_branch(goal, fluents)
        assert result is None


class TestGetBestBranch:
    def test_or_picks_satisfied(self):
        goal = OrGoal([LiteralGoal(F("a")), LiteralGoal(F("b"))])
        fluents = {F("a")}
        result = get_best_branch(goal, fluents)
        # Should pick branch with A (ratio 1.0 > 0.0)
        assert result is not None
        assert isinstance(result, LiteralGoal)

    def test_or_of_ands_picks_better(self):
        branch1 = AndGoal([LiteralGoal(F("a")), LiteralGoal(F("b"))])
        branch2 = AndGoal([LiteralGoal(F("c")), LiteralGoal(F("d"))])
        goal = OrGoal([branch1, branch2])
        fluents = {F("a")}
        result = get_best_branch(goal, fluents)
        # Branch1 has ratio 0.5, branch2 has 0.0 → picks branch1
        assert result is not None
        literals = result.get_all_literals()
        literal_names = {f.name for f in literals}
        assert "a" in literal_names

    def test_and_or_nested(self):
        goal = AndGoal([
            LiteralGoal(F("a")),
            OrGoal([LiteralGoal(F("b")), LiteralGoal(F("c"))]),
        ])
        fluents = {F("a"), F("b")}
        result = get_best_branch(goal, fluents)
        assert result is not None
        literals = result.get_all_literals()
        literal_names = {f.name for f in literals}
        assert "a" in literal_names
        assert "b" in literal_names


# ------------------------------------------------------------------ #
# Best-path progress (PlannerDashboard._compute_best_path_progress)
# ------------------------------------------------------------------ #

class TestComputeBestPathProgress:
    """Test the recursive best-path progress computation.

    Uses a minimal PlannerDashboard with mocked environment.
    """

    @pytest.fixture
    def dashboard(self):
        """Minimal dashboard for testing _compute_best_path_progress."""
        move_op = operators.construct_move_operator_blocking(lambda r, a, b: 1.0)
        no_op = operators.construct_no_op_operator(no_op_time=1.0, extra_cost=10.0)
        initial_state = State(0.0, {F("at r1 kitchen"), F("free r1")}, [])
        env = SymbolicEnvironment(
            state=initial_state,
            objects_by_type={
                "robot": {"r1"},
                "location": {"kitchen", "bedroom"},
            },
            operators=[move_op, no_op],
        )
        goal = F("at r1 bedroom")
        return PlannerDashboard(
            goal, env,
            force_interactive=False,
            print_on_exit=False,
        )

    def test_literal_satisfied(self, dashboard):
        goal = LiteralGoal(F("a"))
        fluents = {F("a")}
        assert dashboard._compute_best_path_progress(goal, fluents) == (1, 1)

    def test_literal_unsatisfied(self, dashboard):
        goal = LiteralGoal(F("a"))
        fluents: set[Fluent] = set()
        assert dashboard._compute_best_path_progress(goal, fluents) == (0, 1)

    def test_and_both_satisfied(self, dashboard):
        goal = AndGoal([LiteralGoal(F("a")), LiteralGoal(F("b"))])
        fluents = {F("a"), F("b")}
        assert dashboard._compute_best_path_progress(goal, fluents) == (2, 2)

    def test_and_one_satisfied(self, dashboard):
        goal = AndGoal([LiteralGoal(F("a")), LiteralGoal(F("b"))])
        fluents = {F("a")}
        assert dashboard._compute_best_path_progress(goal, fluents) == (1, 2)

    def test_or_one_satisfied(self, dashboard):
        goal = OrGoal([LiteralGoal(F("a")), LiteralGoal(F("b"))])
        fluents = {F("a")}
        assert dashboard._compute_best_path_progress(goal, fluents) == (1, 1)

    def test_and_or_nested_both(self, dashboard):
        goal = AndGoal([
            LiteralGoal(F("a")),
            OrGoal([LiteralGoal(F("b")), LiteralGoal(F("c"))]),
        ])
        fluents = {F("a"), F("b")}
        assert dashboard._compute_best_path_progress(goal, fluents) == (2, 2)

    def test_and_or_nested_partial(self, dashboard):
        goal = AndGoal([
            LiteralGoal(F("a")),
            OrGoal([LiteralGoal(F("b")), LiteralGoal(F("c"))]),
        ])
        fluents = {F("a")}
        assert dashboard._compute_best_path_progress(goal, fluents) == (1, 2)


# ------------------------------------------------------------------ #
# Utility functions
# ------------------------------------------------------------------ #

class TestShortenName:
    @pytest.mark.parametrize("name,expected", [
        ("crawler", "c"),
        ("robot1", "r1"),
        ("BigRedRobot", "BRR"),
        ("myRobot3", "mR3"),
    ])
    def test_shorten(self, name, expected):
        assert _shorten_name(name) == expected


class TestGenerateCoordinates:
    def test_empty(self):
        assert _generate_coordinates([]) == {}

    def test_single(self):
        result = _generate_coordinates(["kitchen"])
        assert result == {"kitchen": (0.0, 0.0)}

    def test_n_on_unit_circle(self):
        names = ["a", "b", "c", "d"]
        result = _generate_coordinates(names)
        assert len(result) == 4
        for name in names:
            x, y = result[name]
            assert math.isclose(x**2 + y**2, 1.0, rel_tol=1e-9)


# ------------------------------------------------------------------ #
# Trajectory interpolation
# ------------------------------------------------------------------ #

class TestGetEntityPositionsAtTimes:
    """Test interpolation of entity positions at query times."""

    @pytest.fixture
    def dashboard_with_trajectory(self):
        """Dashboard with known entity positions for interpolation testing."""
        move_op = operators.construct_move_operator_blocking(lambda r, a, b: 10.0)
        no_op = operators.construct_no_op_operator(no_op_time=1.0, extra_cost=10.0)
        initial_state = State(0.0, {F("at r1 A"), F("free r1")}, [])
        env = SymbolicEnvironment(
            state=initial_state,
            objects_by_type={
                "robot": {"r1"},
                "location": {"A", "B"},
            },
            operators=[move_op, no_op],
        )
        goal = F("at r1 B")
        db = PlannerDashboard(
            goal, env,
            force_interactive=False,
            print_on_exit=False,
        )
        db.known_robots = {"r1"}
        # Manually set up trajectory: A at (0,0) time 0 -> B at (10,0) time 10
        db._entity_positions = {
            "r1": [
                (0.0, "A", None),
                (10.0, "B", None),
            ],
        }
        db._goal_time = 10.0
        return db

    def test_query_at_start(self, dashboard_with_trajectory):
        import numpy as np
        db = dashboard_with_trajectory
        coords = {"A": (0.0, 0.0), "B": (10.0, 0.0)}
        result = db.get_entity_positions_at_times(
            [0.0], location_coords=coords,
        )
        assert "r1" in result
        np.testing.assert_allclose(result["r1"][0], [0.0, 0.0], atol=1e-6)

    def test_query_at_end(self, dashboard_with_trajectory):
        import numpy as np
        db = dashboard_with_trajectory
        coords = {"A": (0.0, 0.0), "B": (10.0, 0.0)}
        result = db.get_entity_positions_at_times(
            [10.0], location_coords=coords,
        )
        assert "r1" in result
        np.testing.assert_allclose(result["r1"][0], [10.0, 0.0], atol=1e-6)

    def test_query_at_midpoint(self, dashboard_with_trajectory):
        import numpy as np
        db = dashboard_with_trajectory
        coords = {"A": (0.0, 0.0), "B": (10.0, 0.0)}
        result = db.get_entity_positions_at_times(
            [5.0], location_coords=coords,
        )
        assert "r1" in result
        np.testing.assert_allclose(result["r1"][0], [5.0, 0.0], atol=1e-6)

    def test_query_before_start_clamps(self, dashboard_with_trajectory):
        import numpy as np
        db = dashboard_with_trajectory
        coords = {"A": (0.0, 0.0), "B": (10.0, 0.0)}
        result = db.get_entity_positions_at_times(
            [-5.0], location_coords=coords,
        )
        assert "r1" in result
        np.testing.assert_allclose(result["r1"][0], [0.0, 0.0], atol=1e-6)

    def test_query_after_end_clamps(self, dashboard_with_trajectory):
        import numpy as np
        db = dashboard_with_trajectory
        coords = {"A": (0.0, 0.0), "B": (10.0, 0.0)}
        result = db.get_entity_positions_at_times(
            [20.0], location_coords=coords,
        )
        assert "r1" in result
        np.testing.assert_allclose(result["r1"][0], [10.0, 0.0], atol=1e-6)

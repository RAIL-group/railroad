"""Unit tests for dashboard internal logic.

Tests pure functions and calculable numeric outputs — avoids testing
Rich panels or matplotlib rendering (fragile, not meaningful).
"""

import math

import pytest

from railroad._bindings import (
    LiteralGoal,
    AndGoal,
    OrGoal,
    TrueGoal,
    FalseGoal,
)
from railroad.core import Fluent as F
from railroad.dashboard._goals import format_goal, get_satisfied_branch, get_best_branch
from railroad.dashboard._tui import _shorten_name, _generate_coordinates

from env_helpers import move_dashboard


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
        or_line = [line for line in lines if "OR(" in line][0]
        assert or_line.startswith("  ")  # indented one level


class TestGetSatisfiedBranch:
    """Which branch comes back, not merely whether one does.

    The three positive cases used to assert `result is not None`, which an
    implementation returning any arbitrary goal would satisfy. The `{b}` row is
    the one that discriminates: with only the *second* disjunct satisfied, a
    "return the first child" bug returns non-None and would have passed.
    """

    A, B = LiteralGoal(F("a")), LiteralGoal(F("b"))
    LIT = LiteralGoal(F("at r1 kitchen"))

    @pytest.mark.parametrize(
        ("goal", "fluents", "expected"),
        [
            (LIT, {F("at r1 kitchen")}, LIT),
            (LIT, set(), None),
            (OrGoal([A, B]), {F("a")}, A),
            (OrGoal([A, B]), {F("b")}, B),
            (AndGoal([A, B]), {F("a"), F("b")}, AndGoal([A, B])),
            (AndGoal([A, B]), {F("a")}, None),
        ],
        ids=["literal_satisfied", "literal_unsatisfied", "or_first", "or_second",
             "and_both", "and_partial"],
    )
    def test_returns_the_satisfied_branch(self, goal, fluents, expected):
        assert get_satisfied_branch(goal, fluents) == expected


class TestGetBestBranch:
    def test_or_picks_satisfied(self):
        a, b = LiteralGoal(F("a")), LiteralGoal(F("b"))
        goal = OrGoal([a, b])
        # Pin the branch, not its type: `isinstance(result, LiteralGoal)` was
        # true of either child, so it could not detect picking the wrong one.
        assert get_best_branch(goal, {F("a")}) == a
        assert get_best_branch(goal, {F("b")}) == b

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
        return move_dashboard(
            locations=("kitchen", "bedroom"), start="kitchen", move_time=1.0,
            goal_loc="bedroom", trajectory=None,
        )

    @pytest.mark.parametrize(
        ("goal", "fluents", "expected"),
        [
            (LiteralGoal(F("a")), {F("a")}, (1, 1)),
            (LiteralGoal(F("a")), set(), (0, 1)),
            (AndGoal([LiteralGoal(F("a")), LiteralGoal(F("b"))]),
             {F("a"), F("b")}, (2, 2)),
            (AndGoal([LiteralGoal(F("a")), LiteralGoal(F("b"))]),
             {F("a")}, (1, 2)),
            # An OR contributes only its best branch, so the denominator is 1.
            (OrGoal([LiteralGoal(F("a")), LiteralGoal(F("b"))]), {F("a")}, (1, 1)),
            (AndGoal([LiteralGoal(F("a")),
                      OrGoal([LiteralGoal(F("b")), LiteralGoal(F("c"))])]),
             {F("a"), F("b")}, (2, 2)),
            (AndGoal([LiteralGoal(F("a")),
                      OrGoal([LiteralGoal(F("b")), LiteralGoal(F("c"))])]),
             {F("a")}, (1, 2)),
        ],
        ids=["literal_satisfied", "literal_unsatisfied", "and_both", "and_one",
             "or_one", "and_or_nested_both", "and_or_nested_partial"],
    )
    def test_progress_counts_the_best_path(self, dashboard, goal, fluents, expected):
        assert dashboard._compute_best_path_progress(goal, fluents) == expected


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
        """r1 travels A (0,0) -> B (10,0) between t=0 and t=10."""
        return move_dashboard()

    COORDS = {"A": (0.0, 0.0), "B": (10.0, 0.0)}

    @pytest.mark.parametrize(
        ("query_time", "expected"),
        [
            (0.0, [0.0, 0.0]),
            (10.0, [10.0, 0.0]),
            (5.0, [5.0, 0.0]),
            # Outside the trajectory, clamp to its ends rather than
            # extrapolating off the map.
            (-5.0, [0.0, 0.0]),
            (20.0, [10.0, 0.0]),
        ],
        ids=["at_start", "at_end", "at_midpoint", "before_start_clamps",
             "after_end_clamps"],
    )
    def test_position_is_interpolated_and_clamped(
        self, dashboard_with_trajectory, query_time, expected,
    ):
        import numpy as np
        result = dashboard_with_trajectory.get_entity_positions_at_times(
            [query_time], location_coords=self.COORDS,
        )
        assert "r1" in result
        np.testing.assert_allclose(result["r1"][0], expected, atol=1e-6)

    def test_stationary_segment_holds_position(self, dashboard_with_trajectory):
        """A B->B segment (e.g. pick/place) must keep the robot at B for the
        full duration, not drift toward the next move's destination."""
        import numpy as np
        db = dashboard_with_trajectory
        # Add a stationary hold at B from t=10 to t=20 (a "pick"), then a move
        # back to A from t=20 to t=30. Without the fix, distance-based timing
        # collapses the stationary segment and interp leaks into the next move.
        db._entity_positions = {
            "r1": [
                (0.0, "A", None),
                (10.0, "B", None),
                (20.0, "B", None),
                (30.0, "A", None),
            ],
        }
        db._goal_time = 30.0
        coords = {"A": (0.0, 0.0), "B": (10.0, 0.0)}
        result = db.get_entity_positions_at_times(
            [10.0, 15.0, 20.0], location_coords=coords,
        )
        assert "r1" in result
        # All three query points are within the stationary B->B window.
        np.testing.assert_allclose(result["r1"][0], [10.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result["r1"][1], [10.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result["r1"][2], [10.0, 0.0], atol=1e-6)


# ------------------------------------------------------------------ #
# get_plot_image
# ------------------------------------------------------------------ #

class TestGetPlotImage:
    """Test the get_plot_image() method for JPEG rendering."""

    def test_returns_jpeg_bytes(self):
        """Dashboard with entity positions produces valid JPEG bytes."""
        db = move_dashboard()
        image_bytes = db.get_plot_image(
            location_coords={"A": (0.0, 0.0), "B": (10.0, 0.0)},
        )

        assert image_bytes is not None
        # JPEG magic bytes
        assert image_bytes[:3] == b"\xff\xd8\xff"
        assert len(image_bytes) > 100

    def test_returns_none_when_no_trajectories(self):
        """Empty dashboard with no entity positions returns None."""
        db = move_dashboard(
            locations=("kitchen",), start="kitchen", goal_loc="kitchen",
            move_time=1.0, trajectory=None,
        )

        image_bytes = db.get_plot_image()
        assert image_bytes is None

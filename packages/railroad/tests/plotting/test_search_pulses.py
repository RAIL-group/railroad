"""The search ring's timing, geometry and scale."""

import numpy as np
import pytest

from railroad.core import Fluent as F
from railroad.plotting.pulses import (
    PULSE_MIN_FRACTION,
    PULSE_RADIUS_M,
    SearchWindow,
    by_robot,
    pulse_radius,
    sample,
    search_windows,
)

ROBOTS = {"r1", "r2"}


def entry(time, *fluents):
    return {"time": time, "fluents": {F(value) for value in fluents}}


def windows(actions, history, *, horizon=100.0, known_robots=ROBOTS):
    return search_windows(
        actions_taken=actions, history=history,
        known_robots=known_robots, horizon=horizon,
    )


class TestSearchWindows:
    def test_runs_from_dispatch_to_the_snapshot_that_frees_the_robot(self):
        assert windows(
            [("search r1 shelf mug", 0.0)],
            [entry(0, "free r1"), entry(4, "at r1 shelf"), entry(10, "free r1")],
        ) == [SearchWindow("r1", 0.0, 10.0)]

    def test_the_dispatch_snapshot_does_not_end_the_window_it_starts(self):
        """`free r1` is still in the state the action was chosen from.

        The dashboard records an action against the time of the state
        *before* it ran, which is the last one in which the robot was free.
        Matching that snapshot would collapse every window to nothing.
        """
        assert windows(
            [("search r1 shelf mug", 4.0)],
            [entry(4, "free r1"), entry(14, "free r1")],
        ) == [SearchWindow("r1", 4.0, 14.0)]

    def test_another_robot_freeing_mid_search_does_not_end_it(self):
        assert windows(
            [("search r1 shelf mug", 0.0)],
            [entry(0, "free r1", "free r2"), entry(3, "free r2"), entry(9, "free r1")],
        ) == [SearchWindow("r1", 0.0, 9.0)]

    def test_a_search_still_running_at_the_end_runs_to_the_horizon(self):
        assert windows(
            [("search r1 shelf mug", 20.0)],
            [entry(0, "free r1"), entry(20, "free r1"), entry(26, "at r1 shelf")],
            horizon=26.0,
        ) == [SearchWindow("r1", 20.0, 26.0)]

    def test_search_frontier_counts_too(self):
        """Its result lands under `searched-frontier`, but `free` is `free`."""
        assert windows(
            [("search-frontier r1 frontier_2 mug", 0.0)],
            [entry(0, "free r1"), entry(5, "free r1")],
        ) == [SearchWindow("r1", 0.0, 5.0)]

    @pytest.mark.parametrize(
        "action", ["move r1 shelf counter", "pick r1 shelf mug", "researched r1 shelf"],
    )
    def test_other_actions_are_ignored(self, action):
        assert windows([(action, 0.0)], [entry(5, "free r1")]) == []

    def test_an_action_naming_no_known_robot_is_ignored(self):
        assert windows(
            [("search ghost shelf mug", 0.0)], [entry(5, "free ghost")],
        ) == []

    def test_each_search_gets_its_own_window(self):
        assert windows(
            [("search r1 shelf mug", 0.0), ("search r1 counter mug", 12.0)],
            [entry(0, "free r1"), entry(10, "free r1"), entry(12, "free r1"),
             entry(22, "free r1")],
        ) == [SearchWindow("r1", 0.0, 10.0), SearchWindow("r1", 12.0, 22.0)]

    def test_windows_group_by_robot(self):
        grouped = by_robot(
            windows(
                [("search r1 shelf mug", 0.0), ("search r2 counter mug", 0.0)],
                [entry(0, "free r1", "free r2"), entry(6, "free r2"),
                 entry(10, "free r1")],
            )
        )
        assert grouped == {
            "r1": [SearchWindow("r1", 0.0, 10.0)],
            "r2": [SearchWindow("r2", 0.0, 6.0)],
        }


class TestSample:
    WINDOW = SearchWindow("r1", 10.0, 20.0)

    def test_grows_from_small_to_full_and_fades_as_it_goes(self):
        radii, alpha = sample(
            [self.WINDOW], [10.0, 15.0, 19.999], radius=4.0,
        )
        assert radii[0] == pytest.approx(4.0 * PULSE_MIN_FRACTION)
        assert radii[-1] == pytest.approx(4.0, rel=1e-3)
        assert list(radii) == sorted(radii)
        assert alpha[0] == pytest.approx(1.0)
        assert alpha[-1] == pytest.approx(0.0, abs=1e-3)

    def test_is_invisible_outside_the_window(self):
        radii, alpha = sample([self.WINDOW], [0.0, 9.99, 20.0, 40.0], radius=4.0)
        assert not radii.any()
        assert not alpha.any()

    def test_a_degenerate_window_draws_nothing(self):
        radii, alpha = sample(
            [SearchWindow("r1", 5.0, 5.0)], [4.0, 5.0, 6.0], radius=4.0,
        )
        assert not radii.any() and not alpha.any()

    def test_consecutive_windows_each_restart_the_ring(self):
        radii, _alpha = sample(
            [SearchWindow("r1", 0.0, 10.0), SearchWindow("r1", 10.0, 20.0)],
            [9.0, 10.0], radius=4.0,
        )
        assert radii[1] < radii[0], "the second search starts a new, small ring"

    def test_no_windows_leaves_every_frame_hidden(self):
        radii, alpha = sample([], np.linspace(0.0, 10.0, 5), radius=4.0)
        assert radii.shape == alpha.shape == (5,)
        assert not radii.any() and not alpha.any()


class TestRadius:
    def test_metres_become_cells_through_the_scene_resolution(self):
        assert pulse_radius(None, 0.05) == pytest.approx(PULSE_RADIUS_M / 0.05)

    def test_without_a_resolution_it_falls_back_to_the_axes_scale(self):
        """A symbolic plot has no metres, so the ring is sized on screen."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from railroad.plotting.sprites import ring_radius, sprite_extent

        fig, ax = plt.subplots()
        try:
            ax.plot([0.0, 40.0], [0.0, 40.0])
            expected = ring_radius(1, sprite_extent(ax))
            assert pulse_radius(ax, None) == pytest.approx(expected)
            assert expected > 0.0
        finally:
            plt.close(fig)

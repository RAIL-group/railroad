"""The search ring's timing, geometry and scale."""

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


def windows(actions, history, *, horizon=100.0):
    return search_windows(
        actions_taken=actions, history=history,
        known_robots=ROBOTS, horizon=horizon,
    )


@pytest.mark.parametrize(
    "action", ["search r1 shelf mug", "search-frontier r1 frontier_2 mug"],
)
def test_a_window_runs_from_dispatch_to_the_snapshot_that_frees_the_robot(action):
    """Three traps in one history.

    The dispatch snapshot still holds `free r1` -- the dashboard records an
    action against the state it was chosen *from* -- and matching it would
    collapse every window to nothing. r2 freeing partway through must not end
    r1's search either. And `search-frontier` files its result under its own
    predicate, which is why the window keys on `free`.
    """
    assert windows(
        [(action, 0.0)],
        [entry(0, "free r1", "free r2"), entry(3, "free r2"), entry(10, "free r1")],
    ) == [SearchWindow("r1", 0.0, 10.0)]


def test_a_search_still_running_at_the_end_runs_to_the_horizon():
    assert windows(
        [("search r1 shelf mug", 20.0)],
        [entry(20, "free r1"), entry(26, "at r1 shelf")], horizon=26.0,
    ) == [SearchWindow("r1", 20.0, 26.0)]


def test_every_search_gets_a_window_and_nothing_else_does():
    """Two robots, one of them searching twice, and two actions to ignore."""
    assert by_robot(
        windows(
            [("search r1 shelf mug", 0.0), ("search r2 counter mug", 0.0),
             ("move r1 shelf counter", 4.0), ("search ghost shelf mug", 4.0),
             ("search r1 counter mug", 10.0)],
            [entry(0, "free r1", "free r2"), entry(6, "free r2"),
             entry(10, "free r1"), entry(22, "free r1")],
        )
    ) == {
        "r1": [SearchWindow("r1", 0.0, 10.0), SearchWindow("r1", 10.0, 22.0)],
        "r2": [SearchWindow("r2", 0.0, 6.0)],
    }


def test_the_ring_grows_and_fades_over_the_search_and_is_hidden_outside_it():
    radii, alpha = sample(
        [SearchWindow("r1", 10.0, 20.0)],
        [9.99, 10.0, 15.0, 19.999, 20.0], radius=4.0,
    )
    assert (radii[0], alpha[0]) == (0.0, 0.0), "nothing before the search"
    assert (radii[-1], alpha[-1]) == (0.0, 0.0), "nothing after it"
    assert radii[1] == pytest.approx(4.0 * PULSE_MIN_FRACTION)
    assert radii[3] == pytest.approx(4.0, rel=1e-3)
    assert list(radii[1:4]) == sorted(radii[1:4])
    assert alpha[1] == pytest.approx(1.0)
    assert alpha[3] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize(
    "searches", [[SearchWindow("r1", 5.0, 5.0)], []],
    ids=["a window of no length", "no search at all"],
)
def test_nothing_is_drawn_when_there_is_no_search_to_draw(searches):
    radii, alpha = sample(searches, [4.0, 5.0, 6.0], radius=4.0)
    assert radii.shape == alpha.shape == (3,)
    assert not radii.any() and not alpha.any()


def test_metres_become_cells_through_the_scene_resolution():
    assert pulse_radius(None, 0.05) == pytest.approx(PULSE_RADIUS_M / 0.05)


def test_without_a_resolution_the_ring_is_sized_on_screen():
    """A symbolic plot has no metres, so there is nothing to convert."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        ax.plot([0.0, 40.0], [0.0, 40.0])
        assert 0.0 < pulse_radius(ax, None) < 40.0
    finally:
        plt.close(fig)

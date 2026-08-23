"""Shared plan fixtures for the sprite tests.

At the tests root rather than under ``dashboard/`` because the compositing
suite in ``test_video_compositing.py`` drives the same plan, and a conftest one
directory down is invisible to it.

Environment construction helpers live in ``env_helpers.py`` next door, so that
static tooling can resolve them by an unambiguous module name.
"""

import pytest

from railroad import operators
from railroad._bindings import State
from railroad.core import Fluent as F, get_action_by_name
from railroad.dashboard import PlannerDashboard
from railroad.environment import ObjectSearchEnvironment


#: Markers this package's tests use, registered here rather than in its own
#: pyproject. A [tool.pytest.ini_options] table there would make pytest choose
#: packages/railroad as rootdir for any targeted run -- which drops the
#: repo-root conftest (and with it the warnings tally) entirely. Registering
#: from a conftest keeps the markers known when the package is tested
#: standalone, without touching rootdir either way.
_MARKERS = {
    "slow": "marks tests as slow to run (deselect with '-m \"not slow\"')",
    "stochastic": (
        "outcome depends on planner sampling "
        "(deselect with '-m \"not stochastic\"')"
    ),
}


def pytest_configure(config):
    """Register the markers the root pyproject declares, if it is not in play."""
    already = {line.split(":", 1)[0].strip() for line in config.getini("markers")}
    for name, description in _MARKERS.items():
        if name not in already:
            config.addinivalue_line("markers", f"{name}: {description}")


PLAN = (
    "search r1 shelf mug",
    "pick r1 shelf mug",
    "move r1 shelf counter",
    "place r1 counter mug",
)


class FetchEnvironment(ObjectSearchEnvironment):
    """Search a shelf, carry what is found to the counter."""

    def define_operators(self):
        return [
            operators.construct_search_operator(1.0, 10.0),
            operators.construct_pick_operator_blocking(4.0),
            operators.construct_move_operator_blocking(lambda r, a, b: 8.0),
            operators.construct_place_operator_blocking(6.0),
        ]


@pytest.fixture
def fetch_dashboard():
    """A dashboard holding a completed fetch plan, ready to plot.

    ``sponge`` is on the shelf too and is revealed by the same search, so this
    also exercises the filter that keeps it off the map.
    """
    env = FetchEnvironment(
        state=State(0.0, {F("at r1 shelf"), F("free r1")}, []),
        objects_by_type={
            "robot": {"r1"},
            "location": {"shelf", "counter"},
            "object": {"mug", "sponge"},
        },
        true_object_locations={"shelf": {"mug", "sponge"}, "counter": set()},
    )
    goal = F("at mug counter")
    dashboard = PlannerDashboard(
        goal, env, force_interactive=False, print_on_exit=False,
    )
    with dashboard:
        for name in PLAN:
            env.act(get_action_by_name(env.get_actions(), name))
            dashboard._do_update(env.state, last_action_name=name)
    return dashboard


@pytest.fixture
def location_coords():
    return {"shelf": (2.0, 2.0), "counter": (8.0, 6.0)}

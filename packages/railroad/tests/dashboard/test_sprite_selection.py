"""Which objects earn a sprite.

Searching a location reveals everything truly there, so the set of objects with
recorded positions is much wider than the set the plan is about.
"""

from railroad.core import Fluent as F
from railroad.dashboard._sprites import select_objects


def _entry(time, fluents):
    return {"time": time, "fluents": set(fluents)}


def _positions(*names):
    return {name: [(0.0, "shelf", None)] for name in names}


def test_incidental_revelations_are_excluded():
    """A search reveals a whole receptacle; only the target belongs on the map."""
    selected = select_objects(
        goal_fluents=[F("at mug counter")],
        actions_taken=[("search r1 shelf mug", 0.0)],
        history=[_entry(10.0, [F("at mug shelf"), F("at sponge shelf"), F("at egg shelf")])],
        entity_positions=_positions("mug", "sponge", "egg"),
        known_robots={"r1"},
    )
    assert selected == {"mug"}


def test_held_objects_are_included_without_appearing_in_the_goal():
    """Picking something up is enough on its own to make it plan-relevant."""
    selected = select_objects(
        goal_fluents=[F("at mug counter")],
        actions_taken=[],
        history=[_entry(10.0, [F("holding r1 sponge")])],
        entity_positions=_positions("mug", "sponge"),
        known_robots={"r1"},
    )
    assert selected == {"mug", "sponge"}


def test_action_arguments_are_included():
    selected = select_objects(
        goal_fluents=[],
        actions_taken=[("pick r1 shelf egg", 0.0)],
        history=[],
        entity_positions=_positions("mug", "egg"),
        known_robots={"r1"},
    )
    assert selected == {"egg"}


def test_goal_object_never_found_is_dropped():
    """Nothing to draw: it has no recorded position."""
    selected = select_objects(
        goal_fluents=[F("at mug counter"), F("at ghost counter")],
        actions_taken=[],
        history=[],
        entity_positions=_positions("mug"),
        known_robots={"r1"},
    )
    assert selected == {"mug"}


def test_robots_and_locations_are_never_selected():
    """`at` names the robot first and the location second; neither is an object."""
    selected = select_objects(
        goal_fluents=[F("at r1 counter")],
        actions_taken=[("move r1 shelf counter", 0.0)],
        history=[],
        entity_positions={**_positions("mug"), "r1": [(0.0, "shelf", None)]},
        known_robots={"r1"},
    )
    assert selected == set()


def test_objects_by_type_narrows_the_candidates():
    selected = select_objects(
        goal_fluents=[F("at mug counter")],
        actions_taken=[("pick r1 shelf counter", 0.0)],
        history=[],
        entity_positions=_positions("mug", "counter"),
        known_robots={"r1"},
        objects_by_type={"object": {"mug"}},
    )
    assert selected == {"mug"}


def test_explicit_glyph_objects_override_the_derivation():
    selected = select_objects(
        goal_fluents=[F("at mug counter")],
        actions_taken=[],
        history=[],
        entity_positions=_positions("mug", "sponge"),
        known_robots={"r1"},
        glyph_objects=["sponge", "absent"],
    )
    assert selected == {"sponge"}


def test_empty_dashboard_state_selects_nothing():
    """The existing video fixtures set positions directly and leave history empty."""
    selected = select_objects(
        goal_fluents=[],
        actions_taken=[],
        history=[],
        entity_positions={"r1": [(0.0, "A", None)]},
        known_robots={"r1"},
    )
    assert selected == set()

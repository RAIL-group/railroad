import pytest

from railroad.core import Fluent as F
from railroad.plotting.sprites import select_objects


def positions(*names):
    return {name: [(0.0, "shelf", None)] for name in names}


def select(**changes):
    arguments = {
        "goal_fluents": [],
        "actions_taken": [],
        "history": [],
        "entity_positions": {},
        "known_robots": {"r1"},
    }
    arguments.update(changes)
    return select_objects(**arguments)


def test_incidental_search_results_are_excluded():
    assert select(
        goal_fluents=[F("at mug counter")],
        history=[{"time": 10, "fluents": {F("at mug shelf"), F("at sponge shelf")}}],
        entity_positions=positions("mug", "sponge"),
    ) == {"mug"}


@pytest.mark.parametrize(
    "changes",
    [
        {"history": [{"fluents": {F("holding r1 egg")}}]},
        {"actions_taken": [("pick r1 shelf egg", 0.0)]},
    ],
)
def test_held_or_action_objects_are_included(changes):
    assert select(entity_positions=positions("egg"), **changes) == {"egg"}


def test_unlocated_objects_robots_and_locations_are_excluded():
    assert not select(
        goal_fluents=[F("at ghost counter"), F("at r1 counter")],
        actions_taken=[("move r1 shelf counter", 0.0)],
        entity_positions={"r1": [(0.0, "shelf", None)]},
    )


def test_object_types_narrow_candidates():
    assert select(
        goal_fluents=[F("at mug counter")],
        actions_taken=[("pick r1 shelf counter", 0.0)],
        entity_positions=positions("mug", "counter"),
        objects_by_type={"object": {"mug"}},
    ) == {"mug"}


def test_explicit_objects_override_derivation():
    assert select(
        goal_fluents=[F("at mug counter")],
        entity_positions=positions("mug", "sponge"),
        glyph_objects=["sponge", "absent"],
    ) == {"sponge"}

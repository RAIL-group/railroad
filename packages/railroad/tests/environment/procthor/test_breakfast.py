"""The breakfast maps, the objects some of them add, and the task built on them."""

import pytest

from railroad.core import Fluent as F
from railroad.environment.procthor import ProcTHORScene
from railroad.environment.procthor.breakfast import DISHES, MAPS, BreakfastEnvironment
from railroad.planner import MCTSPlanner


def _contents(scene):
    return {(loc, obj) for loc, objs in scene.object_locations.items() for obj in objs}


def test_added_objects_go_on_a_kitchen_container_and_leave_names_alone():
    plain = ProcTHORScene(seed=7005)
    added = ProcTHORScene(seed=7005, extra_objects=[("countertop", "bread"), ("fridge", "egg")])
    new = _contents(added) - _contents(plain)
    assert _contents(plain) <= _contents(added)
    assert {(loc.split("_")[0], obj.split("_")[0]) for loc, obj in new} == {("countertop", "bread"), ("fridge", "egg")}
    assert plain.locations == added.locations


def test_adding_to_a_missing_container_is_an_error():
    with pytest.raises(ValueError, match="no 'spaceship'"):
        ProcTHORScene(seed=7005, extra_objects=[("spaceship", "egg")])


@pytest.mark.parametrize("map_name", sorted(MAPS))
def test_every_dish_can_be_made_on_every_map(map_name):
    env = BreakfastEnvironment(map_name, num_robots=1, find_prob="oracle")
    for food, _, tool, dish in DISHES:
        assert env.roles[food] and env.roles[tool] and env.roles[dish], (food, tool, dish)
    assert env.destination.startswith("bed_")
    planner = MCTSPlanner(env.get_actions())
    assert 0.0 < planner.heuristic(env.state, env.goal) < float("inf")


def test_a_served_dish_satisfies_the_goal():
    env = BreakfastEnvironment("8614", num_robots=1, find_prob="oracle")
    egg, bowl = sorted(env.roles["egg"])[0], sorted(env.roles["bowl"])[0]
    served = {F(f"boiled {egg}"), F(f"in {egg} {bowl}"), F(f"at {bowl} {env.destination}")}
    assert env.goal.evaluate(served)
    assert not env.goal.evaluate(served - {F(f"boiled {egg}")})

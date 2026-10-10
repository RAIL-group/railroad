"""Breakfast in ProcTHOR: one goal, several ways to reach it.

After the breakfast task of Arnob et al., "Effective Task Planning with Missing
Objects using Learning-Informed Object Search" (arXiv 2602.11468): breakfast is
served at a bed as any one of

- a boiled egg in a bowl (boiling needs a pot or a kettle),
- a peeled apple, tomato or potato on a plate (peeling needs a knife),
- toasted bread on a plate (toasting needs a toaster).

The robots know which objects the house has but not where they are; searching
a place finds each object there with a (learned or oracle) probability. Which
dish, which instance of each object and which robot does what are all left to
the planner.

ProcTHOR's appliances are objects on a countertop or a table rather than
fixtures, and it has no stove, so boiling happens wherever the pot or kettle
is. The maps are ProcTHOR-10k houses in which every dish can be made: three as
they are, and five kitchens completed with the few objects each lacks.
"""

from __future__ import annotations

from functools import reduce
from operator import or_
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from railroad import operators
from railroad._bindings import State
from railroad.core import Effect, Fluent as F, Goal, Operator

from .environment import ProcTHOREnvironment
from .scene import ProcTHORScene

# Each role and the ProcTHOR object types that can fill it.
ROLES: Dict[str, Tuple[str, ...]] = {
    "egg": ("egg",),
    "vessel": ("pot", "kettle"),
    "bowl": ("bowl",),
    "produce": ("apple", "tomato", "potato"),
    "knife": ("knife",),
    "plate": ("plate",),
    "bread": ("bread",),
    "toaster": ("toaster",),
}

# The dishes: (food role, what preparing it makes true, tool role, dish role).
DISHES: Tuple[Tuple[str, str, str, str], ...] = (
    ("egg", "boiled", "vessel", "bowl"),
    ("produce", "peeled", "knife", "plate"),
    ("bread", "toasted", "toaster", "plate"),
)

# Every breakfast object is also an "item": what robots search for, pick up and
# put down. Objects a search reveals that are not items get no actions.
ITEM = "item"

# ProcTHOR-10k houses in which every dish can be made: map name -> (scene
# seed, (container type, object type) pairs to add). The first three need
# nothing added; the rest are kitchens completed with what they lack.
MAPS: Dict[str, Tuple[int, Tuple[Tuple[str, str], ...]]] = {
    "2627": (2627, ()),
    "3594": (3594, ()),
    "3984": (3984, ()),
    "7005": (7005, (("countertop", "bread"), ("countertop", "knife"))),
    "8611": (8611, (("fridge", "egg"), ("diningtable", "plate"), ("countertop", "knife"))),
    "8614": (8614, (("countertop", "bread"), ("countertop", "toaster"))),
    "8615": (8615, (("countertop", "knife"), ("countertop", "toaster"))),
    "1089": (1089, (("countertop", "pot"), ("countertop", "bread"), ("countertop", "toaster"))),
}

SEARCH_TIME = 10.0
PICK_TIME = 10.0
PLACE_TIME = 10.0
PREPARE_TIME = {"boiled": 20.0, "peeled": 15.0, "toasted": 15.0}
PREPARE_NAME = {"boiled": "boil", "peeled": "peel", "toasted": "toast"}
PUT_TIME = 10.0


def construct_prepare_operator(done: str, food_role: str, tool_role: str, duration: float) -> Operator:
    """Prepare a held food with a tool where the tool is (boil, peel, toast)."""
    return Operator(
        name=PREPARE_NAME[done],
        parameters=[("?r", "robot"), ("?loc", "location"), ("?food", food_role), ("?tool", tool_role)],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("holding ?r ?food"), F("at ?tool ?loc"),
                       ~F(f"{done} ?food")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(time=duration, resulting_fluents={F("free ?r"), F(f"{done} ?food")}),
        ],
    )


def construct_put_in_operator(food_role: str, dish_role: str, duration: float) -> Operator:
    """Put a held food into (or onto) a dish where the dish is; it stays there."""
    return Operator(
        name="put-in",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?food", food_role), ("?dish", dish_role)],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("holding ?r ?food"), F("at ?dish ?loc")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not holding ?r ?food")}),
            Effect(time=duration, resulting_fluents={F("free ?r"), F("not hand-full ?r"),
                                                     F("in ?food ?dish")}),
        ],
    )


def role_objects(scene: ProcTHORScene) -> Dict[str, Set[str]]:
    """The scene's objects that can fill each role."""
    roles: Dict[str, Set[str]] = {role: set() for role in ROLES}
    for obj in scene.objects:
        kind = obj.split("_")[0]
        for role, kinds in ROLES.items():
            if kind in kinds:
                roles[role].add(obj)
    return roles


def breakfast_goal(roles: Mapping[str, Set[str]], destination: str) -> Goal:
    """Any one dish served at `destination`: an OR over dishes and instances."""
    options = [
        F(f"{done} {food}") & F(f"in {food} {dish}") & F(f"at {dish} {destination}")
        for food_role, done, _, dish_role in DISHES
        for food in sorted(roles.get(food_role, ()))
        for dish in sorted(roles.get(dish_role, ()))
    ]
    if not options:
        raise ValueError("No dish can be made from these objects")
    return reduce(or_, options)


class BreakfastEnvironment(ProcTHOREnvironment):
    """A breakfast map, robots at the start, breakfast as the goal.

    ``find_prob`` is "learned" (the packaged estimator) or "oracle" (0.8 at the
    object's true place, 0.1 elsewhere). ``goal`` and ``destination`` are set
    once the scene has loaded.
    """

    def __init__(
        self,
        map_name: str,
        num_robots: int,
        find_prob: str = "learned",
        nn_model_path: str | None = None,
    ) -> None:
        if map_name not in MAPS:
            raise ValueError(f"Unknown breakfast map {map_name!r}; choose from {sorted(MAPS)}")
        if find_prob not in ("learned", "oracle"):
            raise ValueError(f"find_prob must be 'learned' or 'oracle', got {find_prob!r}")
        seed, extra = MAPS[map_name]
        self.map_name = map_name
        self.robots = [f"robot{i + 1}" for i in range(num_robots)]
        self._find_prob_kind = find_prob
        self._nn_model_path = nn_model_path
        fluents = {F("revealed start_loc")}
        for robot in self.robots:
            fluents |= {F(f"at {robot} start_loc"), F(f"free {robot}")}
        super().__init__(
            seed=seed,
            state=State(0.0, fluents, []),
            objects_by_type={"robot": set(self.robots), "location": {"start_loc"}},
            extra_objects=extra,
        )
        self.objects_by_type["location"] = set(self.scene.locations)
        self.roles = role_objects(self.scene)
        for role, names in self.roles.items():
            self.objects_by_type[role] = set(names)
        self.objects_by_type[ITEM] = set().union(*self.roles.values())
        beds = sorted(loc for loc in self.scene.locations if loc.split("_")[0] == "bed")
        if not beds:
            raise ValueError(f"Breakfast map {map_name} has no bed to serve it at")
        self.destination = beds[0]
        self.goal = breakfast_goal(self.roles, self.destination)
        self._operators = self.define_operators()
        self.invalidate_grounding()

    def _object_find_prob_fn(self):
        if self._find_prob_kind == "learned":
            if not hasattr(self, "_learned_find_prob_fn"):
                from .learning.utils import get_default_fcnn_model_path
                path = Path(self._nn_model_path) if self._nn_model_path else get_default_fcnn_model_path()
                self._learned_find_prob_fn = self.scene.get_object_find_prob_fn(nn_model_path=str(path))
            return self._learned_find_prob_fn

        truth = {obj: loc for loc, objs in self.scene.object_locations.items() for obj in objs}

        def oracle(robot: str, location: str, obj: str) -> float:
            del robot
            return 0.8 if truth.get(obj) == location else 0.1

        return oracle

    def define_operators(self) -> List[Operator]:
        ops = [
            operators.construct_no_op_operator(no_op_time=5.0),
            operators.construct_move_operator_blocking(self.estimate_move_time),
            operators.construct_search_operator(self._object_find_prob_fn(), SEARCH_TIME, object_type=ITEM),
            operators.construct_pick_operator_blocking(PICK_TIME, object_type=ITEM),
            operators.construct_place_operator_blocking(PLACE_TIME, object_type=ITEM),
        ]
        for food_role, done, tool_role, dish_role in DISHES:
            ops.append(construct_prepare_operator(done, food_role, tool_role, PREPARE_TIME[done]))
            ops.append(construct_put_in_operator(food_role, dish_role, PUT_TIME))
        return ops


def map_summary(env: BreakfastEnvironment) -> Dict[str, Sequence[str]]:
    """Where each breakfast object really is, for logs and plots."""
    truth = {obj: loc for loc, objs in env.scene.object_locations.items() for obj in objs}
    return {role: sorted(f"{obj}@{truth[obj]}" for obj in names) for role, names in env.roles.items()}

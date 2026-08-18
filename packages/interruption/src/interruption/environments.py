from collections import defaultdict
from typing import Sequence
import random
import numpy as np
from railroad import operators
from railroad.core import Action, Goal, LiteralGoal, Operator, State
from railroad.core import Fluent as F
from railroad.environment import SymbolicEnvironment
from railroad.environment.procthor.environment import ProcTHOREnvironment

from .operators import (
    construct_assemble_operator,
    construct_gripper_pick_operator,
    construct_gripper_place_operator,
)
from .alfred_task_generator import get_task_list
from .utilities import get_updated_scene_graph


class KitchenProcTHOREnvironment(ProcTHOREnvironment):
    """
    Kitchen ProcTHOR environment with relevant internal operator construction.
    """
    def define_operators(self) -> list[Operator]:
        move_op = operators.construct_move_operator(self.estimate_move_time)
        pick_op = construct_gripper_pick_operator(20.0)
        place_op = construct_gripper_place_operator(20.0)
        # previously extra_cost was set to 100.0
        # no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=0)
        return [pick_op, place_op, move_op]


    def update_scene_graph(self, action: Action) -> None:
        """
        Method for updating the scene graph to match the environment's
        state after taking the robot takes an action.
        Note: this method assumes a single-robot scenario.
        """
        get_updated_scene_graph(self.scene.scene_graph, self.state, action)


# helper functions
def construct_simple_kitchen_environment() -> SymbolicEnvironment:
    """
    Constructs a SymbolicEnvironment representing the simple kitchen
    environment used for prototyping.
    """
    locations = {
        "refrigerator": np.array([0, 0]),
        "pantry": np.array([1, 0]),
        "countertop1": np.array([1,1]),
        "countertop2": np.array([2,1]),
        "table": np.array([0,2])
    }

    objects_by_type = {
        "robot": {"robot1"},
        "location": set(locations),
        "object": {"turkey", "bread", "sandwhich"}
    }

    pick_time = 1
    place_time =1
    assemble_time = 1

    # define operators
    def move_time(robot, loc_from, loc_to):
        return float(np.linalg.norm(locations[loc_from] - locations[loc_to]))

    move = operators.construct_move_operator(move_time)
    pick = operators.construct_pick_operator(pick_time)
    place = operators.construct_place_operator(place_time)
    assemble = construct_assemble_operator(assemble_time)

    # initial state
    initial_fluents = {
        F("free robot1"), F("at robot1 table"), F("is-turkey turkey"), F("is-bread bread"),
        ~F("hand-full robot1"), F("at turkey refrigerator"), F("at bread pantry"),
        ~F("prep-station table"), F("prep-station countertop2"), ~F("prep-station refrigerator"),
        ~F("sandwhich-made"), F("prep-station countertop1"), F("is-sandwhich sandwhich"),
        ~F("prep-station pantry")
    }
    initial_state = State(0.0, initial_fluents)

    env = SymbolicEnvironment(
        state=initial_state, objects_by_type=objects_by_type,
        operators=[move, pick, place, assemble],
    )
    return env


def get_simple_task_distribution() -> list[tuple[list[Goal], list[float]]]:
    """
    Returns the task distributions for the prototype kitchen scenario.
    """
    interrupting_task_dists = [
        (
            [
                (
                    ~F("at turkey countertop1") & ~F("at bread countertop1") &
                    ~F("hand-full robot1") & ~F("at sandwhich countertop1")
                )
            ],
            [1.0]
        ),
        (
        [
            (
                ~F("at turkey countertop2") & ~F("at bread countertop2") &
                ~F("hand-full robot1") & ~F("at sandwhich countertop2")
            )
        ],
        [1.0]
        )
    ]

    return interrupting_task_dists


def get_simple_goal() -> F | Goal:
    """
    Gets the initial goal for the prototype kitchen scenario.
    """
    return (F("sandwhich-made") & F("at sandwhich table"))


def construct_procthor_kitchen_environment(
    seed: int, object_seed: int | None = None
) -> KitchenProcTHOREnvironment:
    """
    Constructs a KitchenProcTHOREnvironment representing the scene 
    corresponding to the scene from ProcTHOR-10k.
    """
    initial_fluents = {
        F("at robot1 start_loc"), F("free robot1"),
        F("gripper-of r1-left robot1"), F("gripper-of r1-right robot1"),
        ~F("hand-full r1-left"), ~F("hand-full r1-right")
    }
    initial_state = State(0.0, initial_fluents)

    env = KitchenProcTHOREnvironment(
        seed,
        initial_state,
        {
            "robot": {"robot1"},
            "gripper": {"r1-left", "r1-right"},
            "location": {"start_loc"},
        },
        object_seed
    )

    # Fully populate symbolic environment now that scene is available internally.
    env.objects_by_type["location"] = set(env.scene.locations.keys())
    env.objects_by_type["object"] = env.scene.objects
    setup_procthor_initial_state(env.fluents, env.scene.object_locations)
    return env


def setup_procthor_initial_state(
    initial_fluents: set[F],
    objects_by_location: dict[str, set]
) -> None:
    """
    Helper function that adds fluents related to the location of 
    objects (e.g., at ?obj ?loc) to the initial fluents of a
    KitchenProcTHOREnvironment.
    """
    object_location_fluents = [
        F(f"at {obj} {loc}")
        for loc in objects_by_location
        for obj in objects_by_location[loc]
    ]
    return initial_fluents.update(object_location_fluents)


def get_example_procthor_task_distribution(index: int) -> tuple[Sequence[Goal], list[float]]:
    """
    Returns the task distribution for an example ProcTHOR kitchen scenario
    (seed=201).
    """
    tasks_dists = [
        [LiteralGoal(F("at spraybottle garbagecan"))],
        [
            LiteralGoal(F("at spraybottle garbagecan")),
            LiteralGoal(F("at spraybottle fridge")),
            LiteralGoal(F("at apple stool")),
            LiteralGoal(F("at apple shelvingunit")),
            LiteralGoal(F("at spoon countertop")),
            LiteralGoal(F("at spoon fridge")),
            LiteralGoal(F("at peppershaker garbagecan")),
            LiteralGoal(F("at peppershaker stool")),
            LiteralGoal(F("at knife countertop")),
            LiteralGoal(F("at knife garbagecan"))
        ]
    ]
    probs = [[1/len(dist)] * len(dist) for dist in tasks_dists]

    return list(zip(tasks_dists, probs))[index]


def get_alfred_task_distribution(
    scene_objects: set[str],
    scene_locations: set[str],
    size: int = 10,
    seed: int = 63,
    one_object_per_taskdist: bool = True
) -> tuple[Sequence[Goal], list[float]]:
    """
    Returns a task distribution of tasks from the ALFRED dataset 
    of a specified size for a ProcTHOR scene.
    """
    rng = random.Random(seed)

    task_list = get_task_list(
        {loc.split("_")[0] for loc in scene_locations},
        {loc.split("_")[0] for loc in scene_objects}
    )

    # each object will only be contained in one task of the task distribution
    if one_object_per_taskdist:
        task_dict = defaultdict(list)
        for obj, location in task_list:
            task_dict[obj].append(location)

        # sample from the task dict
        goals = [
            LiteralGoal(F(f"at {object_key} {rng.choice(possible_locations)}"))
            for object_key, possible_locations in task_dict.items()
        ]
    else:
        goals = [LiteralGoal(F(f"at {obj} {loc}")) for obj, loc in rng.sample(task_list, size)]
    probs = [1/len(goals)] * len(goals)

    return goals, probs


def get_example_procthor_goal() -> Goal:
    """
    Gets the initial goal for an example ProcTHOR kitchen scenario (seed=201).
    """
    return LiteralGoal(F("at knife shelvingunit"))

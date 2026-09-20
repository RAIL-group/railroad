from collections import defaultdict
from typing import Sequence, Optional
import copy
import random
import numpy as np
from railroad import operators
from railroad.core import Action, Goal, LiteralGoal, Operator, State
from railroad.core import Fluent as F
from railroad.environment import SymbolicEnvironment
from railroad.environment.procthor.environment import ProcTHOREnvironment

from .operators import (
    construct_assemble_operator,
    # construct_gripper_pick_operator,
    # construct_gripper_place_operator,
    construct_pick_with_left_hand_operator,
    construct_pick_with_right_hand_operator,
    construct_place_with_left_hand_operator,
    construct_place_with_right_hand_operator,
)
from .alfred_task_generator import get_task_list
from .utilities import (
    get_object_container_slots,
    get_updated_scene_graph,
    permute_object_containers,
)


class KitchenProcTHOREnvironment(ProcTHOREnvironment):
    """
    Kitchen ProcTHOR environment with relevant internal operator construction.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # the scene's original container occupancy, before any action has
        # moved an object - what randomize_object_locations resets to
        self._original_container_slots = get_object_container_slots(self.scene.scene_graph)

    def fork(self) -> "KitchenProcTHOREnvironment":
        """
        An independent copy of this environment in its current state, to be
        advanced without touching this one. Everything a step mutates
        (fluents, time, skills, RNG, scene graph, ground-truth maps) is
        copied. The per-scene data no step touches is shared: the scene's
        cached data, occupancy grid, path caches and the action grounding
        (which depends only on the object universe and static facts). That
        sharing is what makes a fork ~1 ms instead of ~60 ms.
        """
        thor = self.scene._thor
        shared = [
            thor.cached_data, thor.g2p_map, thor.occupancy_grid, thor.scene,
            getattr(self, "_cost_grid_cache", None),  # created lazily on first path query
            self._grounding_cache,
        ]
        clone = copy.deepcopy(self, {id(obj): obj for obj in shared})
        # deepcopy leaves the copied operators bound to *this* environment's
        # methods; rebind them, or every fork would keep its ancestors alive
        clone._operators = clone.define_operators()
        return clone

    def define_operators(self) -> list[Operator]:
        move_op = operators.construct_move_operator(self.estimate_move_time)
        left_pick_op = construct_pick_with_left_hand_operator(10.0)
        right_pick_op = construct_pick_with_right_hand_operator(10.0)
        left_place_op = construct_place_with_left_hand_operator(10.0)
        right_place_op = construct_place_with_right_hand_operator(10.0)
        return [move_op, left_pick_op, right_pick_op, left_place_op, right_place_op]


    def update_scene_graph(self, action: Action) -> None:
        """
        Method for updating the scene graph to match the environment's
        state after taking the robot takes an action.
        Note: this method assumes a single-robot scenario.
        """
        get_updated_scene_graph(self.scene.scene_graph, self.state, action)

    def randomize_object_locations(self, rng: random.Random) -> None:
        """
        Re-deals the objects over the scene's original container occupancy
        (see permute_object_containers), so a scene whose tasks have all been
        completed is reset to a fresh arrangement with the original
        per-container counts. Keeps every record of placement consistent: the
        scene graph, the `at obj loc` fluents, and both ground-truth maps (the
        scene's and this environment's own copy), which are rebuilt from the
        graph. Objects the robot holds are part of the shuffle (see
        permute_object_containers). Requires no skill in flight.
        """
        if self.state.upcoming_effects:
            raise RuntimeError("cannot randomize object locations while skills are in flight")

        graph = self.scene.scene_graph
        moves = permute_object_containers(graph, self._original_container_slots, rng)
        robot_idx = graph.robot_indices[0]
        # the environment's name for the robot, which fluents use - not the
        # graph node's name, which need not match it
        (robot,) = self.objects_by_type["robot"]

        def name(idx: int) -> str:
            return f"{graph.get_node_name_by_idx(idx)}_{idx}"

        # The same number of objects stay held, so no hand empties or fills: a
        # hand that lets go of one object takes the object that was grasped
        # instead (the two lists are the same length).
        released = sorted(o for o, (old, _) in moves.items() if old == robot_idx)
        grasped = sorted(o for o, (_, new) in moves.items() if new == robot_idx)
        for released_idx, grasped_idx in zip(released, grasped):
            hand = (
                "left" if F(f"holding-in-left {robot} {name(released_idx)}") in self.fluents
                else "right"
            )
            self.fluents.remove(F(f"holding-in-{hand} {robot} {name(released_idx)}"))
            self.fluents.add(F(f"holding-in-{hand} {robot} {name(grasped_idx)}"))

        for obj_idx, (old_idx, new_idx) in moves.items():
            if old_idx != robot_idx:
                self.fluents.discard(F(f"at {name(obj_idx)} {name(old_idx)}"))
            if new_idx != robot_idx:
                self.fluents.add(F(f"at {name(obj_idx)} {name(new_idx)}"))

        # The ground-truth maps are fixed at construction and never follow
        # pick/place, so they are stale by now: rebuild them from the graph,
        # which is the live record, rather than patching them.
        self.scene.refresh_object_locations()
        self._objects_at_locations = {
            loc: set(objs) for loc, objs in self.scene.object_locations.items()
        }


    def apply_time_penalty(self, time: float) -> None:
        """
        Method for moving the environment's time forward due to incurring
        a penalty cost. (Ex. If the planner failed to find a solution to a task,
        a failure penalty is applied.)
        """
        self._time += time


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
    seed: int,
    object_seed: int | None = None,
    relevant_objects: Optional[list[str]] = None,
    remove_duplicates: bool = False
) -> KitchenProcTHOREnvironment:
    """
    Constructs a KitchenProcTHOREnvironment representing the scene 
    corresponding to the scene from ProcTHOR-10k.
    """
    initial_fluents = {
        F("at robot1 start_loc"), F("free robot1"),
        ~F("left-hand-full robot1"), ~F("right-hand-full robot1")
    }
    initial_state = State(0.0, initial_fluents)

    env = KitchenProcTHOREnvironment(
        seed,
        initial_state,
        {
            "robot": {"robot1"},
            "location": {"start_loc"},
        },
        object_seed,
        relevant_objects,
        remove_duplicates
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
        goals = goals[:size]
    else:
        goals = [LiteralGoal(F(f"at {obj} {loc}")) for obj, loc in rng.sample(task_list, size)]
    probs = [1/len(goals)] * len(goals)

    return goals, probs


def get_example_procthor_goal() -> Goal:
    """
    Gets the initial goal for an example ProcTHOR kitchen scenario (seed=201).
    """
    return LiteralGoal(F("at knife shelvingunit"))

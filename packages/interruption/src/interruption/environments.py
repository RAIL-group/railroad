from collections import defaultdict
from collections.abc import Callable
from typing import Sequence, Optional
import copy
import random
from pathlib import Path
from railroad import operators
from railroad.core import Action, Goal, LiteralGoal, Operator, State
from railroad.core import Fluent as F
from railroad.environment import Environment
from railroad.environment.procthor.environment import ProcTHOREnvironment

from .operators import (
    construct_pick_with_left_hand_operator_blocking,
    construct_place_with_left_hand_operator_blocking,
    only_clears_blocking_facts,
)
from .alfred_task_generator import get_task_list
from .constants import NUM_ROOMS_FILTER, NUM_TASKS, ONE_ROOM_FILTER
from .procthor_scenes import get_task_distribution_from_remap_dir
from .scene_graph import (
    get_object_container_slots, get_updated_scene_graph, permute_object_containers,
)


class FinishesBlockingCleanUps(Environment):
    """
    Environment mixin: an action blocked only by a blocking fact whose clean-up
    is still pending (see operators.BLOCKING_PREDICATES) first lets that clean-up
    happen. List it before the concrete Environment subclass.
    """

    def act(self, action: Action, loop_callback_fn: Callable[[], None] | None = None) -> State:
        """
        Environment.act, except that an action blocked only by a blocking fact
        whose clean-up is still pending (see operators.BLOCKING_PREDICATES) first
        lets that clean-up happen, at most BLOCKING_DURATION later. Blocking keeps
        a plan from moving twice in a row, but a new plan starts without blocking
        facts and may begin with a move straight after the last plan's.
        """
        if not self.state.satisfies_precondition(action) and self._only_blocking_clean_ups_pending():
            self._finish_pending_effects()
        return super().act(action, loop_callback_fn)

    def _only_blocking_clean_ups_pending(self) -> bool:
        upcoming = self.state.upcoming_effects
        return bool(upcoming) and all(
            only_clears_blocking_facts(effect) for _, effect in upcoming
        )

    def _finish_pending_effects(self) -> None:
        """Advances time until every in-flight skill has applied all its effects."""
        while self._active_skills:
            next_time = min(skill.time_to_next_event for skill in self._active_skills)
            if next_time == float("inf"):
                break
            for skill in self._active_skills:
                skill.advance(next_time, self)
            self._time = next_time
            self._active_skills = [skill for skill in self._active_skills if not skill.is_done]
            self._after_skills_advanced(next_time)


def kitchen_operators(env: "KitchenProcTHOREnvironment") -> list[Operator]:
    """
    The kitchen's default operators. Blocking variants: no move straight after a
    move, no placing an object straight after picking it, no picking it straight
    back up after placing it. Moves take as long as the scene's path between the
    locations; picks and places take 10.
    """
    move_op = operators.construct_move_operator_blocking(env.estimate_move_time)
    left_pick_op = construct_pick_with_left_hand_operator_blocking(10.0)
    left_place_op = construct_place_with_left_hand_operator_blocking(10.0)
    return [move_op, left_pick_op, left_place_op]


# builds an environment's operators; given the environment, so an operator can
# use its scene (e.g. kitchen_operators' moves, timed by env.estimate_move_time)
OperatorFactory = Callable[["KitchenProcTHOREnvironment"], list[Operator]]


class KitchenProcTHOREnvironment(FinishesBlockingCleanUps, ProcTHOREnvironment):
    """
    Kitchen ProcTHOR environment. Its operators come from the operators factory,
    kitchen_operators unless given, e.g. to add one:

        KitchenProcTHOREnvironment(..., operators=lambda env: [
            *kitchen_operators(env), construct_no_op_operator(5.0)
        ])
    """
    def __init__(self, *args, operators: OperatorFactory = kitchen_operators, **kwargs) -> None:
        # define_operators runs during Environment.__init__, so set this first
        self._operator_factory = operators
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
        return self._operator_factory(self)


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
        # a blocking fact's pending clean-up is all that remains after an action;
        # anything else means an action is still under way
        if not all(
            only_clears_blocking_facts(effect) for _, effect in self.state.upcoming_effects
        ):
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

        # discard everything first: an object can move into a category another
        # moved object is leaving
        for obj_idx, (old_idx, new_idx) in moves.items():
            if old_idx != robot_idx:
                self.fluents.discard(F(f"at {name(obj_idx)} {name(old_idx)}"))
                self.fluents.discard(
                    F(f"at-category {name(obj_idx)} {location_category(name(old_idx))}")
                )
        for obj_idx, (old_idx, new_idx) in moves.items():
            if new_idx != robot_idx:
                self.fluents.add(F(f"at {name(obj_idx)} {name(new_idx)}"))
                self.fluents.add(
                    F(f"at-category {name(obj_idx)} {location_category(name(new_idx))}")
                )

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


def construct_procthor_kitchen_environment(
    seed: int,
    object_seed: int | None = None,
    relevant_objects: Optional[list[str]] = None,
    remove_duplicate_objects: bool = False,
    operators: OperatorFactory = kitchen_operators,
) -> KitchenProcTHOREnvironment:
    """
    Constructs a KitchenProcTHOREnvironment representing the scene 
    corresponding to the scene from ProcTHOR-10k. remove_duplicate_objects keeps
    one object per object category; duplicate containers are always kept, since
    they are part of the home's real layout. operators builds the environment's
    operators (see KitchenProcTHOREnvironment).
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
        remove_duplicate_objects=remove_duplicate_objects,
        operators=operators,
    )

    # Fully populate symbolic environment now that scene is available internally.
    env.objects_by_type["location"] = set(env.scene.locations.keys())
    env.objects_by_type["object"] = env.scene.objects
    env.objects_by_type["category"] = {
        location_category(loc) for loc in env.objects_by_type["location"]
    }
    setup_procthor_initial_state(
        env.fluents, env.scene.object_locations, env.objects_by_type["location"]
    )
    return env


def location_category(location: str) -> str:
    """The category of a scene location: diningtable_6 -> diningtable."""
    return location.split("_")[0]


def setup_procthor_initial_state(
    initial_fluents: set[F],
    objects_by_location: dict[str, set],
    locations: set[str],
) -> None:
    """
    Helper function that adds fluents related to the location of 
    objects (e.g., at ?obj ?loc) to the initial fluents of a
    KitchenProcTHOREnvironment: each location's category (category ?loc ?cat),
    and each object's location and the category it is in (at-category ?obj ?cat).
    """
    initial_fluents.update(F(f"category {loc} {location_category(loc)}") for loc in locations)
    initial_fluents.update(
        fluent
        for loc in objects_by_location
        for obj in objects_by_location[loc]
        for fluent in (F(f"at {obj} {loc}"), F(f"at-category {obj} {location_category(loc)}"))
    )


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


def get_scene_task_distribution(
    procthor_seed: int,
    remap_dir: Path | None = None,
    scene_filter: dict[str, int | set[str]] = ONE_ROOM_FILTER,
) -> tuple[Sequence[Goal], list[float]]:
    """
    The task distribution to evaluate a ProcTHOR scene on. With remap_dir (see
    use_remapped_scenes), the one the multi-scene data-generation run that made
    it used, given the scene_filter that run used. Otherwise built from the
    plain scene's own objects and locations.
    """
    if remap_dir is not None:
        return get_task_distribution_from_remap_dir(remap_dir, scene_filter, NUM_ROOMS_FILTER)
    env = construct_procthor_kitchen_environment(procthor_seed, remove_duplicate_objects=True)
    return get_alfred_task_distribution(
        env.scene.objects,
        set(env.scene.locations),
        size=NUM_TASKS,
        one_object_per_taskdist=True,
    )


def get_example_procthor_goal() -> Goal:
    """
    Gets the initial goal for an example ProcTHOR kitchen scenario (seed=201).
    """
    return LiteralGoal(F("at knife shelvingunit"))

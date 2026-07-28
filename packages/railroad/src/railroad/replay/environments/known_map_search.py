"""Known-map object-search replay.

The complement of :mod:`~railroad.replay.environments.unknown_search`: there the
map is *unknown* (frontier exploration + confinement sensing); here the map is
**fully known** (e.g. a ProcTHOR floorplan) and only the objects' *presence in
containers* is unrecorded.

Because the map is known, **travel is exact** — there is no unseen-as-free
optimism in the travel term, and (no unobserved space) no frontier exploration.
Object *presence*, though, is known only where the deployment actually
**searched**. We do **not** assume an object lives in a single container, so a
revealed-but-unsearched container's emptiness cannot be inferred from the target
being found elsewhere: searching such a container forces not-found and logs an
optimistic commit (``optimistic_to_goal = 0`` — the object could be right here;
with no unobserved space there is no frontier term). So the reported cost is a
**commit-based lower bound** (``optimistic_lb`` vs. makespan), collapsing onto
the exact makespan only when the deployment searched every container.

The key simplification: :class:`~railroad.environment.object_search.ObjectSearchEnvironment`
already resolves a ``search`` deterministically from ``_objects_at_locations``.
So replay needs no bespoke resolution intercept — it restricts that map to the
**recorded** contents and lets the existing resolution run, adding only
commit/search-log bookkeeping. A container the deployment never searched has
empty recorded contents → ``found`` resolves false there (correct: its true
contents were never observed, so replay must not rely on them).
"""

from __future__ import annotations

from typing import Collection, Dict, List, Set, Tuple

import numpy as np

from railroad import operators as _operators
from railroad._bindings import Fluent, Goal, GroundedEffect, State
from railroad.core import Operator
from railroad.environment.object_search import ObjectSearchEnvironment
from railroad.environment.symbolic import LocationRegistry
from railroad.experimental.unknown_search import FixedObjectFind, ObjectFindEstimator
from railroad.navigation import OccupancyGridPathingMixin

from ..cost import Commit
from ..loop import MctsConfig
from ..recorder import DEFAULT_SEARCH_TIME, DEFAULT_SPEED, START_NAME
from ..types import RolloutLog
from .base import (
    ReplayArenaMixin,
    objects_in_goal,
    require_goal,
    robot_from_free,
)


class ReplayKnownMapSearchEnvironment(
    ReplayArenaMixin, OccupancyGridPathingMixin, ObjectSearchEnvironment
):
    """Known-map object-search replay: ``move`` + ``search`` over a known grid.

    Search outcomes resolve from *recorded_object_locations* (the deployment's
    revealed contents) via the inherited deterministic resolution. The candidate
    policy's container belief only drives MCTS belief, never the outcome; it is
    read through ``self._object_find_statistics`` so
    :func:`~railroad.replay.driver.run_replay` can swap policies on a reused arena.
    """

    default_mcts = MctsConfig(iterations=4000, c=300.0, max_depth=20, heuristic_multiplier=2.0)
    default_max_planning_iterations = 60
    dashboard_fluent_keywords = ("at", "found", "searched")

    def __init__(
        self,
        *,
        known_grid: np.ndarray,
        recorded_object_locations: Dict[str, Set[str]],
        goal: "Goal | Fluent",
        state: State,
        objects_by_type: Dict[str, Set[str]],
        location_registry: LocationRegistry,
        searched_sites: Collection[str] = (),
        search_time: float = DEFAULT_SEARCH_TIME,
        speed_cells_per_sec: float = DEFAULT_SPEED,
    ) -> None:
        # Neutral until apply_policy() installs a candidate.
        self._object_find_statistics: ObjectFindEstimator = FixedObjectFind()
        self._known_grid = np.asarray(known_grid, dtype=float).copy()
        self._speed = float(speed_cells_per_sec)
        self._search_time = float(search_time)
        self._goal = goal
        self._search_log: List[Tuple[str, float, bool]] = []
        # Containers the deployment searched (outcome known → replay exact).
        self._searched_sites = set(searched_sites)
        # One commit per not-found search at an UNsearched container.
        self._replay_commits: List[Commit] = []
        # Bootstrap pathing state (used by estimate_move_time in the operators
        # below) before super().__init__.
        self._location_registry = location_registry

        super().__init__(
            state=state,
            objects_by_type=objects_by_type,
            operators=self._build_operators(search_time),
            true_object_locations=recorded_object_locations,
            location_registry=location_registry,
        )

    def _build_operators(self, search_time: float) -> List[Operator]:
        # search reads the estimator through self._object_find_statistics, so
        # the candidate can be swapped without rebuilding the arena.
        return [
            _operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0),
            _operators.construct_move_operator_blocking(self.estimate_move_time),
            _operators.construct_search_operator(self._container_find_prob, search_time),
        ]

    def apply_policy(self, policy: ObjectFindEstimator) -> None:
        """Install the object-search belief.

        Only ``container_probability`` is ever read: a known map has no
        frontiers, so this flavor consumes the narrowest slice of the three.
        """
        self._object_find_statistics = policy

    def _container_find_prob(self, robot: str, loc: str, obj: str) -> float:
        return self._object_find_statistics.container_probability(robot, loc, obj)

    @classmethod
    def from_log(
        cls,
        log: RolloutLog,
        *,
        config=None,  # nav-config override is not meaningful for a known map
    ) -> "ReplayKnownMapSearchEnvironment":
        """Build a policy-agnostic known-map search replay arena from a *log*.

        Searchable locations (coordinates) and recorded contents come from the
        log's subgoals; the searchable objects are the ones the recorded goal
        names; travel speed and search time come from ``log.config``.
        """
        del config
        goal = require_goal(log)
        robots = log.robots
        start = log.robot_starts[robots[0]]
        coords: Dict[str, np.ndarray] = {START_NAME: np.array(start[:2], dtype=float)}
        recorded: Dict[str, Set[str]] = {}
        searched_sites: Set[str] = set()
        for s in log.subgoals:
            coords[s.signature] = np.array(s.centroid, dtype=float)
            recorded[s.signature] = set(s.contents)
            if s.searched:
                searched_sites.add(s.signature)

        fluents: Set[Fluent] = {Fluent(f"revealed {START_NAME}")}
        for robot in robots:
            fluents |= {Fluent(f"at {robot} {START_NAME}"), Fluent(f"free {robot}")}

        # The objects to search for are exactly those the goal names (minus the
        # robots and locations), so the search operator grounds over them.
        objects = objects_in_goal(goal, exclude=set(robots) | set(coords))

        return cls(
            known_grid=log.recorded_grid,
            recorded_object_locations=recorded,
            goal=goal,
            state=State(0.0, fluents, []),
            objects_by_type={
                "robot": set(robots),
                "location": {START_NAME} | set(coords) | set(recorded),
                "object": objects,
            },
            location_registry=LocationRegistry(coords),
            searched_sites=searched_sites,
            search_time=float(log.config.get("search_time", DEFAULT_SEARCH_TIME)),
            speed_cells_per_sec=float(log.config.get("speed_cells_per_sec", DEFAULT_SPEED)),
        )

    # -- goal / pathing hooks -----------------------------------------

    @property
    def goal(self) -> "Goal | Fluent":
        return self._goal

    @property
    def occupancy_grid(self) -> np.ndarray:
        return self._known_grid

    @property
    def _pathing_unknown_as_obstacle(self) -> bool:
        return False  # the map is fully known free/occupied space

    @property
    def _pathing_speed_cells_per_sec(self) -> float:
        return self._speed

    @property
    def search_log(self) -> List[Tuple[str, float, bool]]:
        """(location, sim_time, found) per executed search."""
        return self._search_log

    @property
    def replay_commits(self) -> List[Commit]:
        """One commit per not-found search at an unsearched container."""
        return self._replay_commits

    # -- record search provenance + optimistic commits -------------------

    def resolve_probabilistic_effect(
        self, effect: GroundedEffect, current_fluents: Set[Fluent]
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        effects, fluents = super().resolve_probabilistic_effect(effect, current_fluents)
        if effect.is_probabilistic:
            loc = _search_location(effect)
            if loc is not None:
                accrued = float(self._time)
                found = any(
                    f.name == "found" and not f.negated
                    for eff in effects
                    for f in eff.resulting_fluents
                )
                self._search_log.append((loc, accrued, found))
                # Commit only at a container the deployment did NOT search: its
                # contents are unknown, so replay forces not-found and this is an
                # unverified subgoal (object could be here → optimistic_to_goal=0).
                # A searched container's outcome is known → replay exact, no commit.
                if not found and loc not in self._searched_sites:
                    self._replay_commits.append(
                        Commit(
                            cost_accrued=accrued,
                            optimistic_to_goal=0.0,
                            robot=robot_from_free(effects),
                            frontier_signature=loc,
                        )
                    )
        return effects, fluents


def _search_location(effect: GroundedEffect) -> str | None:
    """The location of a ``search`` prob-effect (from its success branch)."""
    for _, branch in effect.prob_effects:
        for eff in branch:
            for f in eff.resulting_fluents:
                if f.name == "at" and not f.negated and len(f.args) >= 2:
                    return f.args[1]
    return None

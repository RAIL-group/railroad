"""Object-search replay over a recorded unknown-map deployment.

Same idea as the navigation :mod:`~railroad.replay.environments.point_goal_nav`
but for **frontier-based object search** (``examples/frontier_search.py``): the
robot explores frontiers and searches containers for a target object. Replay
confines the robot to the recorded map (shared :class:`ReplayConfinementMixin`)
and resolves every search outcome from the **recorded ground truth** — so an
alternative search policy's cost is reproduced without redeploying it.

The find-probabilities that drive planner belief are read through the current
candidate policy (``self._policy``), so :func:`~railroad.replay.driver.run_replay`
can swap policies on a reused arena. The frontier find-probability
("is the object beyond this frontier?") is the learned, served-vantage knob: it
is driven by a :class:`~railroad.lsp.frontier_statistics.LearnedFrontierStatistics`
fed the recorded panoramas (best-vantage per frontier); a trained network drops
in at the same call site as in navigation (see :mod:`railroad.replay.stub_model`).
"""

from __future__ import annotations

from typing import Callable, Collection, Dict, List, Sequence, Set, Tuple

import numpy as np

from railroad._bindings import Fluent, Goal, GroundedEffect, State
from railroad.environment.symbolic import LocationRegistry
from railroad.environment.types import Pose
from railroad.experimental.unknown_search import (
    NavigationConfig,
    UnknownSpaceEnvironment,
)
from railroad.experimental.unknown_search.operators import (
    construct_move_navigable_operator,
    construct_search_at_site_operator,
    construct_search_frontier_operator,
)
from railroad.lsp.frontier_statistics import LearnedFrontierStatistics
from railroad.operators import construct_no_op_operator

from ..cost import Commit
from ..loop import MctsConfig
from ..policy import ProbFn
from ..types import RolloutLog
from .base import (
    ReplayArenaMixin,
    ReplayConfinementMixin,
    navigation_config_from_log,
    objects_in_goal,
    require_goal,
    robot_from_free,
)

START_NAME = "start_loc"
DEFAULT_SEARCH_TIME = 20.0


class ReplayUnknownSearchEnvironment(
    ReplayArenaMixin, ReplayConfinementMixin, UnknownSpaceEnvironment
):
    """Replay an object-search policy over a recorded unknown-map deployment."""

    default_mcts = MctsConfig(iterations=4000, c=300.0, max_depth=20, heuristic_multiplier=2.0)
    default_max_planning_iterations = 80
    dashboard_fluent_keywords = ("at", "found", "searched")

    def __init__(
        self,
        *,
        recorded_grid: np.ndarray,
        recorded_object_locations: Dict[str, Set[str]],
        reference_cell: Tuple[int, int],
        goal: "Goal | Fluent",
        hidden_sites: Dict[str, Tuple[int, int]],
        state: State,
        objects_by_type: Dict[str, Set[str]],
        robot_initial_poses: Dict[str, Pose],
        location_registry: LocationRegistry,
        skill_overrides: Dict[str, type],
        config: NavigationConfig | None = None,
        searched_sites: Collection[str] = (),
        pano_records: Sequence = (),
        search_time: float = DEFAULT_SEARCH_TIME,
    ) -> None:
        confinement = self._setup_replay_grids(recorded_grid, pano_records)
        # Neutral policy until apply_policy() swaps in a candidate.
        self._init_policy()

        self._recorded_object_locations = recorded_object_locations
        # Containers the deployment actually searched: their outcome is known, so
        # replay resolves them exactly (no optimistic commit). A revealed-but-
        # unsearched container is unknown — we must not infer it empty from the
        # object being found elsewhere (that assumes one container per object).
        self._searched_sites = set(searched_sites)
        self._reference_cell = (int(reference_cell[0]), int(reference_cell[1]))
        self._goal = goal
        self._search_time = float(search_time)
        # (loc, cost_accrued, found) per executed search.
        self._search_log: List[Tuple[str, float, bool]] = []
        # One commit per not-found search at a subgoal the deployment did NOT
        # verify (unsearched container or frontier). optimistic_lb = min over these.
        self._replay_commits: List[Commit] = []

        super().__init__(
            state=state,
            objects_by_type=objects_by_type,
            operators=self._build_operators(search_time),
            true_grid=confinement,
            true_object_locations=recorded_object_locations,
            robot_initial_poses=robot_initial_poses,
            location_registry=location_registry,
            skill_overrides=skill_overrides,
            hidden_sites=hidden_sites,
            config=config or NavigationConfig(),
        )
        for robot in self._objects_by_type.get("robot", set()):
            self._net_motion.setdefault(robot, 0.0)

    def _build_operators(self, search_time: float) -> list:
        # Operators read the find-probabilities through self._policy, so the
        # candidate can be swapped without rebuilding the arena. move-time is the
        # env's own safe navigable estimate (self exists by the time it's called).
        return [
            construct_move_navigable_operator(self._move_time),
            construct_search_frontier_operator(
                object_find_prob=self._frontier_find_prob, search_time=search_time
            ),
            construct_search_at_site_operator(
                self._container_find_prob,
                search_time=search_time,
                container_type="container",
            ),
            construct_no_op_operator(no_op_time=300.0, extra_cost=100.0),
        ]

    def _move_time(self, robot: str, loc_from: str, loc_to: str) -> float:
        return self.estimate_move_time_safe(robot, loc_from, loc_to)

    def _frontier_find_prob(self, robot: str, frontier: str, obj: str) -> float:
        return self._policy.resolve_frontier_find_prob()(robot, frontier, obj)

    def _container_find_prob(self, robot: str, loc: str, obj: str) -> float:
        return self._policy.resolve_container_find_prob()(robot, loc, obj)

    @classmethod
    def from_log(
        cls,
        log: RolloutLog,
        *,
        config: NavigationConfig | None = None,
        move_skill: type | None = None,
    ) -> "ReplayUnknownSearchEnvironment":
        """Build a policy-agnostic object-search replay arena from a recorded *log*.

        Container coordinates (``hidden_sites``), true contents
        (``recorded_object_locations``), and the actually-searched set are all
        reconstructed from the log's container subgoals, so the arena is
        self-contained. The searchable objects are the ones the recorded goal
        names. The candidate policy's probabilities are applied later by
        :func:`~railroad.replay.driver.run_replay`.
        """
        goal = require_goal(log)
        if move_skill is None:
            from railroad.environment.skill import NavigationMoveSkill

            move_skill = NavigationMoveSkill

        robots = log.robots
        hidden_sites = {
            s.signature: (int(s.centroid[0]), int(s.centroid[1])) for s in log.subgoals
        }
        recorded = {s.signature: set(s.contents) for s in log.subgoals}
        # Containers the deployment actually searched — outcome recorded, so
        # replay resolves them exactly (no optimistic commit).
        searched_sites = {s.signature for s in log.subgoals if s.searched}
        # The objects to search for are exactly those the goal names (minus the
        # robots and locations), so the search operators ground over them.
        objects = objects_in_goal(
            goal, exclude=set(robots) | {START_NAME} | set(hidden_sites)
        )

        fluents: Set[Fluent] = set()
        poses: Dict[str, Pose] = {}
        for robot in robots:
            fluents |= {
                Fluent(f"at {robot} {START_NAME}"),
                Fluent(f"free {robot}"),
                Fluent(f"revealed {START_NAME}"),
            }
            poses[robot] = Pose(*log.robot_starts[robot])
        start_xy = log.robot_starts[robots[0]][:2]

        return cls(
            recorded_grid=log.recorded_grid,
            recorded_object_locations=recorded,
            reference_cell=(int(start_xy[0]), int(start_xy[1])),
            goal=goal,
            hidden_sites=hidden_sites,
            state=State(0.0, fluents, []),
            objects_by_type={
                "robot": set(robots),
                "location": {START_NAME},
                "container": set(),
                "frontier": set(),
                "object": objects,
            },
            robot_initial_poses=poses,
            location_registry=LocationRegistry(
                {START_NAME: np.array(start_xy, dtype=float)}
            ),
            skill_overrides={"move": move_skill},
            config=config or navigation_config_from_log(log.config),
            searched_sites=searched_sites,
            pano_records=log.pano_records,
        )

    # -- goal / served-vantage support --------------------------------

    @property
    def goal(self) -> "Goal | Fluent":
        return self._goal

    @property
    def goal_cell(self) -> Tuple[int, int]:
        """Reference cell for egocentric features in served observations.

        Object search has no point goal; a fixed reference (the start) keeps
        ``compute_frontier_views`` well-defined. A learned model is trained with
        the same convention; the stub model ignores it.
        """
        return self._reference_cell

    @property
    def search_log(self) -> List[Tuple[str, float, bool]]:
        return self._search_log

    @property
    def replay_commits(self) -> List[Commit]:
        """One commit per not-found search (drives the optimistic bound)."""
        return self._replay_commits

    @property
    def oracle_available(self) -> bool:
        return False

    @property
    def oracle_labels(self) -> Dict:
        # No oracle in replay; satisfies the FrontierStatisticsEnvironment protocol.
        return {}

    def refresh_frontiers(self) -> None:
        super().refresh_frontiers()
        for estimator in self._refresh_estimators:
            estimator.refresh(self)

    # -- intercept: resolve search outcomes from the recording --------

    def resolve_probabilistic_effect(
        self, effect: GroundedEffect, current_fluents: Set[Fluent]
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        if effect.is_probabilistic:
            match = self._match_search_branches(effect)
            if match is not None:
                obj, loc, success, failure = match
                found = obj in self._recorded_object_locations.get(loc, set())
                # cost_accrued in seconds (sim time at the search) to match the
                # makespan-based bounds and deployment cost.
                accrued = float(self._time)
                self._search_log.append((loc, accrued, found))
                if not found:
                    self._maybe_log_commit(loc, accrued, failure)
                return (success if found else failure), current_fluents
        return super().resolve_probabilistic_effect(effect, current_fluents)

    def _maybe_log_commit(
        self, loc: str, accrued: float, failure_branch: List[GroundedEffect]
    ) -> None:
        """Log an optimistic commit for a not-found search.

        Every commit uses ``optimistic_to_goal = 0`` — the object could be at/just
        past the committed subgoal — exactly mirroring navigation ``lsp-explore``
        (which commits at the reached frontier), only with a 0 cost-to-goal since
        object search has no point goal.

        - **searched container** → no commit (outcome recorded → replay exactly).
        - **unsearched container** → commit (contents unknown; the object could be
          here — we do not assume one container per object).
        - **frontier** → commit. No "is the space beyond unseen?" check is needed:
          ``search-frontier`` requires ``at ?r ?frontier``, and a frontier whose
          beyond the deployment already revealed is *sensed away* when the robot
          reaches it (its free beyond becomes observed → it stops being a
          frontier). So a ``search-frontier`` that actually **executes** is always
          into genuinely-unseen space.
        """
        is_container = loc in self._recorded_object_locations
        if is_container and loc in self._searched_sites:
            return
        self._replay_commits.append(
            Commit(
                cost_accrued=accrued,
                optimistic_to_goal=0.0,
                robot=robot_from_free(failure_branch),
                frontier_signature=loc,
            )
        )

    @staticmethod
    def _match_search_branches(effect: GroundedEffect):
        """Return (obj, loc, success_effects, failure_effects) for a search effect."""
        branches = effect.prob_effects
        if len(branches) != 2:
            return None
        success = failure = None
        obj = loc = None
        for _, branch in branches:
            found_fluent = next(
                (
                    f
                    for eff in branch
                    for f in eff.resulting_fluents
                    if f.name == "found" and not f.negated and f.args
                ),
                None,
            )
            if found_fluent is not None:
                success = list(branch)
                obj = found_fluent.args[0]
                at_fluent = next(
                    (
                        f
                        for eff in branch
                        for f in eff.resulting_fluents
                        if f.name == "at" and not f.negated and len(f.args) >= 2
                    ),
                    None,
                )
                loc = at_fluent.args[1] if at_fluent is not None else None
            else:
                failure = list(branch)
        if success is None or failure is None or obj is None or loc is None:
            return None
        return obj, loc, success, failure


# ----------------------------------------------------------------------
# Learned served-vantage frontier-search probability
# ----------------------------------------------------------------------


def learned_frontier_search_prob(
    model,
) -> Tuple[LearnedFrontierStatistics, ProbFn]:
    """A served-vantage frontier-search probability backed by a learned model.

    Returns ``(estimator, prob_fn)``: put ``estimator`` in the policy's
    ``refresh_estimators`` (so it is fed the recorded panoramas), and use
    ``prob_fn`` as the policy's ``frontier_find_prob``. The object beyond a
    frontier is predicted from that frontier's best-vantage panorama — exactly
    the navigation learned pipeline, reused for search.
    """
    estimator = LearnedFrontierStatistics(model)

    def prob_fn(robot: str, frontier: str, obj: str) -> float:
        del obj
        return float(estimator.get(robot, frontier).prob_feasible)

    return estimator, prob_fn

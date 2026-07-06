"""Object-search replay over a recorded unknown-environment deployment.

Same idea as the navigation :class:`~railroad.replay.replay_env.ReplayEnvironment`
but for **frontier-based object search** (``examples/frontier_search.py``): the
robot explores frontiers and searches containers for a target object. Replay
confines the robot to the recorded map (shared :class:`ReplayConfinementMixin`)
and resolves every search outcome from the **recorded ground truth** — so an
alternative search policy's cost is reproduced without redeploying it.

The search-frontier probability ("is the object beyond this frontier?") is the
learned, served-vantage knob: it is driven by a
:class:`~railroad.lsp.frontier_statistics.LearnedFrontierStatistics` fed the
recorded panoramas (best-vantage per frontier). A trained network drops in at the
same model call site as in navigation (see :mod:`railroad.replay.stub_model`).
"""

from __future__ import annotations

from typing import Callable, Collection, Dict, List, Sequence, Set, Tuple

import numpy as np

from railroad._bindings import Fluent, GroundedEffect, State
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
from railroad.lsp.frontier_statistics import (
    FrontierStatisticsEstimator,
    LearnedFrontierStatistics,
)
from railroad.operators import construct_no_op_operator

from .base_env import ReplayConfinementMixin, navigation_config_from_log
from .cost import Commit
from .types import ReplayResult, RolloutLog


class SearchReplayEnvironment(ReplayConfinementMixin, UnknownSpaceEnvironment):
    """Replay an object-search policy over a recorded deployment."""

    def __init__(
        self,
        *,
        recorded_grid: np.ndarray,
        recorded_object_locations: Dict[str, Set[str]],
        reference_cell: Tuple[int, int],
        searched_sites: Collection[str] = (),
        refresh_estimators: Sequence[FrontierStatisticsEstimator] = (),
        pano_records: Sequence = (),
        **kwargs,
    ) -> None:
        confinement = self._setup_replay_grids(recorded_grid, pano_records)
        self._recorded_object_locations = recorded_object_locations
        # Containers the deployment actually searched: their outcome is known, so
        # replay resolves them exactly (no optimistic commit). A revealed-but-
        # unsearched container is unknown — we must not infer it empty from the
        # object being found elsewhere (that assumes one container per object).
        self._searched_sites = set(searched_sites)
        self._reference_cell = (int(reference_cell[0]), int(reference_cell[1]))
        self._refresh_estimators = list(refresh_estimators)
        # (loc, cost_accrued, found) per executed search.
        self._search_log: List[Tuple[str, float, bool]] = []
        # One commit per not-found search at a subgoal the deployment did NOT
        # verify (unsearched container or frontier). optimistic_lb = min over
        # these (design §6/§7).
        self._replay_commits: List[Commit] = []
        super().__init__(true_grid=confinement, **kwargs)
        for robot in self._objects_by_type.get("robot", set()):
            self._net_motion.setdefault(robot, 0.0)

    # -- served-vantage perception support ----------------------------

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

    # Arena handle for run_search_replay (option A: rebuild per candidate).
    _source_log: "RolloutLog | None" = None
    _target_object: str = ""

    @classmethod
    def from_log(
        cls,
        log: RolloutLog,
        *,
        target_object: str,
        config: NavigationConfig | None = None,
        move_skill: type | None = None,
    ) -> "SearchReplayEnvironment":
        """Build an object-search replay arena from a recorded *log*.

        ``hidden_sites`` (container coords) and ``recorded_object_locations``
        (true contents) are reconstructed from the log's container subgoals — so
        the arena is self-contained. A neutral search prior is used; the candidate
        policy's probabilities are supplied to :func:`run_search_replay`.
        """
        hidden_sites = {s.signature: (int(s.centroid[0]), int(s.centroid[1])) for s in log.subgoals}
        recorded = {s.signature: set(s.contents) for s in log.subgoals}
        env = build_search_replay_env(
            log,
            frontier_find_prob=lambda r, f, o: 0.5,
            container_find_prob=lambda r, l, o: 0.5,
            hidden_sites=hidden_sites,
            target_object=target_object,
            recorded_object_locations=recorded,
            config=config,
            move_skill=move_skill,
        )
        env._source_log = log
        env._target_object = target_object
        return env

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
          into genuinely-unseen space (design §7; same reasoning as ``lsp-explore``).
        """
        is_container = loc in self._recorded_object_locations
        if is_container and loc in self._searched_sites:
            return
        self._replay_commits.append(
            Commit(
                cost_accrued=accrued,
                optimistic_to_goal=0.0,
                robot=self._robot_of_branch(failure_branch),
                frontier_signature=loc,
            )
        )

    @staticmethod
    def _robot_of_branch(branch: List[GroundedEffect]) -> str:
        """The robot named by a branch's ``free ?robot`` effect (provenance)."""
        for eff in branch:
            for f in eff.resulting_fluents:
                if f.name == "free" and not f.negated and f.args:
                    return f.args[0]
        return ""

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
# Policy + drivers
# ----------------------------------------------------------------------

ProbFn = Callable[[str, str, str], float]


def learned_frontier_search_prob(
    model,
) -> Tuple[LearnedFrontierStatistics, ProbFn]:
    """A served-vantage frontier-search probability backed by a learned model.

    Returns ``(estimator, prob_fn)``: register ``estimator`` with the env's
    ``refresh_estimators`` (so it is fed the recorded panoramas), and pass
    ``prob_fn`` as the ``search-frontier`` operator's ``object_find_prob``. The
    object beyond a frontier is predicted from that frontier's best-vantage
    panorama — exactly the navigation learned pipeline, reused for search.
    """
    estimator = LearnedFrontierStatistics(model)

    def prob_fn(robot: str, frontier: str, obj: str) -> float:
        del obj
        return float(estimator.get(robot, frontier).prob_feasible)

    return estimator, prob_fn


def build_search_replay_env(
    log,
    *,
    frontier_find_prob: ProbFn,
    container_find_prob: ProbFn,
    refresh_estimators: Sequence[FrontierStatisticsEstimator] = (),
    hidden_sites: Dict[str, Tuple[int, int]],
    target_object: str,
    recorded_object_locations: Dict[str, Set[str]],
    search_time: float = 20.0,
    config: NavigationConfig | None = None,
    move_skill: type | None = None,
) -> SearchReplayEnvironment:
    """Construct a :class:`SearchReplayEnvironment` from a recorded *log*."""
    if move_skill is None:
        from railroad.environment.skill import NavigationMoveSkill

        move_skill = NavigationMoveSkill

    robots = log.robots
    env_ref: list = [None]

    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        if env_ref[0] is None:
            return 5.0
        return env_ref[0].estimate_move_time_safe(robot, loc_from, loc_to)

    operators = [
        construct_move_navigable_operator(move_time_fn),
        construct_search_frontier_operator(
            object_find_prob=frontier_find_prob, search_time=search_time
        ),
        construct_search_at_site_operator(
            container_find_prob, search_time=search_time, container_type="container"
        ),
        construct_no_op_operator(no_op_time=300.0, extra_cost=100.0),
    ]

    fluents: Set[Fluent] = set()
    poses: Dict[str, Pose] = {}
    for robot in robots:
        fluents |= {
            Fluent(f"at {robot} start_loc"),
            Fluent(f"free {robot}"),
            Fluent("revealed start_loc"),
        }
        poses[robot] = Pose(*log.robot_starts[robot])
    start_xy = log.robot_starts[robots[0]][:2]

    # Containers the deployment actually searched — their outcome is recorded,
    # so replay resolves them exactly (no optimistic commit). Derived from the
    # log so callers need not thread it.
    searched_sites = {s.signature for s in log.subgoals if s.searched}

    env = SearchReplayEnvironment(
        recorded_grid=log.recorded_grid,
        recorded_object_locations=recorded_object_locations,
        reference_cell=(int(start_xy[0]), int(start_xy[1])),
        searched_sites=searched_sites,
        refresh_estimators=refresh_estimators,
        pano_records=log.pano_records,
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": {"start_loc"},
            "container": set(),
            "frontier": set(),
            "object": {target_object},
        },
        operators=operators,
        skill_overrides={"move": move_skill},
        true_object_locations=recorded_object_locations,
        robot_initial_poses=poses,
        location_registry=LocationRegistry(
            {"start_loc": np.array(start_xy, dtype=float)}
        ),
        hidden_sites=hidden_sites,
        config=config or navigation_config_from_log(log.config),
    )
    env_ref[0] = env
    return env


def run_search_replay(
    arena: "SearchReplayEnvironment",
    *,
    frontier_find_prob: ProbFn,
    container_find_prob: ProbFn,
    refresh_estimators: Sequence[FrontierStatisticsEstimator] = (),
    config: NavigationConfig | None = None,
    max_planning_iterations: int = 80,
    mcts_iterations: int = 4000,
    mcts_c: float = 300.0,
    mcts_max_depth: int = 20,
    mcts_heuristic_multiplier: float = 2.0,
) -> ReplayResult:
    """Replay one object-search candidate policy over *arena*; return its bounds.

    *arena* is a :meth:`SearchReplayEnvironment.from_log` handle. A thin wrapper
    over the unified :func:`~railroad.replay.domains.replay` (unknown-map search
    domain): a fresh arena is built per call and configured with the candidate's
    probability callables; outcomes resolve from the recorded ground truth.
    """
    from .domains import MctsParams, replay
    from .policy import CandidatePolicy

    log = arena._source_log
    assert isinstance(log, RolloutLog)
    return replay(
        log,
        CandidatePolicy(
            frontier_find_prob=frontier_find_prob,
            container_find_prob=container_find_prob,
            refresh_estimators=refresh_estimators,
        ),
        target_object=arena._target_object,
        config=config,
        max_planning_iterations=max_planning_iterations,
        mcts=MctsParams(
            mcts_iterations, mcts_c, mcts_max_depth, mcts_heuristic_multiplier
        ),
    )

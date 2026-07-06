"""The replay environment and the ``run_replay`` driver (navigation).

``ReplayEnvironment`` is a GL-free LSP point-goal environment whose
"world" is a *recorded* final map rather than a live simulator. It
realizes the two §5 patches and the §6 intercept:

* **Pessimistic / confinement sensing.** The laser ranges are cast against
  a confinement grid (recorded map with ``UNOBSERVED -> COLLISION``) so the
  robot is structurally confined to known free space, while the *values*
  written into ``_observed_grid`` are corrected against the **pristine**
  recorded map — so masked/behind-frontier cells are never recorded as
  obstacles (§5.1.1). By construction ``_observed_grid`` only ever holds an
  obstacle where the pristine map does.
* **Intercept.** ``lsp-explore`` always resolves to its *failure* branch
  (the deployment recorded no map beyond a frontier), which sets
  ``explored ?f`` so the existing dead-frontier pruning retires it (§5.2).
  Each commit logs ``cost_accrued + optimistic_cost_to_goal`` for the bound
  (§6), keyed by frontier signature so a re-extracted frontier is never
  committed twice (§6.1).

``run_replay`` drives the plan→act loop with an injectable action selector
(production: MCTS; tests: :func:`frontier_sweep_select`) and reduces the
recorded commits + final makespan to :class:`~railroad.replay.cost.Bounds`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np

from railroad._bindings import Fluent, Goal, GroundedEffect, State
from railroad.environment.symbolic import LocationRegistry
from railroad.environment.types import Pose
from railroad.experimental.unknown_search import (
    NavigationConfig,
    UnknownSpaceEnvironment,
)
from railroad.lsp.env_mixin import LSPEnvironmentMixin
from railroad.lsp.frontier_statistics import FrontierStatisticsEstimator
from railroad.lsp.oracle import frontier_cells_hash

from .base_env import ReplayConfinementMixin, navigation_config_from_log
from .cost import Commit, optimistic_cost_to_goal
from .types import ReplayResult, RolloutLog

START_NAME = "start_loc"

ActionSelector = Callable[["ReplayEnvironment", list, "Goal | Fluent"], str]


class ReplayEnvironment(LSPEnvironmentMixin, ReplayConfinementMixin, UnknownSpaceEnvironment):
    """LSP point-goal environment driven over a recorded final map."""

    def __init__(
        self,
        *,
        recorded_grid: np.ndarray,
        goal_cell: Tuple[int, int],
        frontier_statistics: FrontierStatisticsEstimator,
        robot_initial_poses: Dict[str, Pose],
        location_registry: LocationRegistry,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        skill_overrides: Dict[str, type],
        config: NavigationConfig | None = None,
        pano_records: Sequence = (),
    ) -> None:
        # Confinement/pristine grids + served panos + net-motion (shared base).
        confinement = self._setup_replay_grids(recorded_grid, pano_records)

        self._lsp_goal_cell = (int(goal_cell[0]), int(goal_cell[1]))
        self._lsp_frontier_statistics = frontier_statistics

        robots = objects_by_type.get("robot", set())
        self._net_motion = {r: 0.0 for r in robots}
        self._replay_commits: List[Commit] = []
        self._retired_signatures: Set[str] = set()

        super().__init__(
            state=state,
            objects_by_type=objects_by_type,
            operators=None,
            true_grid=confinement,
            robot_initial_poses=robot_initial_poses,
            location_registry=location_registry,
            skill_overrides=skill_overrides,
            config=config or NavigationConfig(),
        )

    # -- replay-specific accessors ------------------------------------

    # The log this arena was built from (so run_replay can rebuild a fresh
    # arena per candidate policy — see option (A) in the replay architecture).
    _source_log: "RolloutLog | None" = None

    @classmethod
    def from_log(
        cls,
        log: RolloutLog,
        frontier_statistics: FrontierStatisticsEstimator | None = None,
        *,
        config: NavigationConfig | None = None,
        move_skill: type | None = None,
    ) -> "ReplayEnvironment":
        """Build a navigation replay arena from a recorded *log*.

        ``frontier_statistics`` is the candidate policy; when omitted a neutral
        fixed prior is used (the arena is then just a policy-agnostic handle that
        ``run_replay`` rebuilds with each candidate).
        """
        if frontier_statistics is None:
            from railroad.lsp.frontier_statistics import FixedPriorFrontierStatistics

            frontier_statistics = FixedPriorFrontierStatistics()
        env = build_replay_env(
            log, frontier_statistics, config=config, move_skill=move_skill
        )
        env._source_log = log
        return env

    @property
    def replay_commits(self) -> List[Commit]:
        """Commits recorded so far (one per uniquely-explored frontier)."""
        return self._replay_commits

    @property
    def oracle_available(self) -> bool:
        # Replay resolves explore outcomes itself (always failure); there is
        # no true map to label from.
        return False

    # -- intercept: lsp-explore always fails (§6) ---------------------

    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        if effect.is_probabilistic:
            match = self._match_lsp_explore_branches(effect)
            if match is not None:
                frontier_id, _success, failure_effects = match
                self._record_explore_commit(frontier_id, failure_effects)
                return failure_effects, current_fluents
        return super().resolve_probabilistic_effect(effect, current_fluents)

    def _record_explore_commit(
        self, frontier_id: str, branch_effects: List[GroundedEffect]
    ) -> None:
        frontier = self._frontiers.get(frontier_id)
        if frontier is None:
            return
        signature = frontier_cells_hash(frontier)
        if signature in self._retired_signatures:
            return
        self._retired_signatures.add(signature)
        robot = self._robot_from_effects(branch_effects)
        # All replay costs are in deployment units (seconds / makespan), so they
        # compare directly with the recorded actual_total_cost. The committing
        # robot is *at* the frontier now, so the cost paid to reach it is the
        # current sim time; the optimistic frontier->goal segment is geometric
        # (cells) and is converted to seconds with the same speed the env charges
        # for motion (estimate_move_time = cells / speed_cells_per_sec).
        speed = self._config.speed_cells_per_sec
        optimistic = optimistic_cost_to_goal(
            self._pristine_grid,
            (frontier.centroid_row, frontier.centroid_col),
            self._lsp_goal_cell,
        ) / speed
        accrued = float(self._time)
        self._replay_commits.append(
            Commit(
                cost_accrued=accrued,
                optimistic_to_goal=optimistic,
                robot=robot,
                frontier_signature=signature,
            )
        )

    def _robot_from_effects(self, branch_effects: List[GroundedEffect]) -> str:
        robots = self._objects_by_type.get("robot", set())
        for effect in branch_effects:
            for fluent in effect.resulting_fluents:
                if (
                    fluent.name == "free"
                    and not fluent.negated
                    and fluent.args
                    and fluent.args[0] in robots
                ):
                    return fluent.args[0]
        return next(iter(sorted(robots)), "")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def build_replay_env(
    log: RolloutLog,
    frontier_statistics: FrontierStatisticsEstimator,
    *,
    config: NavigationConfig | None = None,
    move_skill: type | None = None,
) -> ReplayEnvironment:
    """Construct a :class:`ReplayEnvironment` from a recorded *log*."""
    if move_skill is None:
        from railroad.environment.skill import NavigationMoveSkill

        move_skill = NavigationMoveSkill

    robots = log.robots
    initial_fluents: Set[Fluent] = {Fluent(f"revealed {START_NAME}")}
    for robot in robots:
        initial_fluents.add(Fluent(f"at {robot} {START_NAME}"))
        initial_fluents.add(Fluent(f"free {robot}"))

    robot_poses = {robot: Pose(*log.robot_starts[robot]) for robot in robots}
    start_xy = log.robot_starts[robots[0]][:2]

    return ReplayEnvironment(
        recorded_grid=log.recorded_grid,
        goal_cell=log.goal_cell,
        frontier_statistics=frontier_statistics,
        robot_initial_poses=robot_poses,
        location_registry=LocationRegistry(
            {START_NAME: np.array(start_xy, dtype=float)}
        ),
        state=State(0.0, initial_fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": {START_NAME},
            "frontier": set(),
            "object": set(),
        },
        skill_overrides={"move": move_skill, "move-to-goal": move_skill},
        config=config or navigation_config_from_log(log.config),
        pano_records=log.pano_records,
    )


def _explored_frontiers(env: ReplayEnvironment) -> Set[str]:
    """Frontier ids already retired via an ``explored`` fluent.

    They remain in ``env.frontiers`` geometrically (the pocket is still
    unobserved), so a goal-seeking selector must skip them — mirroring the
    planner's dead-frontier pruning.
    """
    return {
        fluent.args[0]
        for fluent in env.state.fluents
        if fluent.name == "explored" and not fluent.negated and fluent.args
    }


def goal_fluent(robots: List[str]) -> "Goal | Fluent":
    """The point-goal goal: *any* robot reaches ``goal``."""
    goal = Fluent(f"at {robots[0]} goal")
    for robot in robots[1:]:
        goal = goal | Fluent(f"at {robot} goal")
    return goal


def frontier_sweep_select(
    env: ReplayEnvironment, actions: list, goal: "Goal | Fluent"
) -> str:
    """A deterministic policy: explore reachable frontiers, then head to goal.

    Priority: drive to a revealed goal; else explore the frontier under the
    robot; else move to the lowest-id *unexplored* reachable frontier; else
    any move. Used to make replay tests deterministic (no MCTS stochasticity).
    """
    applicable = [a for a in actions if env.state.satisfies_precondition(a)]
    explored = _explored_frontiers(env)

    def first(predicate) -> str | None:
        chosen = sorted(
            (a for a in applicable if predicate(a)), key=lambda a: a.name
        )
        return chosen[0].name if chosen else None

    def parts(action) -> list:
        return action.name.split()

    goal_move = first(
        lambda a: parts(a)[0] == "move" and parts(a)[-1] == "goal"
    )
    if goal_move is not None:
        return goal_move

    explore = first(lambda a: parts(a)[0] == "lsp-explore")
    if explore is not None:
        return explore

    to_frontier = first(
        lambda a: parts(a)[0] == "move"
        and parts(a)[-1] in env.frontiers
        and parts(a)[-1] not in explored
    )
    if to_frontier is not None:
        return to_frontier

    any_move = first(lambda a: parts(a)[0] in ("move", "move-to-goal"))
    if any_move is not None:
        return any_move

    return "NONE"


def run_replay(
    arena: "ReplayEnvironment | RolloutLog",
    frontier_statistics: FrontierStatisticsEstimator,
    *,
    select_action: ActionSelector | None = None,
    config: NavigationConfig | None = None,
    max_planning_iterations: int = 300,
    mcts_iterations: int = 4000,
    mcts_c: float = 10.0,
    mcts_max_depth: int = 20,
    mcts_heuristic_multiplier: float = 5.0,
) -> ReplayResult:
    """Replay one navigation candidate policy over a recording; return its bounds.

    A thin wrapper over the unified :func:`~railroad.replay.domains.replay`
    (navigation domain). *arena* is a :meth:`ReplayEnvironment.from_log` handle
    (preferred) or a raw :class:`RolloutLog`; a fresh arena is built per call and
    configured with *frontier_statistics* (the candidate policy), so the same
    log replays many candidates.
    """
    from .domains import MctsParams, replay
    from .policy import CandidatePolicy

    log = (
        arena._source_log
        if isinstance(arena, ReplayEnvironment) and arena._source_log is not None
        else arena
    )
    assert isinstance(log, RolloutLog)
    return replay(
        log,
        CandidatePolicy(frontier_statistics=frontier_statistics),
        config=config,
        select_action=select_action,
        max_planning_iterations=max_planning_iterations,
        mcts=MctsParams(
            mcts_iterations, mcts_c, mcts_max_depth, mcts_heuristic_multiplier
        ),
    )

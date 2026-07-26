"""Point-goal navigation replay in unknown space.

``ReplayPointGoalNavEnvironment`` is a GL-free LSP point-goal environment whose
"world" is a *recorded* final map rather than a live simulator. It mirrors the
deployment ``LSPVisualEnvironment`` (same MRO, minus the GL backend) and realizes
the two patches and the intercept:

* **Confinement sensing.** The laser ranges are cast against a confinement grid
  (recorded map with ``UNOBSERVED -> COLLISION``) so the robot is structurally
  confined to known free space, while the *values* written into
  ``_observed_grid`` are corrected against the **pristine** recorded map — so
  masked/behind-frontier cells are never recorded as obstacles (see
  :class:`~railroad.replay.environments.base.ReplayConfinementMixin`).
* **Intercept.** ``lsp-explore`` always resolves to its *failure* branch (the
  deployment recorded no map beyond a frontier), which sets ``explored ?f`` so
  the existing dead-frontier pruning retires it. Each commit logs
  ``cost_accrued + optimistic_cost_to_goal`` for the bound, keyed by frontier
  signature so a re-extracted frontier is never committed twice.

The candidate policy is the frontier-statistics estimator; it is applied via
``apply_policy`` (the arena is built policy-agnostic and reused across candidates).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple, Type

import numpy as np

from railroad._bindings import Fluent, Goal, GroundedEffect, State
from railroad.environment.environment import ActiveSkill
from railroad.environment.symbolic import LocationRegistry
from railroad.environment.types import Pose
from railroad.experimental.unknown_search import (
    NavigationConfig,
    UnknownSpaceEnvironment,
)
from railroad.lsp.env_mixin import LSPEnvironmentMixin
from railroad.lsp.frontier_statistics import (
    FixedPriorFrontierStatistics,
    FrontierStatisticsEstimator,
)
from railroad.lsp.oracle import frontier_cells_hash

from ..cost import Commit, cost_at_cell, optimistic_cost_grid_from_goal
from ..loop import MctsConfig
from ..types import RolloutLog
from .base import (
    ReplayArenaMixin,
    ReplayConfinementMixin,
    navigation_config_from_log,
    robot_from_free,
)

START_NAME = "start_loc"


def goal_fluent(robots: List[str]) -> "Goal | Fluent":
    """The point-goal goal: *any* robot reaches ``goal``."""
    goal = Fluent(f"at {robots[0]} goal")
    for robot in robots[1:]:
        goal = goal | Fluent(f"at {robot} goal")
    return goal


class ReplayPointGoalNavEnvironment(
    ReplayArenaMixin, LSPEnvironmentMixin, ReplayConfinementMixin, UnknownSpaceEnvironment
):
    """LSP point-goal environment driven over a recorded final map."""

    default_mcts = MctsConfig(iterations=4000, c=10.0, max_depth=20, heuristic_multiplier=5.0)
    default_max_planning_iterations = 300
    dashboard_fluent_keywords = ("at", "explored", "revealed")

    def __init__(
        self,
        *,
        recorded_grid: np.ndarray,
        goal_cell: Tuple[int, int],
        robot_initial_poses: Dict[str, Pose],
        location_registry: LocationRegistry,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        skill_overrides: Dict[str, Type[ActiveSkill]],
        config: NavigationConfig | None = None,
        pano_records: Sequence = (),
    ) -> None:
        # Confinement/pristine grids + served panos + net-motion (shared base).
        confinement = self._setup_replay_grids(recorded_grid, pano_records)

        # Optional recorded goal; None -> derive the point-goal from the robots.
        self._goal: "Goal | Fluent | None" = None
        self._lsp_goal_cell = (int(goal_cell[0]), int(goal_cell[1]))
        # Neutral until apply_policy() installs a candidate.
        self._lsp_frontier_statistics = FixedPriorFrontierStatistics()

        robots = objects_by_type.get("robot", set())
        self._net_motion = {r: 0.0 for r in robots}
        self._replay_commits: List[Commit] = []
        self._retired_signatures: Set[str] = set()
        # Both inputs are fixed for the arena's lifetime, so the optimistic
        # cost-to-goal grid is built once instead of per commit.
        self._optimistic_cost_grid = optimistic_cost_grid_from_goal(
            self._pristine_grid, self._lsp_goal_cell
        )

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

    @classmethod
    def from_log(
        cls,
        log: RolloutLog,
        *,
        config: NavigationConfig | None = None,
        move_skill: Type[ActiveSkill] | None = None,
    ) -> "ReplayPointGoalNavEnvironment":
        """Build a policy-agnostic navigation replay arena from a recorded *log*."""
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

        env = cls(
            recorded_grid=log.recorded_grid,
            goal_cell=log.goal_cell,
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
        # Use the recorded goal if the deployment stored one; otherwise the env
        # derives the point-goal (any robot reaches ``goal``) from its robots.
        env._goal = log.goal
        return env

    # -- policy / goal / bookkeeping ----------------------------------

    def apply_policy(self, policy: FrontierStatisticsEstimator) -> None:
        """Install the navigation belief. The mixin's setter also refreshes it,
        and lsp-explore reads it live, so the candidate takes effect on this
        already-built arena without rebuilding operators."""
        self.frontier_statistics = policy

    @property
    def goal(self) -> "Goal | Fluent":
        if self._goal is not None:
            return self._goal
        return goal_fluent(sorted(self._objects_by_type.get("robot", set())))

    @property
    def replay_commits(self) -> List[Commit]:
        """Commits recorded so far (one per uniquely-explored frontier)."""
        return self._replay_commits

    @property
    def oracle_available(self) -> bool:
        """No ground truth here, so ``oracle_labels`` must stay empty.

        ``_true_grid`` is the *confinement* grid (unobserved -> wall), not the
        real world. The inherited ``oracle_labels`` would label against it and
        report a frontier the goal genuinely lies beyond as infeasible —
        silently-wrong ground truth wearing the oracle's name. Returning False
        makes ``oracle_labels`` return ``{}`` instead.

        An oracle *candidate* is unaffected: it carries the scene's true map
        itself rather than reading one off the environment.
        """
        return False

    # -- intercept: lsp-explore always fails ---------------------

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
        robot = robot_from_free(branch_effects, self._objects_by_type.get("robot", set()))
        # All replay costs are in deployment units (seconds / makespan), so they
        # compare directly with the recorded actual_total_cost. The committing
        # robot is *at* the frontier now, so the cost paid to reach it is the
        # current sim time; the optimistic frontier->goal segment is geometric
        # (cells) and is converted to seconds with the same speed the env charges
        # for motion (estimate_move_time = cells / speed_cells_per_sec).
        speed = self._config.speed_cells_per_sec
        optimistic = cost_at_cell(
            self._optimistic_cost_grid,
            (frontier.centroid_row, frontier.centroid_col),
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


def _explored_frontiers(env: ReplayPointGoalNavEnvironment) -> Set[str]:
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


def frontier_sweep_select(
    env: ReplayPointGoalNavEnvironment, actions: list, goal: "Goal | Fluent"
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

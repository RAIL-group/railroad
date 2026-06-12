"""Point-goal-navigation rollouts: shared setup and a headless run loop.

Requires the railsim optional dependency (``railroad[railsim]``), like
:mod:`railroad.lsp.environment`. The interactive example
(``railroad example lsp-point-goal-nav``) builds its environment through
:func:`build_point_goal_setup` and adds a dashboard;
:func:`run_point_goal_rollout` is the dashboard-free variant used for
bulk training-data generation (:mod:`railroad.lsp.bulk`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from railroad._bindings import Fluent, State
from railroad.core import get_action_by_name
from railroad.environment.railsim import RailsimScene
from railroad.environment.symbolic import LocationRegistry
from railroad.experimental.unknown_search import NavigationConfig, Pose
from railroad.planner import MCTSPlanner

from .bulk import RolloutResult
from .data import TrainingDataWriter
from .environment import LSPVisualEnvironment
from .frontier_statistics import (
    FixedPriorFrontierStatistics,
    FrontierStatisticsEstimator,
    OracleFrontierStatistics,
)

F = Fluent

ROBOT = "robot1"
START_NAME = "start_loc"


@dataclass
class PointGoalSetup:
    """Everything a point-goal-navigation run needs."""

    scene: RailsimScene
    env: LSPVisualEnvironment
    goal: Fluent
    goal_cell: tuple[int, int]
    data_writer: TrainingDataWriter | None


def _make_frontier_statistics(
    name: str, prior_prob: float
) -> FrontierStatisticsEstimator:
    if name == "oracle":
        return OracleFrontierStatistics()
    if name == "fixed-prior":
        return FixedPriorFrontierStatistics(
            prob_feasible=prior_prob,
            delta_success_cost=0.0,
            exploration_cost=10.0,
        )
    raise ValueError(
        f"Unknown frontier statistics {name!r}; "
        "expected 'oracle' or 'fixed-prior'"
    )


def build_point_goal_setup(
    env_name: str,
    seed: int,
    *,
    frontier_statistics_name: str = "oracle",
    prior_prob: float = 0.8,
    save_data_dir: str | Path | None = None,
    allow_move_interruptions: bool = False,
) -> PointGoalSetup:
    """Build the scene, environment, and (optionally) data writer."""
    if env_name == "maze":
        scene = RailsimScene.maze(seed=seed)
    elif env_name == "office":
        from railroad.environment.railsim import OfficeConfig

        scene = RailsimScene.office(
            seed=seed,
            config=OfficeConfig(grid_size=(300, 200), num_hallways=4),
        )
    else:
        raise ValueError(f"Unknown env {env_name!r}; expected 'maze' or 'office'")

    start_coord = scene.locations["start_loc"]
    goal_coord = scene.locations["goal_loc"]
    goal_cell = (int(goal_coord[0]), int(goal_coord[1]))

    frontier_statistics = _make_frontier_statistics(
        frontier_statistics_name, prior_prob
    )

    if allow_move_interruptions:
        from railroad.environment.skill import InterruptibleNavigationMoveSkill
        move_skill = InterruptibleNavigationMoveSkill
    else:
        from railroad.environment.skill import NavigationMoveSkill
        move_skill = NavigationMoveSkill

    data_writer = None
    if save_data_dir is not None:
        data_writer = TrainingDataWriter(
            save_data_dir,
            run_metadata={
                "env": env_name,
                "seed": seed,
                "frontier_statistics": frontier_statistics_name,
                "goal_cell": [goal_cell[0], goal_cell[1]],
            },
        )

    env = LSPVisualEnvironment(
        scene=scene,
        frontier_statistics=frontier_statistics,
        data_writer=data_writer,
        state=State(0.0, {
            F(f"at {ROBOT} {START_NAME}"),
            F(f"free {ROBOT}"),
            F(f"revealed {START_NAME}"),
        }, []),
        objects_by_type={
            "robot": {ROBOT},
            "location": {START_NAME},
            "frontier": set(),
            "object": set(),
        },
        skill_overrides={'move': move_skill},
        robot_initial_poses={
            ROBOT: Pose(float(start_coord[0]), float(start_coord[1]), 0.0)
        },
        location_registry=LocationRegistry({
            START_NAME: np.array(start_coord, dtype=float)
        }),
        config=NavigationConfig(
            sensor_range=60.0,
            max_move_action_time=10_000.0,
            interrupt_min_new_cells=30000,
            interrupt_min_dt=30000.0,
        ),
    )

    return PointGoalSetup(
        scene=scene,
        env=env,
        goal=F(f"at {ROBOT} goal"),
        goal_cell=goal_cell,
        data_writer=data_writer,
    )


def run_point_goal_rollout(
    env_name: str,
    seed: int,
    save_data_dir: str | Path,
    *,
    frontier_statistics_name: str = "oracle",
    prior_prob: float = 0.8,
    max_planning_iterations: int = 200,
    mcts_iterations: int = 4000,
    mcts_c: float = 10,
    mcts_max_depth: int = 20,
    mcts_heuristic_multiplier: float = 5,
) -> RolloutResult:
    """Run one full plan/act rollout headlessly, writing training data.

    Exceptions propagate (after the scene's GL resources are released);
    callers that must not crash — the bulk worker — convert them into a
    failure result themselves.
    """
    t0 = time.perf_counter()
    setup = build_point_goal_setup(
        env_name,
        seed,
        frontier_statistics_name=frontier_statistics_name,
        prior_prob=prior_prob,
        save_data_dir=save_data_dir,
    )
    env, goal = setup.env, setup.goal
    try:
        termination = "max_iterations"
        for _ in range(max_planning_iterations):
            if goal.evaluate(env.state.fluents):
                termination = "goal_reached"
                break

            actions = env.get_actions()
            if not actions:
                termination = "no_actions"
                break

            # See examples/lsp_point_goal_nav.py for the MCTS tuning
            # rationale (value-driven point-goal navigation).
            mcts = MCTSPlanner(actions)
            action_name = mcts(
                env.state,
                goal,
                max_iterations=mcts_iterations,
                c=mcts_c,
                max_depth=mcts_max_depth,
                heuristic_multiplier=mcts_heuristic_multiplier,
            )
            if action_name == "NONE":
                termination = "planner_none"
                break

            env.act(get_action_by_name(actions, action_name))

        # The goal may have been reached by the final permitted action.
        if termination == "max_iterations" and goal.evaluate(env.state.fluents):
            termination = "goal_reached"

        return RolloutResult(
            seed=seed,
            env_name=env_name,
            goal_reached=(termination == "goal_reached"),
            termination=termination,
            sim_time=float(env.state.time),
            num_data_written=env.num_data_written,
            num_panoramas=len(env.pano_records),
            wall_time=time.perf_counter() - t0,
        )
    finally:
        if setup.data_writer is not None:
            setup.data_writer.close()
        setup.scene.release()

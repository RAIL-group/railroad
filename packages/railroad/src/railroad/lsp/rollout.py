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

from railroad._bindings import Fluent, Goal, State
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


def _robot_names(num_robots: int) -> list[str]:
    """Names for ``num_robots`` robots: ``robot1``, ``robot2``, ..."""
    if num_robots < 1:
        raise ValueError(f"num_robots must be >= 1, got {num_robots}")
    return [f"robot{i}" for i in range(1, num_robots + 1)]


@dataclass
class PointGoalSetup:
    """Everything a point-goal-navigation run needs."""

    scene: RailsimScene
    env: LSPVisualEnvironment
    goal: Fluent | Goal
    goal_cell: tuple[int, int]
    data_writer: TrainingDataWriter | None


def _make_frontier_statistics(
    name: str, prior_prob: float, network_file: str | Path | None = None
) -> FrontierStatisticsEstimator:
    if name == "oracle":
        return OracleFrontierStatistics()
    if name == "fixed-prior":
        return FixedPriorFrontierStatistics(
            prob_feasible=prior_prob,
            delta_success_cost=0.0,
            exploration_cost=10.0,
        )
    if name == "learned":
        if network_file is None:
            raise ValueError(
                "The 'learned' frontier statistics estimator needs trained "
                "weights; pass network_file (--network-file), e.g. the "
                "LSPFrontierNet.pt that 'railroad lsp train-network' saves."
            )
        from .frontier_statistics import LearnedFrontierStatistics
        from .model import load_frontier_statistics_model

        return LearnedFrontierStatistics(
            load_frontier_statistics_model(network_file)
        )
    raise ValueError(
        f"Unknown frontier statistics {name!r}; "
        "expected 'oracle', 'fixed-prior', or 'learned'"
    )


def build_point_goal_setup(
    env_name: str,
    seed: int,
    *,
    frontier_statistics_name: str = "oracle",
    prior_prob: float = 0.8,
    network_file: str | Path | None = None,
    save_data_dir: str | Path | None = None,
    allow_move_interruptions: bool = False,
    num_robots: int = 1,
) -> PointGoalSetup:
    """Build the scene, environment, and (optionally) data writer.

    With ``num_robots`` > 1, all robots start co-located at ``start_loc``;
    the goal is satisfied as soon as *any* robot reaches it.
    """
    robots = _robot_names(num_robots)
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
        frontier_statistics_name, prior_prob, network_file
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

    initial_fluents: set[Fluent] = {F(f"revealed {START_NAME}")}
    for robot in robots:
        initial_fluents.add(F(f"at {robot} {START_NAME}"))
        initial_fluents.add(F(f"free {robot}"))

    start_pose = Pose(float(start_coord[0]), float(start_coord[1]), 0.0)

    env = LSPVisualEnvironment(
        scene=scene,
        frontier_statistics=frontier_statistics,
        data_writer=data_writer,
        state=State(0.0, initial_fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": {START_NAME},
            "frontier": set(),
            "object": set(),
        },
        skill_overrides={'move': move_skill, 'move-to-goal': move_skill},
        robot_initial_poses={robot: start_pose for robot in robots},
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

    # The goal is reached once any robot arrives at the goal location.
    goal = F(f"at {robots[0]} goal")
    for robot in robots[1:]:
        goal = goal | F(f"at {robot} goal")

    return PointGoalSetup(
        scene=scene,
        env=env,
        goal=goal,
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
    network_file: str | Path | None = None,
    num_robots: int = 1,
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
        network_file=network_file,
        save_data_dir=save_data_dir,
        num_robots=num_robots,
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

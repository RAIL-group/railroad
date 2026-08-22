"""The _on_laser_scan hook fires for every simulated scan."""

from __future__ import annotations

import math

import numpy as np

from railroad._bindings import Fluent, State
from railroad.environment.skill import NavigationMoveSkill
from railroad.environment.symbolic import LocationRegistry
from railroad.experimental.unknown_search import (
    NavigationConfig,
    Pose,
    UnknownSpaceEnvironment,
)
from railroad.experimental.unknown_search.operators import (
    construct_move_navigable_operator,
)
from railroad.navigation.constants import COLLISION_VAL, FREE_VAL

from env_helpers import env_with_operators

F = Fluent


class _RecordingEnv(UnknownSpaceEnvironment):
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.scan_calls: list[tuple[str, float, int]] = []
        super().__init__(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    def _on_laser_scan(
        self, robot: str, pose: Pose, time: float, laser_ranges: np.ndarray
    ) -> None:
        self.scan_calls.append((robot, time, laser_ranges.shape[0]))


def test_on_laser_scan_hook_called_per_scan() -> None:
    grid = COLLISION_VAL * np.ones((30, 30))
    grid[4:7, 4:27] = FREE_VAL

    num_rays = 91
    env = env_with_operators(_RecordingEnv,
        state=State(0.0, {
            F("at robot1 start"),
            F("free robot1"),
            F("revealed start"),
        }, []),
        objects_by_type={
            "robot": {"robot1"},
            "location": {"start"},
            "frontier": set(),
            "object": set(),
        },
        operators=[construct_move_navigable_operator(5.0)],
        true_grid=grid,
        robot_initial_poses={"robot1": Pose(5.0, 5.0, 0.0)},
        location_registry=LocationRegistry({"start": np.array([5, 5])}),
        skill_overrides={"move": NavigationMoveSkill},
        config=NavigationConfig(
            sensor_range=8.0,
            sensor_fov_rad=2 * math.pi,
            sensor_num_rays=num_rays,
            interrupt_min_new_cells=30000,
            interrupt_min_dt=30000.0,
        ),
    )

    # The initial observation at t=0 fires the hook once per robot.
    assert env.scan_calls == [("robot1", 0.0, num_rays)]

    # Sensing during a move fires it again at each sensing step.
    move = next(
        a for a in env.get_actions() if a.name.startswith("move robot1 start")
    )
    env.act(move)
    assert len(env.scan_calls) > 1
    assert all(call[0] == "robot1" for call in env.scan_calls)

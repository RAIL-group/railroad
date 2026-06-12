"""Emission core: turn frontiers + labels + panoramas into training data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

from railroad.experimental.unknown_search.types import Frontier

from .data import (
    FrontierChangeTracker,
    TrainingDataWriter,
    frontier_signature,
    vantage_key,
)
from .types import LSPDataConfig, OracleFrontierLabel, TrainingDatum
from .views import compute_frontier_views

if TYPE_CHECKING:
    from railroad.environment.railsim import PanoRecord


class TrainingDataGenerator:
    """Writes a datum per frontier whenever its label or vantage changes."""

    def __init__(
        self,
        goal_cell: tuple[int, int],
        writer: TrainingDataWriter | None,
        config: LSPDataConfig | None = None,
    ) -> None:
        self.goal_cell = (int(goal_cell[0]), int(goal_cell[1]))
        self.writer = writer
        self.config = config or LSPDataConfig()
        self.tracker = FrontierChangeTracker()

    @property
    def num_written(self) -> int:
        return self.writer.num_written if self.writer is not None else 0

    def update(
        self,
        *,
        frontiers: Mapping[str, Frontier],
        labels: Mapping[str, OracleFrontierLabel],
        pano_records: Sequence["PanoRecord"],
    ) -> int:
        """Emit data for changed frontiers; returns the number written."""
        if self.writer is None:
            return 0

        views = compute_frontier_views(
            frontiers=frontiers,
            pano_records=pano_records,
            goal_cell=self.goal_cell,
            vantage_inflation_radius=self.config.vantage_inflation_radius,
        )
        written = 0
        for frontier_id, view in views.items():
            label = labels.get(frontier_id)
            if label is None:
                continue

            signature = frontier_signature(
                label, vantage_key(view.record),
                cost_round_decimals=self.config.cost_round_decimals,
            )
            if not self.tracker.should_emit(frontier_id, signature):
                continue

            observation = view.observation
            self.writer.write(TrainingDatum(
                image=observation.image,
                frontier_xy_ego=observation.frontier_xy_ego,
                goal_xy_ego=observation.goal_xy_ego,
                label=label.prob_feasible >= 0.5,
                success_cost=label.success_cost,
                optimistic_cost=label.optimistic_cost,
                exploration_cost=label.exploration_cost,
                metadata={
                    "frontier_id": frontier_id,
                    "signature": signature,
                    "robot": view.record.robot,
                    "time": view.record.time,
                    "success_cost": label.success_cost,
                    "optimistic_cost": label.optimistic_cost,
                    "exploration_cost": label.exploration_cost,
                },
            ))
            written += 1

        self.tracker.prune(frontiers.keys())
        return written

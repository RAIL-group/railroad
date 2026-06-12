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
from .pano import make_training_view
from .types import LSPDataConfig, OracleFrontierLabel, TrainingDatum

if TYPE_CHECKING:
    from railroad.environment.railsim import PanoRecord

from .vantage import select_best_vantage


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

        written = 0
        for frontier_id, frontier in frontiers.items():
            label = labels.get(frontier_id)
            if label is None:
                continue
            record = select_best_vantage(
                frontier, pano_records, self.config.vantage_inflation_radius
            )
            if record is None:
                continue

            key = vantage_key(record)
            signature = frontier_signature(
                label, key,
                cost_round_decimals=self.config.cost_round_decimals,
            )
            if not self.tracker.should_emit(frontier_id, signature):
                continue

            frontier_rc = (
                float(frontier.centroid_row), float(frontier.centroid_col)
            )
            image, frontier_xy, goal_xy = make_training_view(
                record, frontier_rc, (float(self.goal_cell[0]), float(self.goal_cell[1]))
            )
            self.writer.write(TrainingDatum(
                image=image,
                frontier_xy_ego=frontier_xy,
                goal_xy_ego=goal_xy,
                label=label.prob_feasible >= 0.5,
                success_cost=label.success_cost,
                optimistic_cost=label.optimistic_cost,
                exploration_cost=label.exploration_cost,
                metadata={
                    "frontier_id": frontier_id,
                    "signature": signature,
                    "robot": record.robot,
                    "time": record.time,
                    "success_cost": label.success_cost,
                    "optimistic_cost": label.optimistic_cost,
                    "exploration_cost": label.exploration_cost,
                },
            ))
            written += 1

        self.tracker.prune(frontiers.keys())
        return written

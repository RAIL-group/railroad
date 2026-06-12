"""Visual LSP environment: point-goal navigation + training-data emission.

Requires the railsim optional dependency (``railroad[railsim]``).
"""

from __future__ import annotations

from typing import Any

from railroad.environment.railsim import RailsimScene, VisualUnknownSpaceEnvironment

from .data import TrainingDataWriter
from .env_mixin import LSPEnvironmentMixin
from .generator import TrainingDataGenerator
from .types import LSPDataConfig


class LSPVisualEnvironment(LSPEnvironmentMixin, VisualUnknownSpaceEnvironment):
    """Frontier exploration toward a point goal, emitting LSP training data.

    On every frontier refresh (until the goal is observed), each frontier
    is labeled against the true map and a training datum is written
    whenever its label or best vantage changed.
    """

    def __init__(
        self,
        *,
        scene: RailsimScene,
        goal_cell: tuple[int, int] | None = None,
        data_writer: TrainingDataWriter | None = None,
        data_config: LSPDataConfig | None = None,
        **kwargs: Any,
    ) -> None:
        # Mixin state must exist before super().__init__: the base
        # constructor performs the initial observation and frontier
        # refresh, which triggers labeling and data emission.
        if goal_cell is None:
            goal_cell = scene.locations["goal_loc"]
        self._lsp_goal_cell = (int(goal_cell[0]), int(goal_cell[1]))
        config = data_config or LSPDataConfig()
        self._lsp_exploration_cost_factor = config.exploration_cost_factor
        self._lsp_generator = TrainingDataGenerator(
            self._lsp_goal_cell, data_writer, config
        )

        # ty resolves super() through the mixin's declared base; at runtime
        # the MRO reaches VisualUnknownSpaceEnvironment, which takes scene.
        super().__init__(scene=scene, **kwargs)  # ty: ignore[unknown-argument]

    @property
    def num_data_written(self) -> int:
        return self._lsp_generator.num_written

    def _after_lsp_refresh(self) -> None:
        if self.goal_observed:
            return
        self._lsp_generator.update(
            frontiers=self.frontiers,
            labels=self.oracle_labels,
            pano_records=self.pano_records,
        )

"""Learning over subgoals planning (LSP).

Vision-based learning about frontier-exploration uncertainty for
point-goal navigation: a frontier-exploration operator parameterized by
per-frontier *statistics* (probability of leading to the goal plus
success/exploration costs), and training-data generation from panoramic
images with best-vantage selection.

Frontier statistics come from one of three estimators:

- :class:`OracleFrontierStatistics` — exact values from the true map
  (simulation only; also used to label training data),
- :class:`FixedPriorFrontierStatistics` — fixed constants, no oracle
  needed (the deployment-safe default),
- :class:`LearnedFrontierStatistics` — a model predicting statistics
  from the same panorama observations the training data stores.

This package is GL-free except for :mod:`railroad.lsp.environment`
(``LSPVisualEnvironment``), which requires the railsim optional
dependency and must be imported explicitly::

    from railroad.lsp.environment import LSPVisualEnvironment
"""

from .data import (
    FrontierChangeTracker,
    TrainingDataWriter,
    frontier_signature,
    load_datum,
    read_index,
    vantage_key,
)
from .env_mixin import LSPEnvironmentMixin
from .frontier_statistics import (
    DEFAULT_FRONTIER_STATISTICS,
    FixedPriorFrontierStatistics,
    FrontierStatisticsEnvironment,
    FrontierStatisticsEstimator,
    FrontierStatisticsModel,
    LearnedFrontierStatistics,
    OracleFrontierStatistics,
)
from .generator import TrainingDataGenerator
from .inspect import inspect_data, make_inspection_figure
from .operators import (
    construct_lsp_explore_operator,
    construct_move_to_goal_operator,
)
from .oracle import (
    build_lookahead_grid,
    compute_oracle_frontier_labels,
    frontier_cells_hash,
    is_goal_observed,
    mask_grid_with_frontiers,
)
from .pano import (
    bearing_to_target,
    egocentric_xy,
    make_training_view,
    roll_pano_to_bearing,
    wrap_angle,
)
from .types import (
    FrontierObservation,
    FrontierStatistics,
    LSPDataConfig,
    OracleFrontierLabel,
    TrainingDatum,
)
from .vantage import count_cells_in_polygon, select_best_vantage
from .views import FrontierView, compute_frontier_views

__all__ = [
    "DEFAULT_FRONTIER_STATISTICS",
    "FixedPriorFrontierStatistics",
    "FrontierChangeTracker",
    "FrontierObservation",
    "FrontierStatistics",
    "FrontierStatisticsEnvironment",
    "FrontierStatisticsEstimator",
    "FrontierStatisticsModel",
    "FrontierView",
    "LSPDataConfig",
    "LSPEnvironmentMixin",
    "LearnedFrontierStatistics",
    "OracleFrontierLabel",
    "OracleFrontierStatistics",
    "TrainingDataGenerator",
    "TrainingDataWriter",
    "TrainingDatum",
    "bearing_to_target",
    "build_lookahead_grid",
    "compute_frontier_views",
    "compute_oracle_frontier_labels",
    "construct_lsp_explore_operator",
    "construct_move_to_goal_operator",
    "count_cells_in_polygon",
    "egocentric_xy",
    "frontier_cells_hash",
    "frontier_signature",
    "inspect_data",
    "is_goal_observed",
    "load_datum",
    "make_inspection_figure",
    "make_training_view",
    "mask_grid_with_frontiers",
    "read_index",
    "roll_pano_to_bearing",
    "select_best_vantage",
    "vantage_key",
    "wrap_angle",
]

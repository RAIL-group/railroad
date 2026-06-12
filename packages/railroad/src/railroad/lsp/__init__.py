"""Learning over subgoals planning (LSP).

Vision-based learning about frontier-exploration uncertainty for
point-goal navigation: oracle labeling of frontiers against the true
map, a frontier-exploration operator with per-frontier success
probability and branch-specific durations, and training-data generation
from panoramic images with best-vantage selection.

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
from .generator import TrainingDataGenerator
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
from .providers import (
    FrontierPropertyProvider,
    OptimisticFrontierPropertyProvider,
    OracleFrontierPropertyProvider,
)
from .types import (
    FrontierProperties,
    LSPDataConfig,
    OracleFrontierLabel,
    TrainingDatum,
)
from .vantage import count_cells_in_polygon, select_best_vantage

__all__ = [
    "FrontierChangeTracker",
    "FrontierProperties",
    "FrontierPropertyProvider",
    "LSPDataConfig",
    "LSPEnvironmentMixin",
    "OptimisticFrontierPropertyProvider",
    "OracleFrontierLabel",
    "OracleFrontierPropertyProvider",
    "TrainingDataGenerator",
    "TrainingDataWriter",
    "TrainingDatum",
    "bearing_to_target",
    "build_lookahead_grid",
    "compute_oracle_frontier_labels",
    "construct_lsp_explore_operator",
    "construct_move_to_goal_operator",
    "count_cells_in_polygon",
    "egocentric_xy",
    "frontier_cells_hash",
    "frontier_signature",
    "is_goal_observed",
    "load_datum",
    "make_training_view",
    "mask_grid_with_frontiers",
    "read_index",
    "roll_pano_to_bearing",
    "select_best_vantage",
    "vantage_key",
    "wrap_angle",
]

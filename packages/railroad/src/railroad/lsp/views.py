"""Pair frontiers with their best panoramic vantage as observations.

Shared by training-data generation and the learned frontier-statistics
estimator, so a model trained on the generated data sees exactly the
same inputs at planning time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Mapping, NamedTuple, Sequence

from railroad.experimental.unknown_search.types import Frontier

from .pano import make_training_view
from .types import FrontierObservation
from .vantage import select_best_vantage

if TYPE_CHECKING:
    from railroad.environment.railsim import PanoRecord


class FrontierView(NamedTuple):
    """A frontier's observation plus the pano record it was built from."""

    frontier_id: str
    observation: FrontierObservation
    record: "PanoRecord"


def compute_frontier_views(
    *,
    frontiers: Mapping[str, Frontier],
    pano_records: Sequence["PanoRecord"],
    goal_cell: tuple[int, int],
    vantage_inflation_radius: float = 1.0,
) -> Dict[str, FrontierView]:
    """Build one observation per frontier from its best vantage.

    Frontiers that no panorama has a view of are omitted.
    """
    goal_rc = (float(goal_cell[0]), float(goal_cell[1]))
    views: Dict[str, FrontierView] = {}
    for frontier_id, frontier in frontiers.items():
        record = select_best_vantage(
            frontier, pano_records, vantage_inflation_radius
        )
        if record is None:
            continue
        frontier_rc = (
            float(frontier.centroid_row), float(frontier.centroid_col)
        )
        image, frontier_xy, goal_xy = make_training_view(
            record, frontier_rc, goal_rc
        )
        views[frontier_id] = FrontierView(
            frontier_id=frontier_id,
            observation=FrontierObservation(
                image=image,
                frontier_xy_ego=frontier_xy,
                goal_xy_ego=goal_xy,
            ),
            record=record,
        )
    return views

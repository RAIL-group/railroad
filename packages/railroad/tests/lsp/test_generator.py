"""End-to-end tests of training-data emission with fake pano records."""

from __future__ import annotations


import numpy as np

from lsp.helpers import FakeRecord, make_frontier, square_polygon

from railroad.environment.types import Pose
from railroad.experimental.unknown_search.types import Frontier
from railroad.lsp import (
    OracleFrontierLabel,
    TrainingDataGenerator,
    TrainingDataWriter,
    frontier_cells_hash,
    read_index,
)


def _label(frontier: Frontier, feasible: bool) -> OracleFrontierLabel:
    return OracleFrontierLabel(
        frontier_id=frontier.id,
        prob_feasible=1.0 if feasible else 0.0,
        success_cost=20.0 if feasible else None,
        optimistic_cost=18.0 if feasible else None,
        exploration_cost=None if feasible else 6.0,
        cells_hash=frontier_cells_hash(frontier),
    )


def _record(time: float, pose_rc: tuple[float, float]) -> FakeRecord:
    rng = np.random.default_rng(int(time * 100))
    return FakeRecord(
        robot="robot1",
        time=time,
        pose_cells=Pose(pose_rc[0], pose_rc[1], 0.0),
        pose_meters=(0.0, 0.0, 0.0),
        image=rng.integers(0, 255, size=(4, 16, 3), dtype=np.uint8),
        visibility_polygon=square_polygon(0, 0, 20, 20),
    )


def test_generator_emits_on_change_only(tmp_path) -> None:  # noqa: ANN001
    writer = TrainingDataWriter(tmp_path)
    generator = TrainingDataGenerator(goal_cell=(50, 50), writer=writer)

    frontier = make_frontier("f1", [(5, 5), (5, 6)])
    label = _label(frontier, feasible=True)
    records = [_record(1.0, (10.0, 10.0))]

    assert generator.update(
        frontiers={"f1": frontier}, labels={"f1": label}, pano_records=records
    ) == 1
    # Same everything: no new datum.
    assert generator.update(
        frontiers={"f1": frontier}, labels={"f1": label}, pano_records=records
    ) == 0

    # Label flips: re-emit.
    failed = _label(frontier, feasible=False)
    assert generator.update(
        frontiers={"f1": frontier}, labels={"f1": failed}, pano_records=records
    ) == 1

    # A better vantage (closer to the frontier): re-emit.
    records.append(_record(2.0, (5.0, 7.0)))
    assert generator.update(
        frontiers={"f1": frontier}, labels={"f1": failed}, pano_records=records
    ) == 1

    assert generator.num_written == 3
    writer.close()

    index = read_index(tmp_path)
    assert [entry["label"] for entry in index] == [True, False, False]
    assert all(entry["frontier_id"] == "f1" for entry in index)


def test_generator_skips_unlabeled_or_unseen_frontiers(tmp_path) -> None:  # noqa: ANN001
    writer = TrainingDataWriter(tmp_path)
    generator = TrainingDataGenerator(goal_cell=(50, 50), writer=writer)

    seen = make_frontier("seen", [(5, 5)])
    unseen = make_frontier("unseen", [(100, 100)])  # outside every polygon
    unlabeled = make_frontier("unlabeled", [(6, 6)])
    records = [_record(1.0, (10.0, 10.0))]

    written = generator.update(
        frontiers={"seen": seen, "unseen": unseen, "unlabeled": unlabeled},
        labels={
            "seen": _label(seen, feasible=True),
            "unseen": _label(unseen, feasible=True),
        },
        pano_records=records,
    )
    assert written == 1
    assert read_index(tmp_path)[0]["frontier_id"] == "seen"
    writer.close()


def test_generator_without_writer_is_noop() -> None:
    generator = TrainingDataGenerator(goal_cell=(50, 50), writer=None)
    frontier = make_frontier("f1", [(5, 5)])
    assert generator.update(
        frontiers={"f1": frontier},
        labels={"f1": _label(frontier, feasible=True)},
        pano_records=[_record(1.0, (10.0, 10.0))],
    ) == 0
    assert generator.num_written == 0


def test_generator_datum_contents(tmp_path) -> None:  # noqa: ANN001
    from railroad.lsp import load_datum

    writer = TrainingDataWriter(tmp_path)
    generator = TrainingDataGenerator(goal_cell=(10, 30), writer=writer)

    frontier = make_frontier("f1", [(10, 18)])
    record = _record(1.0, (10.0, 10.0))
    generator.update(
        frontiers={"f1": frontier},
        labels={"f1": _label(frontier, feasible=False)},
        pano_records=[record],
    )
    writer.close()

    datum = load_datum(tmp_path / "datum_000000.npz")
    assert datum.image.shape == record.image.shape
    assert datum.label is False
    assert datum.exploration_cost == 6.0
    assert datum.success_cost is None
    # Frontier is straight ahead of the vantage (same row, 8 cells east);
    # the goal is along the same line, 20 cells out.
    assert datum.frontier_xy_ego[0] > 0
    assert abs(datum.frontier_xy_ego[1]) < 1e-6
    assert datum.goal_xy_ego[0] > datum.frontier_xy_ego[0]

"""Tests for training-data persistence and change detection."""

from __future__ import annotations

import json

import numpy as np
import pytest

from railroad.lsp import (
    FrontierChangeTracker,
    OracleFrontierLabel,
    TrainingDataWriter,
    TrainingDatum,
    frontier_signature,
    load_datum,
    read_index,
)


def _label(**overrides: object) -> OracleFrontierLabel:
    values: dict = dict(
        frontier_id="f1",
        prob_feasible=1.0,
        success_cost=12.34,
        optimistic_cost=10.0,
        exploration_cost=None,
        cells_hash="abc",
    )
    values.update(overrides)
    return OracleFrontierLabel(**values)


def _datum(label: bool = True) -> TrainingDatum:
    return TrainingDatum(
        image=np.arange(2 * 8 * 3, dtype=np.uint8).reshape(2, 8, 3),
        frontier_xy_ego=(3.0, -1.0),
        goal_xy_ego=(10.0, 4.0),
        label=label,
        success_cost=12.34 if label else None,
        optimistic_cost=10.0 if label else None,
        exploration_cost=None if label else 7.5,
        metadata={"frontier_id": "f1", "robot": "robot1", "time": 1.5},
    )


def test_signature_changes_with_label_fields() -> None:
    vantage = ("robot1", 1.0, 5.0, 5.0, 0.0)
    base = frontier_signature(_label(), vantage)
    assert frontier_signature(_label(), vantage) == base
    assert frontier_signature(_label(success_cost=99.0), vantage) != base
    assert frontier_signature(_label(prob_feasible=0.0), vantage) != base
    assert frontier_signature(_label(cells_hash="zzz"), vantage) != base
    assert frontier_signature(_label(), ("robot1", 2.0, 6.0, 5.0, 0.0)) != base
    assert frontier_signature(_label(), None) != base


def test_signature_rounds_costs() -> None:
    vantage = ("robot1", 1.0, 5.0, 5.0, 0.0)
    a = frontier_signature(_label(success_cost=12.34), vantage)
    b = frontier_signature(_label(success_cost=12.36), vantage)  # rounds to 12.3/12.4
    c = frontier_signature(_label(success_cost=12.31), vantage)  # rounds to 12.3
    assert a != b
    assert a == c


def test_change_tracker() -> None:
    tracker = FrontierChangeTracker()
    assert tracker.should_emit("f1", "sig-a")
    assert not tracker.should_emit("f1", "sig-a")
    assert tracker.should_emit("f1", "sig-b")
    assert tracker.should_emit("f2", "sig-a")

    # Pruning forgets dead frontiers, so a reappearing one re-emits.
    tracker.prune(["f2"])
    assert tracker.should_emit("f1", "sig-b")


def test_writer_roundtrip(tmp_path) -> None:  # noqa: ANN001
    with TrainingDataWriter(tmp_path / "data", {"seed": 7}) as writer:
        path_a = writer.write(_datum(label=True))
        path_b = writer.write(_datum(label=False))
        assert writer.num_written == 2

    datum_a = load_datum(path_a)
    np.testing.assert_array_equal(datum_a.image, _datum().image)
    assert datum_a.image.dtype == np.uint8
    assert datum_a.frontier_xy_ego == (3.0, -1.0)
    assert datum_a.goal_xy_ego == (10.0, 4.0)
    assert datum_a.label is True
    assert datum_a.success_cost == 12.34
    assert datum_a.exploration_cost is None  # NaN round-trips to None

    datum_b = load_datum(path_b)
    assert datum_b.label is False
    assert datum_b.success_cost is None
    assert datum_b.exploration_cost == 7.5

    index = read_index(tmp_path / "data")
    assert len(index) == 2
    assert index[0]["file"] == path_a.name
    assert index[0]["label"] is True
    assert index[0]["frontier_id"] == "f1"
    assert index[1]["label"] is False

    with open(tmp_path / "data" / "meta.json") as f:
        assert json.load(f) == {"seed": 7}


def test_write_after_close_raises(tmp_path) -> None:  # noqa: ANN001
    # A closed writer must not quietly reopen the index in append mode and
    # extend a run that was already finalized.
    with TrainingDataWriter(tmp_path / "data", {"seed": 7}) as writer:
        writer.write(_datum())

    with pytest.raises(ValueError, match="closed"):
        writer.write(_datum())

    # The rejected write leaves nothing behind -- no orphan npz, no index line.
    assert writer.num_written == 1
    assert len(read_index(tmp_path / "data")) == 1
    assert [p.name for p in sorted((tmp_path / "data").glob("*.npz"))] == [
        "datum_000000.npz"
    ]


def test_writer_with_no_data_leaves_empty_index(tmp_path) -> None:  # noqa: ANN001
    # index.jsonl is what marks a finished run, so a rollout that emits no
    # data must still leave one behind (empty) rather than no file at all.
    out_dir = tmp_path / "data"
    with TrainingDataWriter(out_dir, {"seed": 7}):
        pass

    assert (out_dir / "index.jsonl").exists()
    assert read_index(out_dir) == []


def test_read_index_missing_dir(tmp_path) -> None:  # noqa: ANN001
    assert read_index(tmp_path / "nope") == []

"""Tests for MLflow experiment handling (railroad.bench.tracking).

Both cases here are only reachable through ``benchmarks run --experiment``,
which names an experiment explicitly and so can collide with one that already
exists -- the timestamped default names never do.
"""

import mlflow
import pytest

from railroad.bench.tracking import MLflowTracker


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return MLflowTracker(tracking_uri=uri)


def test_named_experiment_is_reused_across_runs(tracker):
    """Successive sweeps accumulate, which is what makes one dashboard page."""
    tracker.create_experiment("acc", {"run_name": "first"})
    first = tracker.experiment_id
    tracker.create_experiment("acc", {"run_name": "second"})
    assert tracker.experiment_id == first


def test_a_deleted_experiment_is_restored_not_written_into(tracker):
    """MLflow returns soft-deleted experiments by name and accepts runs into
    them, where nothing can find the results again."""
    tracker.create_experiment("gone", {})
    experiment_id = tracker.experiment_id
    mlflow.tracking.MlflowClient().delete_experiment(experiment_id)
    assert mlflow.get_experiment(experiment_id).lifecycle_stage == "deleted"

    tracker.create_experiment("gone", {})

    assert mlflow.get_experiment(experiment_id).lifecycle_stage == "active"
    with mlflow.start_run(experiment_id=tracker.experiment_id):
        mlflow.log_metric("x", 1.0)
    assert len(mlflow.search_runs([experiment_id])) == 1

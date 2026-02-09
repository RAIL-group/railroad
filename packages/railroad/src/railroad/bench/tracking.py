"""
MLflow integration layer for experiment tracking.

Provides centralized logging interface for benchmark results.
"""

import sys
from io import StringIO
from typing import Dict, Any, Optional
from pathlib import Path
import gzip
import json
import tempfile

# Suppress MLflow database initialization logs by redirecting stderr during import
_stderr = sys.stderr
sys.stderr = StringIO()
import mlflow  # noqa: E402
sys.stderr = _stderr

from .plan import Task, TaskStatus  # noqa: E402


class MLflowTracker:
    """
    Centralized MLflow logging interface.

    Handles experiment creation and task result logging.

    Experiment hierarchy:
        Experiment: railroad_bench_{timestamp}
            └─ Run (per task): {benchmark_name}_{case_idx}_{repeat_idx}
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking URI. Defaults to sqlite:///mlflow.db
        """
        # Suppress database initialization logs
        _stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                # Default: SQLite database (avoids filesystem deprecation warning)
                mlflow.set_tracking_uri("sqlite:///mlflow.db")
        finally:
            sys.stderr = _stderr

        self.experiment_id = None
        self.experiment_name = None

    def create_experiment(self, name: str, metadata: Dict[str, Any]):
        """
        Create new experiment with metadata tags.

        Args:
            name: Experiment name (e.g., railroad_bench_1234567890)
            metadata: Metadata dictionary to log as experiment tags
        """
        # Suppress database initialization logs
        _stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            self.experiment_name = name

            # Create or get existing experiment
            experiment = mlflow.get_experiment_by_name(name)  # type: ignore[possibly-missing-attribute]
            if experiment:
                self.experiment_id = experiment.experiment_id
            else:
                # Convert metadata to string tags (MLflow only supports string tags)
                tags = {k: str(v) for k, v in metadata.items()}
                self.experiment_id = mlflow.create_experiment(  # type: ignore[possibly-missing-attribute]
                    name=name,
                    tags=tags,
                )

            mlflow.set_experiment(experiment_id=self.experiment_id)
        finally:
            sys.stderr = _stderr

    def log_task(self, task: Task):
        """
        Log a single task execution as an MLflow run.

        Args:
            task: Completed task with results
        """
        with mlflow.start_run(run_name=task.id):  # type: ignore[possibly-missing-attribute]
            # Log parameters (inputs)
            params = {
                "benchmark_name": task.benchmark_name,
                "case_idx": str(task.case_idx),
                "repeat_idx": str(task.repeat_idx),
            }
            # Add benchmark parameters with param_ prefix
            for key, value in task.params.items():
                params[str(key)] = str(value)

            mlflow.log_params(params)  # type: ignore[possibly-missing-attribute]

            # Log tags
            tags = {
                "status": task.status.value,
            }
            # Add benchmark tags
            for i, tag in enumerate(task.tags):
                tags[f"tag_{i}"] = tag

            mlflow.set_tags(tags)  # type: ignore[possibly-missing-attribute]

            # Log metrics (outputs)
            metrics = {}

            if task.result:
                # Separate scalar metrics from complex data
                artifacts = {}

                for key, value in task.result.items():
                    if isinstance(value, (int, float, bool)):
                        metrics[key] = float(value)
                    elif isinstance(value, str) and len(value) < 500:
                        # Short strings can be logged as params
                        mlflow.log_param(f"result_{key}", value)  # type: ignore[possibly-missing-attribute]
                    else:
                        # Complex data goes to artifacts
                        artifacts[key] = value

                # Log artifacts to temp files
                if artifacts:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        for key, value in artifacts.items():
                            # Special handling for HTML artifacts (gzipped to save space)
                            if key == 'log_html':
                                artifact_path = Path(tmpdir) / "log.html.gz"
                                with gzip.open(artifact_path, 'wt', compresslevel=6) as f:
                                    f.write(str(value))
                                mlflow.log_artifact(str(artifact_path))  # type: ignore[possibly-missing-attribute]
                            elif key == 'log_plot':
                                artifact_path = Path(tmpdir) / "plot.jpg"
                                with open(artifact_path, 'wb') as f:
                                    f.write(value)
                                mlflow.log_artifact(str(artifact_path))  # type: ignore[possibly-missing-attribute]
                            else:
                                # Regular artifacts as JSON
                                artifact_path = Path(tmpdir) / f"{key}.json"
                                with open(artifact_path, 'w') as f:
                                    json.dump(value, f, indent=2, default=str)
                                mlflow.log_artifact(str(artifact_path))  # type: ignore[possibly-missing-attribute]

            # Always log wall time and success status
            if task.wall_time is not None:
                metrics["wall_time"] = task.wall_time

            metrics["success"] = 1.0 if task.status == TaskStatus.SUCCESS else 0.0
            metrics["timeout"] = 1.0 if task.status == TaskStatus.TIMEOUT else 0.0

            mlflow.log_metrics(metrics)  # type: ignore[possibly-missing-attribute]

            # Log stdout/stderr as artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                if task.stdout:
                    stdout_path = Path(tmpdir) / "stdout.txt"
                    with open(stdout_path, 'w') as f:
                        f.write(task.stdout)
                    mlflow.log_artifact(str(stdout_path))  # type: ignore[possibly-missing-attribute]

                if task.stderr:
                    stderr_path = Path(tmpdir) / "stderr.txt"
                    with open(stderr_path, 'w') as f:
                        f.write(task.stderr)
                    mlflow.log_artifact(str(stderr_path))  # type: ignore[possibly-missing-attribute]

            # Log error message if failed
            if task.error:
                mlflow.log_param("error_message", task.error[:500])  # type: ignore[possibly-missing-attribute]

    def log_summary(self, summary: Dict[str, Any]):
        """
        Log experiment-level summary as a special run.

        Args:
            summary: Summary statistics dictionary
        """
        with mlflow.start_run(run_name="__summary__"):  # type: ignore[possibly-missing-attribute]
            # Log all summary values as metrics
            metrics = {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(metrics)  # type: ignore[possibly-missing-attribute]

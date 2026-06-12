"""Tests for bulk training-data generation (GL-free via injected rollouts)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from railroad.lsp import TrainingDataWriter, TrainingDatum, read_index
from railroad.lsp.bulk import (
    RolloutResult,
    completed_seeds,
    generate_data,
    parse_seeds,
    seed_dir_name,
)


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


# Module-level so it pickles by reference into spawn workers.
def _fake_rollout(
    env_name: str,
    seed: int,
    save_data_dir: str | Path,
    *,
    frontier_statistics_name: str,
    prior_prob: float,
    max_planning_iterations: int,
) -> RolloutResult:
    with TrainingDataWriter(
        save_data_dir, {"env": env_name, "seed": seed}
    ) as writer:
        writer.write(_datum(label=True))
        writer.write(_datum(label=False))
    return RolloutResult(
        seed=seed,
        env_name=env_name,
        goal_reached=True,
        termination="goal_reached",
        sim_time=20.0,
        num_data_written=2,
        num_panoramas=5,
        wall_time=0.1,
    )


def _fake_rollout_no_goal(
    env_name: str,
    seed: int,
    save_data_dir: str | Path,
    **kwargs: object,
) -> RolloutResult:
    result = _fake_rollout(
        env_name,
        seed,
        save_data_dir,
        frontier_statistics_name="oracle",
        prior_prob=0.8,
        max_planning_iterations=200,
    )
    result.goal_reached = False
    result.termination = "max_iterations"
    return result


def _failing_rollout(
    env_name: str,
    seed: int,
    save_data_dir: str | Path,
    **kwargs: object,
) -> RolloutResult:
    # Leave a partial write behind, like a real mid-rollout crash.
    writer = TrainingDataWriter(save_data_dir, {"seed": seed})
    writer.write(_datum())
    raise RuntimeError("rollout exploded")


def test_parse_seeds() -> None:
    assert parse_seeds("0:5") == [0, 1, 2, 3, 4]
    assert parse_seeds("3,1,3") == [1, 3]
    assert parse_seeds("7") == [7]
    assert parse_seeds(" 2 , 4 ") == [2, 4]

    with pytest.raises(ValueError, match="No seeds"):
        parse_seeds("")
    with pytest.raises(ValueError, match="Invalid seed range"):
        parse_seeds("a:b")
    with pytest.raises(ValueError, match="Empty seed range"):
        parse_seeds("5:5")
    with pytest.raises(ValueError, match="Invalid seed"):
        parse_seeds("1,x")


def test_completed_seeds_scans_dirs(tmp_path: Path) -> None:
    assert completed_seeds(tmp_path / "missing") == set()

    assert seed_dir_name(7) == "seed_00007"
    (tmp_path / "seed_00007").mkdir()
    (tmp_path / "seed_123456").mkdir()  # wider than the padding still parses
    (tmp_path / ".tmp").mkdir()
    (tmp_path / ".tmp" / "seed_00009").mkdir()  # in-flight, not complete
    (tmp_path / "seed_xyz").mkdir()
    (tmp_path / "experiment.json").touch()

    assert completed_seeds(tmp_path) == {7, 123456}


def test_generate_data_happy_path(tmp_path: Path) -> None:
    summary = generate_data(
        [0, 1, 2],
        env_name="maze",
        data_dir=tmp_path,
        experiment_name="exp",
        console=Console(quiet=True),
        _rollout_fn=_fake_rollout,
    )

    assert summary.requested == 3
    assert summary.skipped == 0
    assert summary.succeeded == 3
    assert summary.failed == 0
    assert summary.total_data == 6

    exp_dir = tmp_path / "exp"
    assert json.loads((exp_dir / "experiment.json").read_text())["env"] == "maze"
    for seed in (0, 1, 2):
        seed_dir = exp_dir / seed_dir_name(seed)
        assert len(read_index(seed_dir)) == 2
        meta = json.loads((seed_dir / "meta.json").read_text())
        assert meta["seed"] == seed
        assert meta["outcome"]["goal_reached"] is True
        assert meta["outcome"]["num_data"] == 2
    assert not any((exp_dir / ".tmp").iterdir())


def test_generate_data_skips_completed(tmp_path: Path) -> None:
    console = Console(quiet=True)
    generate_data(
        [0, 1],
        data_dir=tmp_path,
        experiment_name="exp",
        console=console,
        _rollout_fn=_fake_rollout,
    )

    # A second invocation must not re-run anything: a rollout fn that
    # would fail proves the completed seeds are never attempted.
    summary = generate_data(
        [0, 1],
        data_dir=tmp_path,
        experiment_name="exp",
        console=console,
        _rollout_fn=_failing_rollout,
    )
    assert summary.skipped == 2
    assert summary.succeeded == 0
    assert summary.failed == 0

    # New seeds in the range still run.
    summary = generate_data(
        [0, 1, 2],
        data_dir=tmp_path,
        experiment_name="exp",
        console=console,
        _rollout_fn=_fake_rollout,
    )
    assert summary.skipped == 2
    assert summary.succeeded == 1


def test_failed_seed_leaves_no_final_dir_and_retries_cleanly(
    tmp_path: Path,
) -> None:
    console = Console(quiet=True)
    summary = generate_data(
        [4],
        data_dir=tmp_path,
        experiment_name="exp",
        console=console,
        _rollout_fn=_failing_rollout,
    )
    assert summary.failed == 1
    (result,) = summary.results
    assert not result.ok
    assert result.error is not None and "rollout exploded" in result.error

    exp_dir = tmp_path / "exp"
    assert not (exp_dir / seed_dir_name(4)).exists()
    # The partial tmp dir is left behind for diagnosis...
    assert (exp_dir / ".tmp" / seed_dir_name(4) / "index.jsonl").exists()

    # ...and the retry deletes it before writing, so the (append-mode)
    # index has no duplicate lines from the failed attempt.
    summary = generate_data(
        [4],
        data_dir=tmp_path,
        experiment_name="exp",
        console=console,
        _rollout_fn=_fake_rollout,
    )
    assert summary.succeeded == 1
    assert len(read_index(exp_dir / seed_dir_name(4))) == 2


def test_non_goal_termination_still_finalizes(tmp_path: Path) -> None:
    summary = generate_data(
        [0],
        data_dir=tmp_path,
        experiment_name="exp",
        console=Console(quiet=True),
        _rollout_fn=_fake_rollout_no_goal,
    )
    assert summary.succeeded == 1
    (result,) = summary.results
    assert result.ok and not result.goal_reached

    meta = json.loads(
        (tmp_path / "exp" / seed_dir_name(0) / "meta.json").read_text()
    )
    assert meta["outcome"]["goal_reached"] is False
    assert meta["outcome"]["termination"] == "max_iterations"


def test_config_mismatch_warns(tmp_path: Path) -> None:
    generate_data(
        [0],
        data_dir=tmp_path,
        experiment_name="exp",
        prior_prob=0.8,
        console=Console(quiet=True),
        _rollout_fn=_fake_rollout,
    )

    # quiet=True would suppress recording as well; route output to a
    # throwaway buffer instead.
    console = Console(record=True, file=io.StringIO(), width=400)
    generate_data(
        [1],
        data_dir=tmp_path,
        experiment_name="exp",
        prior_prob=0.5,
        console=console,
        _rollout_fn=_fake_rollout,
    )
    text = console.export_text()
    assert "experiment config differs" in text
    assert "prior_prob" in text


def test_parallel_workers_smoke(tmp_path: Path) -> None:
    summary = generate_data(
        [0, 1, 2, 3],
        data_dir=tmp_path,
        experiment_name="exp",
        parallel=2,
        console=Console(quiet=True),
        _rollout_fn=_fake_rollout,
    )
    assert summary.succeeded == 4
    assert completed_seeds(tmp_path / "exp") == {0, 1, 2, 3}

"""Bulk LSP training-data generation: many seeds, parallel, resumable.

One rollout per seed, each writing a standard
:class:`~railroad.lsp.TrainingDataWriter` directory under
``<data_dir>/<experiment_name>/seed_<NNNNN>/`` (so ``railroad lsp
inspect-data`` works on any of them). Robustness properties:

- **Atomic per-seed output**: each rollout writes into
  ``<experiment>/.tmp/seed_<NNNNN>/`` and is renamed into place only on
  clean termination, so a half-completed seed never leaves transient
  data in the experiment directory.
- **Resumable**: seeds whose final directory exists are skipped, so an
  interrupted invocation can simply be re-run.

This module is GL-free at import time; the railsim-dependent rollout is
imported inside the worker, which runs in its own process (each worker
creates and releases its own GL context).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Sequence, Set

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from railroad.bench.capture import capture_output

_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


@dataclass
class RolloutResult:
    """Outcome of one headless rollout (picklable; plain primitives only).

    Defined here (GL-free) rather than in :mod:`railroad.lsp.rollout` so
    orchestration and tests can use it without the railsim dependency;
    the real rollout returns one of these.
    """

    seed: int
    env_name: str
    goal_reached: bool
    termination: str  # goal_reached | no_actions | planner_none | max_iterations
    sim_time: float
    num_data_written: int
    num_panoramas: int
    wall_time: float
    error: str | None = None


def parse_seeds(spec: str) -> List[int]:
    """Parse a seed spec: ``"a:b"`` (half-open range), ``"1,5,9"``, or ``"42"``."""
    spec = spec.strip()
    if ":" in spec:
        start_str, _, end_str = spec.partition(":")
        try:
            start, end = int(start_str), int(end_str)
        except ValueError:
            raise ValueError(f"Invalid seed range {spec!r}; expected 'start:end'")
        if start >= end:
            raise ValueError(
                f"Empty seed range {spec!r}; 'start:end' is half-open with start < end"
            )
        return list(range(start, end))
    seeds = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            seeds.add(int(token))
        except ValueError:
            raise ValueError(f"Invalid seed {token!r} in {spec!r}")
    if not seeds:
        raise ValueError(f"No seeds in {spec!r}")
    return sorted(seeds)


def seed_dir_name(seed: int) -> str:
    return f"seed_{seed:05d}"


def completed_seeds(exp_dir: Path) -> Set[int]:
    """Seeds with a finalized output directory under *exp_dir*."""
    if not exp_dir.is_dir():
        return set()
    found = set()
    for child in exp_dir.iterdir():
        match = _SEED_DIR_RE.match(child.name)
        if match and child.is_dir():
            found.add(int(match.group(1)))
    return found


@dataclass
class SeedTask:
    """One seed's worth of work, picklable for process workers."""

    seed: int
    env_name: str
    frontier_statistics_name: str
    prior_prob: float
    max_planning_iterations: int
    exp_dir: str
    # None = the real GL rollout, imported lazily in the worker. Tests
    # inject a module-level fake (pickled by reference under spawn).
    rollout_fn: Callable[..., "RolloutResult"] | None = None


@dataclass
class SeedResult:
    seed: int
    ok: bool
    goal_reached: bool = False
    termination: str = ""
    num_data: int = 0
    sim_time: float = 0.0
    wall_time: float = 0.0
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


def _seed_worker(task: SeedTask) -> SeedResult:
    """Run one seed: rollout into a tmp dir, finalize atomically.

    Never raises on rollout failure (returns ``ok=False`` instead) so a
    bad seed doesn't take down the pool; KeyboardInterrupt propagates.
    """
    exp_dir = Path(task.exp_dir)
    tmp_dir = exp_dir / ".tmp" / seed_dir_name(task.seed)
    final_dir = exp_dir / seed_dir_name(task.seed)
    t0 = time.perf_counter()
    captured_stdout = ""
    captured_stderr = ""
    try:
        # A stale tmp dir from an interrupted run must be deleted, not
        # reused: TrainingDataWriter appends to index.jsonl, so reuse
        # would leave duplicate index lines.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.parent.mkdir(parents=True, exist_ok=True)

        rollout_fn = task.rollout_fn
        if rollout_fn is None:
            # GL import — worker process only, never the orchestrator.
            from .rollout import run_point_goal_rollout
            rollout_fn = run_point_goal_rollout

        with capture_output() as captured:
            try:
                result = rollout_fn(
                    task.env_name,
                    task.seed,
                    tmp_dir,
                    frontier_statistics_name=task.frontier_statistics_name,
                    prior_prob=task.prior_prob,
                    max_planning_iterations=task.max_planning_iterations,
                )
            finally:
                captured_stdout = captured.stdout
                captured_stderr = captured.stderr

        # Record the outcome in the seed's meta.json (writer is closed),
        # then atomically move the directory into place. Clean non-goal
        # terminations still finalize — oracle labels are valid training
        # data regardless — and training can filter on the outcome.
        meta_path = tmp_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta["outcome"] = {
            "goal_reached": result.goal_reached,
            "termination": result.termination,
            "sim_time": result.sim_time,
            "num_data": result.num_data_written,
            "wall_time": time.perf_counter() - t0,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        # Atomic because .tmp lives inside the experiment directory
        # (same filesystem).
        os.replace(tmp_dir, final_dir)
        return SeedResult(
            seed=task.seed,
            ok=True,
            goal_reached=result.goal_reached,
            termination=result.termination,
            num_data=result.num_data_written,
            sim_time=result.sim_time,
            wall_time=time.perf_counter() - t0,
        )
    except KeyboardInterrupt:
        raise
    except BaseException:
        # The tmp dir is intentionally left behind; the next attempt of
        # this seed deletes it first.
        return SeedResult(
            seed=task.seed,
            ok=False,
            error=traceback.format_exc(),
            stdout=captured_stdout,
            stderr=captured_stderr,
            wall_time=time.perf_counter() - t0,
        )


_EXPERIMENT_CONFIG_KEYS = (
    "env",
    "frontier_statistics",
    "prior_prob",
    "max_planning_iterations",
)


def _check_experiment_config(
    exp_dir: Path, config: dict, console: Console
) -> None:
    """Write experiment.json once; warn if a resume's config differs."""
    path = exp_dir / "experiment.json"
    if not path.exists():
        path.write_text(json.dumps(config, indent=2))
        return
    existing = json.loads(path.read_text())
    differing = [
        key for key in _EXPERIMENT_CONFIG_KEYS
        if existing.get(key) != config.get(key)
    ]
    if differing:
        console.print(
            "[yellow]Warning: experiment config differs from "
            f"{path} on: {', '.join(differing)} — continuing anyway.[/yellow]"
        )


@dataclass
class GenerateSummary:
    requested: int
    skipped: int
    succeeded: int
    failed: int
    total_data: int
    results: List[SeedResult] = field(default_factory=list)


def generate_data(
    seeds: Sequence[int],
    *,
    env_name: str = "maze",
    data_dir: str | Path = "data",
    experiment_name: str | None = None,
    parallel: int = 1,
    frontier_statistics_name: str = "oracle",
    prior_prob: float = 0.8,
    max_planning_iterations: int = 200,
    console: Console | None = None,
    _rollout_fn: Callable[..., "RolloutResult"] | None = None,
) -> GenerateSummary:
    """Generate training data for *seeds*, skipping completed ones."""
    console = console or Console()
    exp_dir = Path(data_dir) / (experiment_name or env_name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    _check_experiment_config(
        exp_dir,
        {
            "env": env_name,
            "frontier_statistics": frontier_statistics_name,
            "prior_prob": prior_prob,
            "max_planning_iterations": max_planning_iterations,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        console,
    )

    done = completed_seeds(exp_dir)
    pending = [s for s in seeds if s not in done]
    skipped = len(seeds) - len(pending)
    if skipped:
        console.print(f"Skipping {skipped} already-completed seed(s).")
    if not pending:
        console.print("Nothing to do — all requested seeds are complete.")
        return GenerateSummary(len(seeds), skipped, 0, 0, 0)

    tasks = [
        SeedTask(
            seed=seed,
            env_name=env_name,
            frontier_statistics_name=frontier_statistics_name,
            prior_prob=prior_prob,
            max_planning_iterations=max_planning_iterations,
            exp_dir=str(exp_dir),
            rollout_fn=_rollout_fn,
        )
        for seed in pending
    ]

    results: List[SeedResult] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task(
            f"{env_name} → {exp_dir}", total=len(pending)
        )

        def handle(result: SeedResult) -> None:
            if result.ok:
                tag = (
                    "[green]done[/green]"
                    if result.goal_reached
                    else f"[yellow]{result.termination}[/yellow]"
                )
                progress.console.log(
                    f"seed {result.seed:>6} {tag}  "
                    f"{result.num_data} data  {result.wall_time:.0f}s"
                )
            else:
                progress.console.log(f"[red]seed {result.seed} FAILED[/red]")
                if result.error:
                    progress.console.log(result.error.rstrip())
                if result.stdout.strip():
                    progress.console.log(
                        f"--- captured stdout ---\n{result.stdout.rstrip()}"
                    )
            results.append(result)
            progress.advance(bar)

        if parallel <= 1:
            for task in tasks:
                handle(_seed_worker(task))
        else:
            with ProcessPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(_seed_worker, task): task for task in tasks
                }
                try:
                    for future in as_completed(futures):
                        try:
                            handle(future.result())
                        except Exception as e:  # worker process crash
                            crashed = futures[future]
                            handle(SeedResult(
                                seed=crashed.seed,
                                ok=False,
                                error=f"worker crashed: {e}",
                            ))
                except KeyboardInterrupt:
                    # Workers get SIGINT via the process group and die
                    # mid-rollout, leaving .tmp/seed_* dirs — exactly the
                    # designed recovery path (cleaned on retry).
                    for future in futures:
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    progress.stop()
                    console.print(
                        f"[yellow]Interrupted — {len(results)}/{len(pending)} "
                        "pending seed(s) finalized; re-run the same command "
                        "to resume.[/yellow]"
                    )
                    raise

    succeeded = sum(1 for r in results if r.ok)
    failed = len(results) - succeeded
    total_data = sum(r.num_data for r in results if r.ok)
    style = "red" if failed else "green"
    console.print(
        f"[{style}]{succeeded} seed(s) succeeded, {failed} failed, "
        f"{skipped} skipped — {total_data} new training data in "
        f"{exp_dir}[/{style}]"
    )
    return GenerateSummary(
        requested=len(seeds),
        skipped=skipped,
        succeeded=succeeded,
        failed=failed,
        total_data=total_data,
        results=results,
    )

"""Railroad command-line interface."""

from typing import Any

import rich_click as click

from railroad.examples import ExampleInfo


class _HelpfulCommand(click.RichCommand):
    """RichCommand that shows full help on usage errors."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        try:
            return super().parse_args(ctx, args)
        except click.UsageError as e:
            print(ctx.get_help(), end="")
            e.ctx = None  # suppress duplicate Usage line in error output
            raise


class _HelpfulGroup(click.RichGroup):
    """RichGroup that shows full help on usage errors.

    Sets ``command_class`` and ``group_class`` so that all subcommands and
    subgroups created under this group inherit the same behaviour.
    """

    command_class = _HelpfulCommand
    group_class = type  # sentinel: subgroups reuse this class

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        try:
            return super().parse_args(ctx, args)
        except click.UsageError as e:
            print(ctx.get_help(), end="")
            e.ctx = None  # suppress duplicate Usage line in error output
            raise


@click.group(cls=_HelpfulGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="railroad")
def main() -> None:
    """Railroad: Multi-Robot Probabilistic Planning."""
    pass


@main.group(invoke_without_command=True)
@click.pass_context
def example(ctx: click.Context) -> None:
    """Run example planning scenarios."""
    if ctx.invoked_subcommand is None:
        # No subcommand given - list examples
        from railroad.examples import EXAMPLES

        click.echo("Available examples:\n")
        for name, info in EXAMPLES.items():
            click.echo(f"  {name:24} {info['description']}")
        click.echo("\nRun an example with: railroad example <name>")


# =============================================================================
# Benchmarks command group
# =============================================================================


@main.group()
def benchmarks() -> None:
    """Run and analyze benchmarks."""
    pass


@benchmarks.command("run")
@click.option("-k", "--filter", "case_filter", default=None,
              help="Filter cases using pytest-style expressions (e.g., 'movie_night and mcts_iterations=400')")
@click.option("--repeat-max", type=int, default=None,
              help="Maximum number of repeats per case")
@click.option("--parallel", type=int, default=None,
              help="Number of parallel workers (default: auto-detect CPU count)")
@click.option("--tags", multiple=True, default=None,
              help="Filter benchmarks by tags")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show plan without executing")
@click.option("--mlflow-uri", default=None,
              help="MLflow tracking URI (default: sqlite:///mlflow.db)")
@click.option("--include", "-i", multiple=True, type=click.Path(exists=True),
              help="Include benchmark file(s) in addition to entry points (can be repeated)")
@click.option("--run-name", default=None,
              help="Human-readable name for this benchmark run (used in MLflow experiment name)")
def benchmarks_run(
    case_filter: str | None,
    repeat_max: int | None,
    parallel: int | None,
    tags: tuple[str, ...] | None,
    dry_run: bool,
    mlflow_uri: str | None,
    include: tuple[str, ...],
    run_name: str | None,
) -> None:
    """Run PDDL planning benchmarks."""
    import sys
    from railroad.bench.discovery import discover_benchmarks
    from railroad.bench.runner import BenchmarkRunner

    # Discover all benchmarks via entry points and included files
    include_files = list(include) if include else None
    all_benchmarks = discover_benchmarks(include_files=include_files)

    if not all_benchmarks:
        click.echo("Error: No benchmarks found. Make sure your benchmarks use the @benchmark decorator.")
        sys.exit(1)

    # Convert tags tuple to list if provided
    tags_list = list(tags) if tags else None

    # Create runner
    runner = BenchmarkRunner(
        benchmarks=all_benchmarks,
        repeat_max=repeat_max,
        parallel=parallel,
        mlflow_tracking_uri=mlflow_uri,
        tags=tags_list,
        case_filter=case_filter,
        include_files=include_files,
        run_name=run_name,
    )

    # Create plan
    plan = runner.create_plan()

    if not plan.tasks:
        click.echo("No tasks to run. Check your filters.")
        sys.exit(0)

    if dry_run:
        runner.dry_run(plan)
    else:
        runner.run(plan)


@benchmarks.command("dashboard")
def benchmarks_dashboard() -> None:
    """Launch the benchmark visualization dashboard."""
    from railroad.bench.dashboard.app import main as run_dashboard
    run_dashboard()


@benchmarks.command("compact")
@click.option("--experiment", default=None,
              help="Compact a single experiment by name (default: compact all eligible)")
@click.option("--force", is_flag=True, default=False,
              help="Invalidate existing caches before recomputing")
@click.option("--mlflow-uri", default=None,
              help="MLflow tracking URI (default: sqlite:///mlflow.db)")
def benchmarks_compact(experiment: str | None, force: bool, mlflow_uri: str | None) -> None:
    """Materialize the compaction cache for finished experiments."""
    from railroad.bench.analysis import BenchmarkAnalyzer
    from railroad.bench import compact

    analyzer = BenchmarkAnalyzer(tracking_uri=mlflow_uri)

    if experiment:
        names = [experiment]
    else:
        exps = analyzer.list_experiments()
        if exps.empty:
            click.echo("No experiments found.")
            return
        names = exps["name"].tolist()

    from railroad.bench.dashboard.figures import create_violin_plots_by_benchmark
    from railroad.bench.dashboard.sweeps import create_all_sweep_plots

    cached = 0
    skipped = 0
    failed = 0
    for name in names:
        if force:
            compact.invalidate(name)
        try:
            df = analyzer.load_experiment(name)
            metadata = analyzer.get_experiment_metadata(name)
            summary = analyzer.get_experiment_summary(name, df=df)
            figures = {
                "violin": create_violin_plots_by_benchmark(df),
                "sweep": create_all_sweep_plots(df),
            }
            if compact.save(name, df, metadata, summary, figures=figures):
                click.echo(f"  ✓ {name} ({summary['total_runs']} runs, figures cached)")
                cached += 1
            else:
                click.echo(f"  ⏭  {name} (in-progress, skipped)")
                skipped += 1
        except Exception as e:
            click.echo(f"  ✗ {name}: {e}")
            failed += 1
    click.echo(f"\nCompacted {cached} experiment(s), skipped {skipped}, failed {failed}.")


@benchmarks.command("cache-clear")
@click.option("--experiment", default=None,
              help="Clear cache for a single experiment by name (default: clear all)")
def benchmarks_cache_clear(experiment: str | None) -> None:
    """Remove compacted cache files."""
    from railroad.bench import compact
    if experiment:
        compact.invalidate(experiment)
        click.echo(f"Cleared cache for '{experiment}'.")
    else:
        compact.invalidate_all()
        click.echo("Cleared all benchmark caches.")


@benchmarks.command("cache-flush")
@click.option("--experiment", default=None,
              help="Flush cache for a single experiment by name (default: flush all)")
def benchmarks_cache_flush(experiment: str | None) -> None:
    """Fully reset benchmark caches.

    Like ``cache-clear``, but also removes the per-experiment stamp files, so
    the cache fingerprint baseline is rebuilt from scratch on the next load. A
    running dashboard rebuilds lazily on its next request.
    """
    from railroad.bench import compact
    if experiment:
        compact.invalidate(experiment)
        compact.remove_stamp(experiment)
        click.echo(f"Flushed cache and stamp for '{experiment}'.")
    else:
        compact.invalidate_all()
        compact.remove_all_stamps()
        click.echo("Flushed all benchmark caches and stamps.")


# =============================================================================
# LSP command group
# =============================================================================


@main.group()
def lsp() -> None:
    """Learning over subgoals planning (LSP) utilities."""
    pass


@lsp.command("inspect-data")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--num", type=int, default=6, show_default=True,
              help="Number of data to plot, sampled evenly through the run")
@click.option("--indices", type=str, default=None,
              help="Comma-separated datum indices to plot (overrides --num)")
@click.option("--save", "save_path", type=str, default=None,
              help="Output figure path (default: <data_dir>/inspect.png)")
@click.option("--show", is_flag=True, default=False,
              help="Show the figure interactively")
def lsp_inspect_data(
    data_dir: str,
    num: int,
    indices: str | None,
    save_path: str | None,
    show: bool,
) -> None:
    """Summarize and visualize LSP training data in DATA_DIR.

    Prints label balance and cost statistics, then renders sampled data:
    each row shows the frontier-centered panorama (frontier and goal
    directions marked) next to a top-down egocentric view.
    """
    from railroad.lsp import inspect_data

    parsed_indices = None
    if indices is not None:
        parsed_indices = [int(token) for token in indices.split(",") if token.strip()]
    inspect_data(
        data_dir,
        num=num,
        indices=parsed_indices,
        save_path=save_path,
        show=show,
    )


@lsp.command("generate-data")
@click.option("--seeds", "seeds_spec", required=True,
              help="Seed range 'a:b' (half-open), comma list '1,5,9', or a single seed")
@click.option("--env", "env_name", type=click.Choice(["maze", "office"]),
              default="maze", show_default=True,
              help="Environment to generate scenes from")
@click.option("--data-dir", type=click.Path(file_okay=False), default="data",
              show_default=True, help="Root directory for generated data")
@click.option("--experiment-name", default=None,
              help="Subdirectory under --data-dir [default: the env name]")
@click.option("--parallel", "-j", type=int, default=None,
              help="Worker processes; each renders its own GL context "
                   "[default: cpu_count - 2]")
@click.option("--frontier-statistics", "frontier_statistics_name",
              type=click.Choice(["oracle", "fixed-prior"]), default="oracle",
              show_default=True,
              help="Estimator steering the rollouts (labels always come from the oracle)")
@click.option("--prior-prob", type=float, default=0.8, show_default=True,
              help="prob_feasible for the fixed-prior estimator")
@click.option("--max-iterations", "max_planning_iterations", type=int,
              default=200, show_default=True,
              help="Planning iterations per rollout before giving up")
def lsp_generate_data(
    seeds_spec: str,
    env_name: str,
    data_dir: str,
    experiment_name: str | None,
    parallel: int | None,
    frontier_statistics_name: str,
    prior_prob: float,
    max_planning_iterations: int,
) -> None:
    """Bulk-generate LSP training data: one rollout per seed, in parallel.

    Each seed's data is finalized atomically into
    <data-dir>/<experiment-name>/seed_<SEED>/ (a directory `railroad lsp
    inspect-data` understands). Already-completed seeds are skipped, so
    an interrupted invocation can simply be re-run.
    """
    import os

    from railroad.lsp.bulk import generate_data, parse_seeds

    try:
        seeds = parse_seeds(seeds_spec)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--seeds")
    if parallel is None:
        parallel = max((os.cpu_count() or 2) - 2, 1)

    summary = generate_data(
        seeds,
        env_name=env_name,
        data_dir=data_dir,
        experiment_name=experiment_name,
        parallel=parallel,
        frontier_statistics_name=frontier_statistics_name,
        prior_prob=prior_prob,
        max_planning_iterations=max_planning_iterations,
    )
    if summary.failed:
        raise SystemExit(1)


@lsp.command("train-network")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--save-dir", type=click.Path(file_okay=False), default=None,
              help="Output directory for weights and logs "
                   "[default: <data-dir>/training]")
@click.option("--network-filename", default="LSPFrontierNet.pt",
              show_default=True,
              help="Filename of the saved weights inside the save dir")
@click.option("--num-epochs", type=int, default=8, show_default=True,
              help="Number of passes over the training data")
@click.option("--batch-size", type=int, default=32, show_default=True,
              help="Number of data per training batch")
@click.option("--learning-rate", type=float, default=2e-3, show_default=True,
              help="Initial Adam learning rate")
@click.option("--learning-rate-decay", type=float, default=0.6,
              show_default=True, help="Learning-rate decay per epoch")
@click.option("--relative-positive-weight", type=float, default=2.0,
              show_default=True,
              help="Weight of positive (feasible) examples in the "
                   "feasibility cross-entropy")
@click.option("--val-fraction", type=float, default=0.1, show_default=True,
              help="Fraction of seed directories held out for validation")
@click.option("--num-workers", type=int, default=4, show_default=True,
              help="DataLoader worker processes")
@click.option("--device", default=None,
              help="Torch device (e.g. cuda, mps, cpu) [default: auto]")
@click.option("--seed", type=int, default=0, show_default=True,
              help="Seed for the train/val split and weight initialization")
@click.option("--log-interval", type=int, default=25, show_default=True,
              help="Print progress every this many training batches "
                   "(0 disables)")
def lsp_train_network(
    data_dir: str,
    save_dir: str | None,
    network_filename: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    learning_rate_decay: float,
    relative_positive_weight: float,
    val_fraction: float,
    num_workers: int,
    device: str | None,
    seed: int,
    log_interval: int,
) -> None:
    """Train the LSP frontier-statistics network on DATA_DIR.

    DATA_DIR is an experiment directory produced by `railroad lsp
    generate-data` (seed_* children) or a single run's data directory.
    Validation holds out whole seeds. The saved weights then guide
    planning:

        railroad example lsp-point-goal-nav --frontier-statistics
        learned --network-file <save-dir>/LSPFrontierNet.pt
    """
    from pathlib import Path

    from railroad.lsp.train import TrainConfig, train_network

    config = TrainConfig(
        data_dir=data_dir,
        save_dir=save_dir if save_dir is not None
        else Path(data_dir) / "training",
        network_filename=network_filename,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        relative_positive_weight=relative_positive_weight,
        val_fraction=val_fraction,
        num_workers=num_workers,
        device=device,
        seed=seed,
        log_interval=log_interval,
    )
    train_network(config)


# =============================================================================
# PDDL converter command group
# =============================================================================


@main.group()
def pddl() -> None:
    """Convert and run IPC PDDL/PPDDL problems (see railroad.pddl_converter)."""
    pass


@pddl.command("list")
@click.argument("collection")
def pddl_list(collection: str) -> None:
    """List domains available in COLLECTION (e.g. ipc-2000, ippc-2008)."""
    from railroad import pddl_converter as pc

    for domain in pc.list_domains(collection):
        click.echo(domain)


@pddl.command("clear-cache")
def pddl_clear_cache() -> None:
    """Delete cached downloads (directory listings and PDDL files).

    Listings are cached indefinitely, so upstream additions or renames are
    invisible until the cache is cleared.
    """
    import shutil

    from railroad.pddl_converter.download import cache_dir

    target = cache_dir()
    if target.exists():
        shutil.rmtree(target)
        click.echo(f"Removed {target}")
    else:
        click.echo(f"Nothing to remove ({target} does not exist)")


@pddl.command("run")
@click.option("--collection", default=None,
              help="Benchmark collection (e.g. ipc-2000, ippc-2008)")
@click.option("--domain", default=None, help="Domain name within the collection")
@click.option("--instance", type=click.IntRange(min=1), default=1,
              show_default=True, help="1-based instance index within the domain")
@click.option("--domain-file", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Local domain file (instead of --collection)")
@click.option("--problem-file", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Local problem file (instead of --collection)")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--max-steps", type=int, default=500, show_default=True)
@click.option("--max-iterations", type=int, default=4000, show_default=True,
              help="MCTS iterations per planning step")
@click.option("--planner", type=click.Choice(["mcts", "greedy"]),
              default="mcts", show_default=True,
              help="Action selection: full MCTS or greedy expected-heuristic")
@click.option("--verbose", is_flag=True, default=False,
              help="Print each executed action")
def pddl_run(collection: str | None, domain: str | None, instance: int,
             domain_file: str | None, problem_file: str | None,
             seed: int, max_steps: int, max_iterations: int,
             planner: str, verbose: bool) -> None:
    """Download (if needed), convert, and solve one PDDL instance."""
    from railroad import pddl_converter as pc

    if domain_file and problem_file:
        problem = pc.load_problem(domain_file, problem_file)
        source = problem_file
    elif collection and domain:
        fetched = pc.fetch_domain(collection, domain, max_instances=instance)
        if instance > len(fetched.instances):
            raise click.ClickException(
                f"{collection}/{domain} has only {len(fetched.instances)} instances"
            )
        instance_path = fetched.instances[instance - 1]
        problem = pc.load_problem(fetched.domain_for(instance_path), instance_path)
        source = str(instance_path)
    else:
        raise click.ClickException(
            "Provide either --collection and --domain, or --domain-file and --problem-file"
        )

    click.echo(f"source:  {source}")
    click.echo(f"problem: {problem.problem_name} (domain {problem.domain_name})")
    click.echo(f"metric:  {problem.metric}")
    click.echo(f"actions: {len(problem.ground_actions())} grounded")
    result = pc.solve(problem, seed=seed, max_steps=max_steps,
                      max_iterations=max_iterations, planner=planner,
                      verbose=verbose)
    if result.success:
        click.echo(f"SOLVED in {len(result.plan)} actions, "
                   f"cost/time {result.sim_time:g} "
                   f"({result.wall_time:.1f}s wall)")
        if not verbose:
            for step in result.plan:
                click.echo(f"  {step}")
    else:
        raise click.ClickException(f"FAILED: {result.failure_reason}")


@pddl.command("check")
@click.argument("collection")
@click.option("--instances", "n_instances", type=click.IntRange(min=1), default=1,
              show_default=True, help="Instances to try converting per domain")
@click.option("--ground", is_flag=True, default=False,
              help="Also ground actions (slower; catches grounding blowups)")
@click.option("--markdown", is_flag=True, default=False,
              help="Emit a markdown table (for the README)")
def pddl_check(collection: str, n_instances: int, ground: bool,
               markdown: bool) -> None:
    """Scan a collection and report which domains convert to railroad format."""
    from railroad import pddl_converter as pc

    rows: list[tuple[str, str, str]] = []
    for domain_name in pc.list_domains(collection):
        try:
            fetched = pc.fetch_domain(collection, domain_name,
                                      max_instances=n_instances)
        except Exception as exc:  # noqa: BLE001 - report and continue scanning
            rows.append((domain_name, "fetch-error", str(exc)[:80]))
            continue
        if not fetched.instances:
            rows.append((domain_name, "fetch-error", "no instances found"))
            continue
        status, note = "ok", ""
        for instance_path in fetched.instances:
            try:
                problem = pc.load_problem(
                    fetched.domain_for(instance_path), instance_path
                )
                if ground:
                    note = f"{len(problem.ground_actions())} actions"
            except pc.UnsupportedPDDLError as exc:
                status, note = "unsupported", exc.reason
                break
            except pc.PDDLParseError as exc:
                status, note = "parse-error", str(exc)[:80]
                break
            except Exception as exc:  # noqa: BLE001 - report and continue scanning
                status, note = "error", f"{type(exc).__name__}: {str(exc)[:70]}"
                break
        rows.append((domain_name, status, note))

    if markdown:
        click.echo("| Domain | Status | Notes |")
        click.echo("|---|---|---|")
        for name, status, note in rows:
            note = note.replace("|", "\\|")
            click.echo(f"| {name} | {status} | {note} |")
    else:
        width = max((len(r[0]) for r in rows), default=10)
        for name, status, note in rows:
            click.echo(f"{name:{width}}  {status:12}  {note}")
    n_ok = sum(1 for _, status, _ in rows if status == "ok")
    click.echo(f"\n{n_ok}/{len(rows)} domains convert ({collection}, "
               f"{n_instances} instance(s) each)")


def _make_example_command(name: str, info: ExampleInfo) -> None:
    """Create and register a click command for an example."""
    description = info["description"]
    options = info.get("options", [])

    @example.command(name, help=description)
    @click.pass_context
    def _run(ctx: click.Context, **kwargs: object) -> None:
        from railroad.examples import EXAMPLES

        example_info = EXAMPLES[name]
        click.echo(f"Running example: {name}")
        click.echo(f"  {example_info['description']}\n")
        example_fn = example_info["main"]
        example_fn(**kwargs)

    # Add example-specific options dynamically
    for opt in reversed(options):
        option_name = opt["name"]
        param_name = opt.get("param_name", option_name.lstrip("-").replace("-", "_"))
        if opt.get("is_flag", False):
            _run = click.option(option_name, param_name, is_flag=True, default=opt.get("default", False), help=opt.get("help", ""))(_run)
        else:
            extra_kwargs: dict[str, Any] = {}
            if "type" in opt:
                extra_kwargs["type"] = opt["type"]
            _run = click.option(option_name, param_name, default=opt.get("default"), show_default=True, help=opt.get("help", ""), **extra_kwargs)(_run)

    # Global plot/video options for every example command
    _run = click.option("--video-dpi", "video_dpi", type=int, default=150, show_default=True, help="Video resolution in dots per inch")(_run)
    _run = click.option("--video-fps", "video_fps", type=int, default=60, show_default=True, help="Video frames per second")(_run)
    _run = click.option("--save-video", "save_video", default=None, help="Save trajectory animation to file (e.g. out.mp4)")(_run)
    _run = click.option("--show-plot", "show_plot", is_flag=True, default=False, help="Show trajectory plot interactively")(_run)
    _run = click.option("--save-plot", "save_plot", default=None, help="Save trajectory plot to file (e.g. out.png)")(_run)
    # Option group panels (last applied = displayed first)
    _run = click.option_panel("Options", options=[opt["name"] for opt in options] + ["--help"])(_run)
    _run = click.option_panel("Plot/video options", options=["--save-plot", "--show-plot", "--save-video", "--video-fps", "--video-dpi"])(_run)


# Register each example as a direct subcommand
def _register_examples() -> None:
    from railroad.examples import EXAMPLES

    for name, info in EXAMPLES.items():
        _make_example_command(name, info)


_register_examples()


if __name__ == "__main__":
    main()

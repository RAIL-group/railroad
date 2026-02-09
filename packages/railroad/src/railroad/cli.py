"""Railroad command-line interface."""

from typing import Any

import rich_click as click

from railroad.examples import ExampleInfo


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
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
def benchmarks_run(
    case_filter: str | None,
    repeat_max: int | None,
    parallel: int | None,
    tags: tuple[str, ...] | None,
    dry_run: bool,
    mlflow_uri: str | None,
    include: tuple[str, ...],
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

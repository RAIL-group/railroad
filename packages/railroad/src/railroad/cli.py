"""Railroad command-line interface."""

import click


@click.group()
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


def _make_example_command(name: str, description: str) -> None:
    """Create and register a click command for an example."""

    @example.command(name, help=description)
    def _run() -> None:
        from railroad.examples import EXAMPLES

        example_info = EXAMPLES[name]
        click.echo(f"Running example: {name}")
        click.echo(f"  {example_info['description']}\n")
        example_fn = example_info["main"]
        example_fn()


# Register each example as a direct subcommand
def _register_examples() -> None:
    from railroad.examples import EXAMPLES

    for name, info in EXAMPLES.items():
        _make_example_command(name, info["description"])


_register_examples()


if __name__ == "__main__":
    main()

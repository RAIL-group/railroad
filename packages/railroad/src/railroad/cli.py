"""Railroad command-line interface."""

import click


@click.group()
@click.version_option(package_name="railroad")
def main() -> None:
    """Railroad: Multi-Robot Probabilistic Planning."""
    pass


@main.group()
def example() -> None:
    """Run example planning scenarios."""
    pass


@example.command("list")
def list_examples() -> None:
    """List available examples."""
    from railroad.examples import EXAMPLES

    click.echo("Available examples:\n")
    for name, info in EXAMPLES.items():
        click.echo(f"  {name:24} {info['description']}")
    click.echo("\nRun an example with: railroad example run <name>")


@example.command("run")
@click.argument("name")
def run_example(name: str) -> None:
    """Run an example by name."""
    from railroad.examples import EXAMPLES

    if name not in EXAMPLES:
        click.echo(f"Error: Unknown example '{name}'", err=True)
        click.echo("Run 'railroad example list' to see available examples", err=True)
        raise SystemExit(1)

    example_info = EXAMPLES[name]
    click.echo(f"Running example: {name}")
    click.echo(f"  {example_info['description']}\n")

    example_fn = example_info["main"]
    example_fn()


if __name__ == "__main__":
    main()

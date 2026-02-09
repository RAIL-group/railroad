"""Example planning scenarios for railroad.

This module provides runnable examples demonstrating various planning capabilities.
Examples can be run via the CLI: `railroad example run <name>`
"""

from typing import Any, Callable, Dict, List, TypedDict


class OptionInfo(TypedDict, total=False):
    """Information about a CLI option for an example."""

    name: str  # e.g., "--interruptible-moves"
    is_flag: bool  # True for boolean flags
    default: Any  # Default value
    type: Any  # Click type (e.g., int, str). Inferred from default if not set.
    help: str  # Help text
    param_name: str  # Python parameter name (e.g., "use_interruptible_moves")


class ExampleInfo(TypedDict, total=False):
    """Information about an example."""

    main: Callable[..., None]
    description: str
    options: List[OptionInfo]  # Optional CLI options


def _lazy_import(module_name: str, fn_name: str = "main") -> Callable[..., None]:
    """Lazy import to avoid loading all examples at startup."""

    def wrapper(**kwargs: Any) -> None:
        import importlib

        mod = importlib.import_module(f"railroad.examples.{module_name}")
        fn = getattr(mod, fn_name)
        return fn(**kwargs)

    return wrapper


def _procthor_available() -> bool:
    """Check if procthor dependencies are installed."""
    from railroad.environment.procthor import is_available

    return is_available()


GLOBAL_EXAMPLE_OPTIONS: List[OptionInfo] = [
    {"name": "--save-plot", "default": None, "help": "Save trajectory plot to file (e.g. out.png)", "param_name": "save_plot"},
    {"name": "--show-plot", "is_flag": True, "default": False, "help": "Show trajectory plot interactively", "param_name": "show_plot"},
    {"name": "--save-video", "default": None, "help": "Save trajectory animation to file (e.g. out.mp4)", "param_name": "save_video"},
    {"name": "--video-fps", "type": int, "default": 60, "help": "Video frames per second", "param_name": "video_fps"},
    {"name": "--video-dpi", "type": int, "default": 150, "help": "Video resolution in dots per inch", "param_name": "video_dpi"},
]

EXAMPLES: Dict[str, ExampleInfo] = {
    "clear-table": {
        "main": _lazy_import("clear_the_table"),
        "description": "Clear objects from a table (demonstrates negative goals)",
    },
    "multi-object-search": {
        "main": _lazy_import("multi_object_search"),
        "description": "Search for and collect multiple objects with multiple robots",
    },
    "find-and-move-couch": {
        "main": _lazy_import("find_and_move_couch"),
        "description": "Cooperative task requiring two robots (demonstrates wait operators)",
    },
    "heterogeneous-robots": {
        "main": _lazy_import("heterogeneous_robots"),
        "description": "Heterogeneous robots with different capabilities (drone, rover, crawler)",
        "options": [
            {
                "name": "--interruptible-moves",
                "is_flag": True,
                "default": False,
                "help": "Enable interruptible move actions",
                "param_name": "use_interruptible_moves",
            },
        ],
    },
}

# Only show procthor example if dependencies are installed
if _procthor_available():
    EXAMPLES["procthor-search"] = {
        "main": _lazy_import("procthor_search"),
        "description": "Multi-robot search in ProcTHOR 3D environment",
        "options": [
            {
                "name": "--seed",
                "type": int,
                "default": None,
                "help": "Scene seed (default: use hardcoded scene/objects)",
                "param_name": "seed",
            },
            {
                "name": "--num-objects",
                "default": 2,
                "help": "Number of objects to search for",
                "param_name": "num_objects",
            },
            {
                "name": "--num-robots",
                "default": 2,
                "help": "Number of robots",
                "param_name": "num_robots",
            },
        ],
    }


def _apply_options(cmd: Any, options: List["OptionInfo"]) -> Any:
    """Apply a list of OptionInfo entries as click options to a command."""
    import rich_click as click

    for opt in reversed(options):
        option_name = opt["name"]
        param_name = opt.get("param_name", option_name.lstrip("-").replace("-", "_"))
        if opt.get("is_flag", False):
            cmd = click.option(
                option_name, param_name,
                is_flag=True, default=opt.get("default", False),
                help=opt.get("help", ""),
            )(cmd)
        else:
            extra_kwargs: Dict[str, Any] = {}
            if "type" in opt:
                extra_kwargs["type"] = opt["type"]
            cmd = click.option(
                option_name, param_name,
                default=opt.get("default"), show_default=True,
                help=opt.get("help", ""), **extra_kwargs,
            )(cmd)
    return cmd



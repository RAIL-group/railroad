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
    "frontier-search": {
        "main": _lazy_import("frontier_search"),
        "description": "Explore unknown space and search discovered sites for objects",
        "options": [
            {
                "name": "--procthor",
                "is_flag": True,
                "default": False,
                "help": "Use ProcTHOR scene instead of synthetic grid",
                "param_name": "procthor",
            },
            {
                "name": "--seed",
                "type": int,
                "default": None,
                "help": "ProcTHOR scene seed (requires --procthor)",
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
                "default": 1,
                "help": "Number of robots",
                "param_name": "num_robots",
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
            {
                "name": "--estimate-object-find-prob",
                "is_flag": True,
                "default": False,
                "help": "Use learned model to estimate object find probabilities",
                "param_name": "estimate_object_find_prob",
            },
            {
                "name": "--nn-model-path",
                "type": str,
                "default": None,
                "help": ("Path to trained neural network model for estimating object find probabilities. "
                         "Defaults to the packaged ProcTHOR model if omitted."),
                "param_name": "nn_model_path",
            }
        ],
    }

"""Example planning scenarios for railroad.

This module provides runnable examples demonstrating various planning capabilities.
Examples can be run via the CLI: `railroad example run <name>`
"""

from typing import Callable, Dict, TypedDict


class ExampleInfo(TypedDict):
    """Information about an example."""

    main: Callable[[], None]
    description: str


def _lazy_import(module_name: str, fn_name: str = "main") -> Callable[[], None]:
    """Lazy import to avoid loading all examples at startup."""

    def wrapper() -> None:
        import importlib

        mod = importlib.import_module(f"railroad.examples.{module_name}")
        fn = getattr(mod, fn_name)
        return fn()

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
    },
}

# Only show procthor example if dependencies are installed
if _procthor_available():
    EXAMPLES["procthor-search"] = {
        "main": _lazy_import("procthor_search"),
        "description": "Multi-robot search in ProcTHOR 3D environment",
    }

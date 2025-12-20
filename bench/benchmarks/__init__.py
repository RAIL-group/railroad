"""
Benchmarks for PDDL planning system.

Import all benchmark modules to trigger @benchmark decorator registration.
Benchmarks are automatically registered when decorated.
"""

# Import all benchmark modules (triggers @benchmark decorators)
from . import basic_planning
from . import movie_night
from . import multi_object_search

__all__ = [
    "basic_planning",
    "movie_night",
    "multi_object_search",
]

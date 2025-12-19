"""
Benchmarks for PDDL planning system.

Import all benchmark modules to trigger @benchmark decorator registration.
Benchmarks are automatically registered when decorated.
"""

# Import all benchmark modules (triggers @benchmark decorators)
from . import basic_planning
from . import multi_robot

__all__ = [
    "basic_planning",
    "multi_robot",
]

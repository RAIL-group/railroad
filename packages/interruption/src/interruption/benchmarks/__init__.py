"""
Real-time task stream benchmarks for PDDL planning system.

Import all benchmark modules to trigger @benchmark decorator registration.
Benchmarks are automatically registered when decorated.
"""

# Import all benchmark modules (triggers @benchmark decorators)
from . import bench_interruption

__all__ = [
    "bench_interruption"
]

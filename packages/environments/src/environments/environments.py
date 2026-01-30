"""Environment implementations for robot simulation.

SimpleEnvironment has been moved to railroad.environment.
This module re-exports it for backward compatibility.
"""

from railroad.environment import SimpleEnvironment

__all__ = ["SimpleEnvironment"]

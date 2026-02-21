"""Experimental module containing older/alternative implementations.

This module contains implementations that are still functional but are not
the recommended approach for new code. They are preserved for backward
compatibility and for specific use cases.

Submodules:
- environment: Legacy environment interface (EnvironmentInterface, AbstractEnvironment, etc.)
- unknown_search: Frontier-based unknown-space exploration environment
"""

from . import environment

__all__ = ["environment", "unknown_search"]

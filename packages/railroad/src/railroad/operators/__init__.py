"""Operator constructors for PDDL planning.

This module provides operator constructors for common robot planning actions.
All operators are re-exported here for convenient imports.

Usage:
    from railroad.operators import construct_move_operator, construct_pick_operator

Naming convention:
- Default operators (without suffix) are non-blocking
- `_blocking` suffix indicates operators with just-moved/just-picked/just-placed preconditions
- `_constrained` suffix indicates operators with additional constraints

Available operators:
- Movement: construct_move_operator, construct_move_operator_blocking,
            construct_move_visited_operator, construct_move_visited_operator_constrained
- Search: construct_search_operator, construct_search_and_pick_operator
- Pick: construct_pick_operator, construct_pick_operator_blocking
- Place: construct_place_operator, construct_place_operator_blocking
- Wait: construct_wait_operator, construct_no_op_operator

Navigation/frontier operators are in ``railroad.environment.unknown_space.operators``.
"""

from .core import (
    # Move operators
    construct_move_operator,
    construct_move_operator_blocking,
    construct_move_visited_operator,
    construct_move_visited_operator_constrained,
    # Search operators
    construct_search_operator,
    construct_search_and_pick_operator,
    # Pick operators
    construct_pick_operator,
    construct_pick_operator_blocking,
    # Place operators
    construct_place_operator,
    construct_place_operator_blocking,
    # Wait operators
    construct_wait_operator,
    construct_no_op_operator,
)

from ._utils import (
    Numeric,
    numeric,
    _to_numeric,
    Num,
    OptNumeric,
)

__all__ = [
    # Move operators
    "construct_move_operator",
    "construct_move_operator_blocking",
    "construct_move_visited_operator",
    "construct_move_visited_operator_constrained",
    # Search operators
    "construct_search_operator",
    "construct_search_and_pick_operator",
    # Pick operators
    "construct_pick_operator",
    "construct_pick_operator_blocking",
    # Place operators
    "construct_place_operator",
    "construct_place_operator_blocking",
    # Wait operators
    "construct_wait_operator",
    "construct_no_op_operator",
    # Utilities
    "Numeric",
    "numeric",
    "_to_numeric",
    "Num",
    "OptNumeric",
]

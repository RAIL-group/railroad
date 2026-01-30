"""Operator constructors - backward compatibility shim.

This module re-exports operators from railroad.operators for backward compatibility.
New code should import directly from railroad.operators.

Naming aliases for backward compatibility:
- construct_move_operator_nonblocking -> construct_move_operator
- construct_pick_operator_nonblocking -> construct_pick_operator
- construct_place_operator_nonblocking -> construct_place_operator
"""

# Re-export from railroad.operators
from railroad.operators import (
    # Move operators - import with aliases for clarity
    construct_move_operator as construct_move_operator_nonblocking,
    construct_move_operator_blocking,
    construct_move_visited_operator,
    construct_move_visited_operator_constrained,
    # Search operators
    construct_search_operator,
    construct_search_and_pick_operator,
    # Pick operators - import with aliases for clarity
    construct_pick_operator as construct_pick_operator_nonblocking,
    construct_pick_operator_blocking,
    # Place operators - import with aliases for clarity
    construct_place_operator as construct_place_operator_nonblocking,
    construct_place_operator_blocking,
    # Wait operators
    construct_wait_operator,
    construct_no_op_operator,
    # Utilities
    _make_callable,
    _invert_prob,
)

# BACKWARD COMPATIBILITY: In the old environments.operators module:
# - construct_move_operator was the BLOCKING version (with just-moved precondition)
# - construct_pick_operator was the BLOCKING version (with just-picked precondition)
# - construct_place_operator was the BLOCKING version (with just-placed precondition)
# We preserve this behavior for backward compatibility.
construct_move_operator = construct_move_operator_blocking
construct_pick_operator = construct_pick_operator_blocking
construct_place_operator = construct_place_operator_blocking

__all__ = [
    # Move operators
    "construct_move_operator",
    "construct_move_operator_nonblocking",  # Alias for backward compatibility
    "construct_move_operator_blocking",
    "construct_move_visited_operator",
    "construct_move_visited_operator_constrained",
    # Search operators
    "construct_search_operator",
    "construct_search_and_pick_operator",
    # Pick operators
    "construct_pick_operator",
    "construct_pick_operator_nonblocking",  # Alias for backward compatibility
    "construct_pick_operator_blocking",
    # Place operators
    "construct_place_operator",
    "construct_place_operator_nonblocking",  # Alias for backward compatibility
    "construct_place_operator_blocking",
    # Wait operators
    "construct_wait_operator",
    "construct_no_op_operator",
    # Utilities
    "_make_callable",
    "_invert_prob",
]
